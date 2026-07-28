#!/usr/bin/env bash
#
# Fetch the public infant-pose / autism-behaviour datasets into $ROOT/datasets.
#
# NEEDS OUTBOUND INTERNET. Same constraint as 10_fetch_youtube.sh: BigPurple
# compute nodes are usually firewalled, so this either runs on a login node or
# on data_mover. The preflight below fails in 10 seconds rather than hanging.
#
# Resumable: every dataset writes a .complete marker when it finishes, and every
# transfer uses curl -C- / git pull, so re-running is cheap and safe to kill.
#
# THE DATASETS SPLIT IN TWO, AND NO SCRIPT CAN CHANGE THAT:
#
#   AUTO   open API or public git -- fully handled here.
#          babypose, chambers, tariq, dream, aggpose(code), syrip(best effort)
#
#   MANUAL gated behind a request form, a signed data-use agreement, or simply
#          never released. Automating these is not a hard problem, it is an
#          impossible one -- the bytes are not on the public internet. For each,
#          the script writes the exact page, contact and required step into
#          MANUAL_ACCESS.md and moves on.
#          mini_rgbd, ssbd, mmdb, 3d_ad, infantnet
#
# A NOTE ON WHAT "TARIQ" AND "CHAMBERS" ACTUALLY CONTAIN: neither release ships
# raw identifiable video. Chambers is pose data + metadata + YouTube URLs (the
# videos you fetch yourself, exactly like the youtube_dataset); Tariq is
# feature-rating CSVs + classifier code. Do not plan on pixels from either.
#
# Usage:
#   scripts/12_fetch_datasets.sh                      # everything automatable
#   scripts/12_fetch_datasets.sh --list               # show the plan, fetch nothing
#   scripts/12_fetch_datasets.sh --only babypose,dream
#   scripts/12_fetch_datasets.sh --dry-run            # print URLs, download nothing
#   scripts/12_fetch_datasets.sh --no-extract         # keep archives zipped
#   scripts/12_fetch_datasets.sh --delete-archives    # unpack, then drop the zips
#
set -euo pipefail

ROOT="${ROOT:-/gpfs/scratch/sd6701/personal/ibdp}"
DATA="${DATA:-$ROOT/datasets}"
LOG_DIR="$DATA/_logs"
MANUAL_DOC="$DATA/MANUAL_ACCESS.md"
STATUS="$DATA/STATUS.tsv"

ALL_AUTO="syrip babypose aggpose chambers tariq dream"
ALL_MANUAL="mini_rgbd ssbd mmdb 3d_ad infantnet"

ONLY=""
DRY=0
EXTRACT=1
DEL_ARCHIVES=0
LIST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)       ONLY="${2:?--only needs a comma-separated list}"; shift 2 ;;
    --only=*)     ONLY="${1#*=}"; shift ;;
    --dry-run)    DRY=1; shift ;;
    --no-extract) EXTRACT=0; shift ;;
    # Halves the on-disk footprint. Safe: an archive is only removed after it
    # unpacked cleanly, and .complete stops the next run re-downloading it.
    --delete-archives) DEL_ARCHIVES=1; shift ;;
    --list)       LIST=1; shift ;;
    -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
for tool in curl python3 git; do
  command -v "$tool" >/dev/null || { echo "ERROR: $tool not on PATH" >&2; exit 1; }
done
# unzip is not optional in practice: SyRIP, BabyPose, Chambers and Tariq all
# ship zips, so without it you get ~9GB of archives and no usable dataset.
if [[ $EXTRACT -eq 1 ]] && ! command -v unzip >/dev/null; then
  echo "WARN: unzip not on PATH -- archives will be downloaded but NOT extracted." >&2
  echo "      conda install -c conda-forge unzip   (or rerun with --no-extract to silence)" >&2
fi
command -v wget  >/dev/null || echo "WARN: wget missing; the SyRIP directory mirror will be skipped." >&2

if [[ $LIST -eq 0 && $DRY -eq 0 ]]; then
  # Fail fast on a firewalled node instead of burning walltime on a hung curl.
  if ! curl -sSf -m 10 -o /dev/null https://zenodo.org; then
    echo "ERROR: no route to zenodo.org. You are probably on a firewalled compute node." >&2
    echo "       Run this on a login node, or: sbatch --partition=data_mover scripts/12_fetch_datasets.sbatch" >&2
    exit 1
  fi
fi

mkdir -p "$DATA" "$LOG_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Per-dataset provenance file. Written every run so a directory found on disk in
# six months still says where it came from and which paper it belongs to.
sources() {
  local dir="$1"; shift
  mkdir -p "$dir"
  printf '%s\n' "$@" > "$dir/SOURCES.txt"
}

have() { [[ -f "$DATA/$1/.complete" ]]; }
mark_done() { touch "$DATA/$1/.complete"; }

# curl with resume. -C- continues a partial file, so a killed job does not
# restart a 1.3GB transfer from zero. --fail so a 404 is an error, not an HTML
# error page silently saved as your dataset.
get() {
  local url="$1" out="$2"
  if [[ $DRY -eq 1 ]]; then echo "  [dry] $url -> $out"; return 0; fi
  mkdir -p "$(dirname "$out")"
  curl -L --fail --retry 5 --retry-delay 5 -C - --progress-bar -o "$out" "$url" \
    || { echo "  FAILED: $url" >&2; return 1; }
}

# Zenodo: the REST API lists every file in a record with a direct link, so we
# never scrape HTML and never hardcode a filename that the depositor may rename.
zenodo() {
  local rec="$1" dest="$2"
  echo "  zenodo record $rec"
  local json; json="$(curl -sSf --retry 3 "https://zenodo.org/api/records/$rec")" \
    || { echo "  FAILED: zenodo API $rec" >&2; return 1; }
  local lines; lines="$(printf '%s' "$json" | python3 -c '
import json, sys
for f in json.load(sys.stdin).get("files", []):
    name = f.get("key") or f.get("filename") or ""
    links = f.get("links") or {}
    url = links.get("self") or links.get("download") or ""
    print("\t".join([name, url, str(f.get("size", 0))]))
')"
  [[ -z "$lines" ]] && { echo "  no files listed in record $rec" >&2; return 1; }
  while IFS=$'\t' read -r name link size; do
    [[ -z "$name" ]] && continue
    printf '    %s (%s bytes)\n' "$name" "$size"
    get "$link" "$dest/$name" || return 1
  done <<< "$lines"
}

# Figshare: same idea. supplied_md5 is published, so we verify -- a truncated
# 1.3GB download that unzips "mostly fine" is the worst possible failure mode.
figshare() {
  local art="$1" dest="$2" host="${3:-api.figshare.com}"
  echo "  figshare article $art"
  local json; json="$(curl -sSf --retry 3 "https://$host/v2/articles/$art/files")" \
    || { echo "  FAILED: figshare API $art" >&2; return 1; }
  local lines; lines="$(printf '%s' "$json" | python3 -c '
import json, sys
for f in json.load(sys.stdin):
    print("\t".join([f.get("name", ""), f.get("download_url", ""),
                     f.get("supplied_md5") or ""]))
')"
  [[ -z "$lines" ]] && { echo "  no files listed in article $art" >&2; return 1; }
  while IFS=$'\t' read -r name url md5; do
    [[ -z "$name" ]] && continue
    echo "    $name"
    get "$url" "$dest/$name" || return 1
    if [[ $DRY -eq 0 && -n "$md5" && "$md5" != "None" ]] && command -v md5sum >/dev/null; then
      local got; got="$(md5sum "$dest/$name" | cut -d' ' -f1)"
      if [[ "$got" != "$md5" ]]; then
        echo "    MD5 MISMATCH ($got != $md5) -- deleting, rerun to retry" >&2
        rm -f "$dest/$name"; return 1
      fi
    fi
  done <<< "$lines"
}

clone() {
  local url="$1" dir="$2"
  if [[ $DRY -eq 1 ]]; then echo "  [dry] git clone $url -> $dir"; return 0; fi
  if [[ -d "$dir/.git" ]]; then
    echo "  git pull $(basename "$dir")"
    git -C "$dir" pull --ff-only --quiet || echo "  WARN: pull failed, keeping existing checkout" >&2
  else
    echo "  git clone $url"
    # --depth 1: we want the files, not eight years of history.
    git clone --depth 1 --quiet "$url" "$dir" || { echo "  FAILED: clone $url" >&2; return 1; }
  fi
}

# Unpack in place, once. The .unpacked stamp is per-archive so adding a new file
# to a dataset later does not re-extract the ones already done -- and an
# interrupted unzip has no stamp, so a rerun redoes it (unzip -o overwrites).
#
# TWO ROUNDS, not one: several of these releases are archives-of-archives
# (SyRIP.zip contains per-split zips; the BabyPose deposits nest per-subject).
# Round 1 unpacks the outer layer, round 2 catches what round 1 revealed. The
# stamps make round 2 a no-op for anything already done, so this is cheap.
#
# Space: extracting alongside the archive roughly DOUBLES the footprint --
# ~9GB of downloads becomes ~18GB+ on scratch. Pass --delete-archives to drop
# each archive once it has unpacked cleanly.
extract_all() {
  local dir="$1"
  [[ $EXTRACT -eq 1 && $DRY -eq 0 ]] || return 0

  local round z t out
  for round in 1 2; do
    if command -v unzip >/dev/null; then
      while IFS= read -r -d '' z; do
        [[ -f "$z.unpacked" ]] && continue
        echo "  unzip $(basename "$z")"
        out="${z%.zip}"
        if unzip -q -o "$z" -d "$out"; then
          touch "$z.unpacked"
          [[ $DEL_ARCHIVES -eq 1 ]] && { rm -f "$z"; echo "    removed archive"; }
        else
          echo "    WARN: unzip failed on $(basename "$z") -- left in place" >&2
        fi
      done < <(find "$dir" -maxdepth 4 -name '*.zip' -print0 2>/dev/null)
    fi

    while IFS= read -r -d '' t; do
      [[ -f "$t.unpacked" ]] && continue
      echo "  untar $(basename "$t")"
      # Strip the FULL suffix: %%.t* would turn meta.test.tar.gz into "meta".
      out="${t%.tar.gz}"; out="${out%.tgz}"
      mkdir -p "$out"
      if tar -xzf "$t" -C "$out"; then
        touch "$t.unpacked"
        [[ $DEL_ARCHIVES -eq 1 ]] && { rm -f "$t"; echo "    removed archive"; }
      else
        echo "    WARN: untar failed on $(basename "$t") -- left in place" >&2
      fi
    done < <(find "$dir" -maxdepth 4 \( -name '*.tar.gz' -o -name '*.tgz' \) -print0 2>/dev/null)
  done
}

# ---------------------------------------------------------------------------
# AUTO datasets
# ---------------------------------------------------------------------------

fetch_syrip() {
  local d="$DATA/syrip"
  sources "$d" \
    "SyRIP -- Synthetic and Real Infant Pose" \
    "paper : https://arxiv.org/abs/2010.06100" \
    "code  : https://github.com/ostadabbas/Infant-Pose-Estimation" \
    "data  : https://coe.northeastern.edu/Research/AClab/SyRIP/"
  # The code repo is small and always public, and its README carries the
  # authoritative, current download instructions -- grab it first so that even
  # if the mirror below finds nothing, you have the recipe on disk.
  clone https://github.com/ostadabbas/Infant-Pose-Estimation "$d/code" || return 1

  # The AClab URL is a plain Apache directory index, so we read it and take the
  # TOP-LEVEL FILES only -- currently README.md, SyRIP.zip (312M),
  # SyRIP_Posture.zip (153M) and syrip_for_train.zip (607M). The index also
  # exposes annotations/ and images/ subtrees, which are the same content loose
  # on disk; recursing into them would roughly double the transfer for nothing.
  # Set SYRIP_FULL=1 if you specifically want the loose trees as well.
  #
  # Parsing the index rather than hardcoding names means a renamed or added
  # archive is picked up automatically. If the lab ever swaps this for a landing
  # page or a Drive link, no file links are found and we fail loudly instead of
  # leaving an empty directory that looks like a dataset.
  local idx="https://coe.northeastern.edu/Research/AClab/SyRIP/"
  local files
  files="$(curl -sSf --retry 3 "$idx" \
           | grep -o 'href="[^"?/][^"]*"' | sed 's/href="//; s/"$//' \
           | grep -v '/$' | sort -u)" || { echo "  FAILED: cannot read $idx" >&2; return 1; }
  if [[ -z "$files" ]]; then
    echo "  NOTE: no files in the AClab index -- follow the link in $d/code/README.md" >&2
    return 1
  fi
  local f
  for f in $files; do
    echo "    $f"
    get "$idx$f" "$d/raw/$f" || return 1
  done

  if [[ "${SYRIP_FULL:-0}" == "1" ]]; then
    if [[ $DRY -eq 1 ]]; then
      echo "  [dry] wget -r ${idx}images/ ${idx}annotations/"
    elif command -v wget >/dev/null; then
      echo "  SYRIP_FULL=1: mirroring loose images/ and annotations/"
      # -np stays inside the SyRIP subtree; -nH --cut-dirs=3 strips
      # /Research/AClab/SyRIP so the tree lands flat under raw/.
      wget -q --show-progress -r -np -nH --cut-dirs=3 -R "index.html*" \
           -P "$d/raw" "${idx}images/" "${idx}annotations/" \
        || echo "  WARN: mirror incomplete" >&2
    else
      echo "  WARN: SYRIP_FULL=1 but wget is missing; skipped" >&2
    fi
  fi
  extract_all "$d"
}

fetch_babypose() {
  local d="$DATA/babypose"
  sources "$d" \
    "babyPose -- depth-camera infant joint annotations (Data in Brief)" \
    "paper : https://pmc.ncbi.nlm.nih.gov/articles/PMC7551984/" \
    "data  : https://doi.org/10.5281/zenodo.3891404 (records 3891404, 5336836)"
  # Two records: 3891404 is the original deposit, 5336836 the later version.
  # Both are kept -- they are cited separately in the literature.
  zenodo 3891404 "$d/zenodo_3891404" || return 1
  zenodo 5336836 "$d/zenodo_5336836" || return 1
  extract_all "$d"
}

fetch_aggpose() {
  local d="$DATA/aggpose"
  sources "$d" \
    "AggPose / InfantPose -- Deep Aggregation ViT for infant pose (IJCAI 2022)" \
    "paper : https://arxiv.org/abs/2205.05277" \
    "code  : https://github.com/PediaMedAI/AggPose" \
    "NOTE  : repo ships code + model-weight links. The full InfantPose image" \
    "        corpus is NOT in the repo -- request it from the authors."
  clone https://github.com/PediaMedAI/AggPose "$d/code" || return 1
}

fetch_chambers() {
  local d="$DATA/chambers"
  sources "$d" \
    "Chambers et al. -- Computer Vision to Automatically Assess Infant Neuromotor Risk" \
    "paper : https://pmc.ncbi.nlm.nih.gov/articles/PMC8011647/" \
    "data  : https://doi.org/10.6084/m9.figshare.8161430  (~1.33 GB)" \
    "code  : https://github.com/cchamber/Infant_movement_assessment" \
    "NOTE  : pose data + metadata + YouTube URLs. No raw video in the release." \
    "        Feed the URL list to yt-dlp the same way 10_fetch_youtube.sh does."
  figshare 8161430 "$d/figshare" || return 1
  clone https://github.com/cchamber/Infant_movement_assessment "$d/code" || return 1
  extract_all "$d"
}

fetch_tariq() {
  local d="$DATA/tariq"
  sources "$d" \
    "Tariq et al. -- Mobile Detection of Autism Through ML on Home Video (PLOS Med 2018)" \
    "paper : https://doi.org/10.1371/journal.pmed.1002705" \
    "data  : https://plos.figshare.com/articles/dataset/.../7389038" \
    "code  : https://github.com/qandeelt/Tariq-Wall-2018-PLOS-MEDICINE" \
    "NOTE  : feature-rating CSVs + classifiers. Raw home videos are identifiable" \
    "        and are NOT distributed."
  # plos.figshare.com is a figshare instance; the public v2 API serves it too.
  figshare 7389038 "$d/figshare" || return 1
  clone https://github.com/qandeelt/Tariq-Wall-2018-PLOS-MEDICINE "$d/code" || return 1
  extract_all "$d"
}

fetch_dream() {
  local d="$DATA/dream"
  sources "$d" \
    "DREAM -- robot-enhanced therapy for ASD, 61 children, 3D pose + gaze" \
    "paper : https://doi.org/10.1371/journal.pone.0236939" \
    "code  : https://github.com/dream2020/data" \
    "data  : https://doi.org/10.5878/17p8-6k13 (SND catalogue snd1156-1)" \
    "NOTE  : the GitHub repo is the openly released JSON dataset. The SND record" \
    "        is the archival copy and may require an order form."
  clone https://github.com/dream2020/data "$d/data" || return 1
}

# ---------------------------------------------------------------------------
# MANUAL datasets -- write the exact ask, do not pretend to download
# ---------------------------------------------------------------------------

manual_entry() {
  local name="$1" title="$2" why="$3"; shift 3
  local d="$DATA/$name"
  sources "$d" "$title" "$@"
  {
    echo "## $name -- $title"
    echo
    echo "**Why it is not scripted:** $why"
    echo
    for line in "$@"; do echo "- $line"; done
    echo
    echo "Once you have the files, drop them under \`$d/\` and \`touch $d/.complete\`."
    echo
  } >> "$MANUAL_DOC"
  echo "  MANUAL -- see $MANUAL_DOC"
}

fetch_mini_rgbd()  { manual_entry mini_rgbd "MINI-RGB-D (Fraunhofer IOSB)" \
  "Fraunhofer releases it on request; the download link is issued per requester, not published." \
  "paper   : https://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Hesse_Computer_Vision_for_Medical_Infant_Motion_Analysis_State_of_the_ECCVW_2018_paper.pdf" \
  "page    : https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html" \
  "shortlink: http://s.fhg.de/mini-rgbd" \
  "action  : use the request form / contact on the IOSB motion-analysis page"; }

fetch_ssbd()       { manual_entry ssbd "SSBD -- Self-Stimulatory Behaviours Dataset" \
  "Distributed as annotation XML plus YouTube URLs from the maintainer's page; many source videos have since been removed, so a blind mirror gives a misleading hit rate." \
  "paper  : https://openaccess.thecvf.com/content_iccv_workshops_2013/W22/papers/Rajagopalan_Self-Stimulatory_Behaviours_in_2013_ICCV_paper.pdf" \
  "page   : https://rolandgoecke.net/research/datasets/ssbd/" \
  "action : download the annotation archive from that page, then resolve the video URLs with yt-dlp and expect substantial link rot"; }

fetch_mmdb()       { manual_entry mmdb "MMDB -- Multimodal Dyadic Behavior Dataset (Georgia Tech)" \
  "Human-subjects video behind a signed data-use agreement. No automated path exists, by design." \
  "paper  : https://openaccess.thecvf.com/content_cvpr_2013/papers/Rehg_Decoding_Childrens_Social_2013_CVPR_paper.pdf" \
  "page   : https://cbs.ic.gatech.edu/mmdb/dataset.php" \
  "action : submit the data-use agreement on that page; expect institutional sign-off (NYU Langone IRB/legal) to be required"; }

fetch_3d_ad()      { manual_entry 3d_ad "3D-AD -- 3D Autism Dataset (Kinect)" \
  "No active public download page was found; the IEEE record is the only verified primary source." \
  "paper : https://doi.org/10.1109/AVSS.2017.8078544" \
  "hal   : https://amu.hal.science/hal-03605592" \
  "action: email the corresponding author listed on the HAL record"; }

fetch_infantnet()  { manual_entry infantnet "InfantNet -- infant body pose and shape" \
  "Paper is on OpenReview; no separate public release of the images / SMIL annotations was found." \
  "paper : https://openreview.net/forum?id=e80U2cGtv4" \
  "action: contact the authors via the OpenReview thread and ask for the release status"; }

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

TARGETS="$ALL_AUTO $ALL_MANUAL"
if [[ -n "$ONLY" ]]; then
  TARGETS="$(echo "$ONLY" | tr ',' ' ')"
  for t in $TARGETS; do
    case " $ALL_AUTO $ALL_MANUAL " in
      *" $t "*) ;;
      *) echo "unknown dataset: $t" >&2
         echo "known: $ALL_AUTO $ALL_MANUAL" >&2; exit 2 ;;
    esac
  done
fi

if [[ $LIST -eq 1 ]]; then
  printf '%-12s %-8s %s\n' DATASET MODE STATE
  for ds in $TARGETS; do
    mode=MANUAL; case " $ALL_AUTO " in *" $ds "*) mode=AUTO ;; esac
    state=pending; have "$ds" && state=complete
    printf '%-12s %-8s %s\n' "$ds" "$mode" "$state"
  done
  exit 0
fi

# Rewritten from scratch each run so it reflects only what is still outstanding.
{
  echo "# Datasets requiring manual access"
  echo
  echo "Generated by scripts/12_fetch_datasets.sh. Everything here is gated behind"
  echo "a request form, a data-use agreement, or was never publicly released."
  echo
} > "$MANUAL_DOC"

echo "root: $DATA"
echo "targets: $TARGETS"
echo

: > "$STATUS"
printf '#dataset\tmode\tstate\tbytes\n' >> "$STATUS"

for ds in $TARGETS; do
  mode=MANUAL; case " $ALL_AUTO " in *" $ds "*) mode=AUTO ;; esac
  echo "=== $ds [$mode] ==="

  if have "$ds" && [[ $DRY -eq 0 ]]; then
    echo "  already complete (rm $DATA/$ds/.complete to force)"
    state=complete
  else
    log="$LOG_DIR/$ds.log"
    # set -e does not apply inside an if-condition, so each fetch_* function
    # guards its own critical steps with `|| return 1`. A dataset that fails is
    # recorded and the run continues -- one dead mirror must not cost you the
    # other ten.
    if fetch_"$ds" 2>&1 | tee -a "$log"; then
      state=ok
      [[ $mode == AUTO && $DRY -eq 0 ]] && { mark_done "$ds"; state=complete; }
    else
      state=FAILED
      echo "  see $log" >&2
    fi
  fi

  # du -sk, not -sb: -b is GNU-only and silently yields nothing elsewhere.
  bytes=0
  if [[ -d "$DATA/$ds" ]]; then
    kb="$(du -sk "$DATA/$ds" 2>/dev/null | cut -f1)"
    [[ -n "${kb:-}" ]] && bytes=$(( kb * 1024 ))
  fi
  printf '%s\t%s\t%s\t%s\n' "$ds" "$mode" "$state" "$bytes" >> "$STATUS"
  echo
done

echo "---------------------------------------------------------------"
column -t -s$'\t' "$STATUS" 2>/dev/null || cat "$STATUS"
echo
du -sh "$DATA" 2>/dev/null || true
echo
echo "status : $STATUS"
echo "manual : $MANUAL_DOC   <- read this, it is the rest of the work"
