#!/usr/bin/env bash
#
# Annotate ONE video with several models back to back, then compare them.
#
#   scripts/28_run_all_models.sh 0HkcGRBsPUM
#   scripts/28_run_all_models.sh 0HkcGRBsPUM --limit-segments 3   # quick shakedown
#   MODELS="InternVL3-38B Qwen2.5-VL-72B-Instruct" scripts/28_run_all_models.sh VID
#   SCENES=1 scripts/28_run_all_models.sh 8yDn1uFbs4s   # annotate scripts/32's
#                                                       # shot-boundary splits
#                                                       # instead of a 10s grid
#   SCENES=1 scripts/28_run_all_models.sh all           # every video that has
#                                                       # splits, one W&B run
#                                                       # per model, NO compare
#
# SEQUENTIAL ON PURPOSE. 145GB + 156GB of weights do not coexist on 240GB of
# VRAM, and backgrounding them with & buys nothing anyway: one model already
# saturates the cards. Each python process exits before the next starts, which
# is also what frees the VRAM -- there is no in-process unload to get wrong.
#
# One model failing does not stop the rest. A missing checkpoint or an OOM on
# the 78B should still leave you with the other two and a comparison.
#
# Resume is per (model, segment), so re-running this after a walltime kill picks
# up exactly where each model stopped.
#
set -uo pipefail          # NOT -e: a failing model is handled, not fatal

ROOT="${ROOT:-/gpfs/scratch/sd6701/personal/ibdp}"
OUTDIR="${OUTDIR:-$ROOT/outputs}"
LOGDIR="$OUTDIR/logs"

# Ordered smallest-first: if something in the harness is wrong you find out on
# the 61GB model in a couple of minutes, not 40 minutes into loading the 78B.
#
#   Qwen3-VL-30B-A3B-Instruct   ~61GB   MoE, only ~3.3B params active per token
#   Qwen3-Omni-30B-A3B-Instruct ~71GB   MoE, NATIVE AUDIO + video
#   Qwen3-VL-32B-Instruct       ~64GB   DENSE, same generation as the baseline
#   InternVL3-38B               ~76GB   dense
#   Qwen2.5-VL-72B-Instruct    ~145GB   dense, older generation than Qwen3-VL
#   InternVL3-78B              ~156GB   dense
#
# The 30B-A3B is the incumbent -- the one the existing corpus was annotated
# with. It belongs in the comparison as the BASELINE: the question is not just
# which of the new models is best, but whether any of them beats what you
# already have enough to justify re-annotating 11.6k segments.
#
# Qwen3-Omni sits directly after it ON PURPOSE. Same family, same MoE size, so
# the pair isolates ONE variable: does hearing the clip beat only seeing it?
# Any other ordering confounds that with a size or generation difference.
#
# Qwen3-VL-32B comes next, and is the OTHER controlled pair: same generation and
# roughly the same nominal size as the 30B-A3B baseline, but DENSE -- 32B active
# per token against 3.3B. If dense-vs-MoE is worth anything on this task, that
# is the pair that shows it, uncontaminated by a generation gap the way
# Qwen2.5-VL-72B is.
MODELS="${MODELS:-Qwen3-VL-30B-A3B-Instruct Qwen3-Omni-30B-A3B-Instruct Qwen3-VL-32B-Instruct InternVL3-38B Qwen2.5-VL-72B-Instruct InternVL3-78B}"

# --- Audio ------------------------------------------------------------------
# AUDIO=1 gives the four video-only models a Whisper transcript in the prompt.
# Qwen3-Omni ignores it -- it gets the waveform itself.
#
# WHY THIS IS A SEPARATE SWITCH rather than always-on: --transcribe changes
# prompt_sha, so audio and video-only records never resume into each other.
# Flipping it mid-corpus is safe; it just re-annotates rather than silently
# mixing two modalities in one file.
AUDIO="${AUDIO:-0}"

# --- Scene splits ------------------------------------------------------------
# SCENES=1 annotates the shot-boundary splits made by scripts/32_split_scenes.py
# (outputs/scenes/<video>/split_NN/clip.mp4) instead of cutting a fixed 10s
# grid over the full video: one segment per source clip, boundaries on the
# compilation's own edit points. SCENES=<path> points at a different splits
# tree. Default off, so existing grid workflows are untouched.
#
# Records carry segmentation="scenes" and NEVER resume into grid records (or
# vice versa) even though the prompt -- and therefore prompt_sha -- is
# identical: the two modes number different stretches of the same video with
# overlapping indices. The compare step below is pinned to the same mode for
# the same reason.
#
# Splits are annotated WHOLE -- one forward pass per source clip. Only a clip
# longer than 15s (--clip-max-sec) is windowed into 10s (--seconds) pieces.
# MAX_SECONDS still works if set explicitly ("only splits starting inside the
# first N seconds"), but unlike grid mode it defaults to OFF here: the splits
# are already few and short, so a full pass IS the cheap pass.
SCENES="${SCENES:-0}"
SCENES_DIR=""
if [[ "$SCENES" == "1" ]]; then
  SCENES_DIR="$ROOT/outputs/scenes"
elif [[ "$SCENES" != "0" ]]; then
  SCENES_DIR="$SCENES"
fi

# --- Shakedown window ------------------------------------------------------
# Annotate only the first MAX_SECONDS of the video. 20s at the default 10s
# window is 2 segments per model -- under a minute of inference each, against
# ~8 minutes for a full 22-segment pass.
#
# DEFAULTED TO 20 WHILE WE ARE STILL TESTING. This is a trap if you forget it:
# a "full run" that quietly stops at 20 seconds looks like a complete corpus.
# Hence the banner below, and MAX_SECONDS=0 to turn it off:
#
#   MAX_SECONDS=0 scripts/28_run_all_models.sh VIDEO_ID
#
# Segment indices match a full run, so the capped records are the genuine first
# two and a later MAX_SECONDS=0 run resumes over the top of them.
#
# SCENES MODE DEFAULTS TO 0 (whole video): scene splits are already the cheap
# unit of work -- a 220s video is ~26 clips of ~8s -- and capping them to the
# first 20s would annotate two clips and look like a finished run. An explicit
# MAX_SECONDS=N still wins in either mode.
if [[ -n "$SCENES_DIR" ]]; then
  MAX_SECONDS="${MAX_SECONDS:-0}"
else
  MAX_SECONDS="${MAX_SECONDS:-20}"
fi

VIDEO="${1:-}"
shift 2>/dev/null || true
EXTRA=("$@")              # everything else is forwarded to the python script

# --- W&B: on by default ------------------------------------------------------
# --wandb is appended automatically and WANDB_MODE defaults to online, so the
# command line stays short: one W&B run per model (see WANDB_SEPARATE_RUNS
# below) plus one compare run, each carrying its results table. WANDB=0 turns
# all logging off; WANDB_TABLES_ONLY=1 below still strips --wandb back out of
# the model runs and keeps only the compare run.
WANDB="${WANDB:-1}"
if [[ "$WANDB" == "1" ]]; then
  _seen_wandb=0
  for a in ${EXTRA[@]+"${EXTRA[@]}"}; do
    [[ "$a" == "--wandb" ]] && _seen_wandb=1
  done
  [[ "$_seen_wandb" == "1" ]] || EXTRA+=(--wandb)
  export WANDB_MODE="${WANDB_MODE:-online}"
fi

# --- Fresh W&B runs, always --------------------------------------------------
# Every invocation mints a unique tag, appended to every W&B run name
# (<model>--<video>--<tag>, compare--<video>--<tag>), and any resume state
# inherited from the shell is dropped -- a sweep NEVER attaches to or resumes a
# previous W&B run. RUN_TAG=... overrides the tag if you want to name a sweep.
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
export RUN_TAG
unset WANDB_RUN_ID WANDB_RESUME

# --- W&B: tables only -------------------------------------------------------
# WANDB_TABLES_ONLY=1 gives you ONE W&B run containing ONLY tables -- the
# combined all_models table (model as the first column, one row per model x
# segment), health, agreement, and the wide per-segment view. No charts.
#
# How: the annotator is what logs per-segment metrics (the charts), so --wandb
# is STRIPPED from what the models see, and the comparison step is forced to
# log instead. The models run clean; the one compare run carries the tables.
#
# Costs the GPU-utilisation charts too -- they come from the same per-model
# runs. Want both? Use --wandb WITHOUT this switch and ignore the charts.
WANDB_TABLES_ONLY="${WANDB_TABLES_ONLY:-0}"
if [[ "$WANDB_TABLES_ONLY" == "1" ]]; then
  # ${arr[@]+"${arr[@]}"} not "${arr[@]:-}": the latter turns an empty array
  # into ONE EMPTY-STRING ARG, which argparse rejects as "unrecognized
  # arguments:" when it reaches the python script.
  declare -a _KEEP=()
  for a in ${EXTRA[@]+"${EXTRA[@]}"}; do
    [[ "$a" == "--wandb" ]] || _KEEP+=("$a")
  done
  EXTRA=(${_KEEP[@]+"${_KEEP[@]}"})
fi

[[ -n "$VIDEO" ]] || {
  echo "usage: $0 <video_id> [extra args forwarded to 26_describe_segments_hf.py]" >&2
  echo "       e.g. $0 0HkcGRBsPUM --seconds 10 --max-new-tokens 256" >&2
  exit 2; }

cd "$(dirname "$0")/.."   # repo root, so scripts/... resolves
mkdir -p "$LOGDIR" || {
  echo "cannot create $LOGDIR -- is \$ROOT right for this machine?" >&2
  exit 1; }

# ---------------------------------------------------------------------------
# Preflight -- everything that can be known before a 150GB load is checked now
# ---------------------------------------------------------------------------
command -v python >/dev/null || { echo "no python -- conda activate ibdp" >&2; exit 1; }

python - <<'PY' || exit 1
import sys
try:
    import torch
except ImportError:
    sys.exit("torch not importable -- wrong env? conda activate ibdp")
if not torch.cuda.is_available():
    sys.exit("no GPU visible -- this needs a compute node with A100s, not a login node.")
n = torch.cuda.device_count()
free, total = torch.cuda.mem_get_info(0)
print(f"gpus: {n} x {torch.cuda.get_device_name(0)} "
      f"({total/1e9:.0f}GB each, {n*total/1e9:.0f}GB total)")
if n * total / 1e9 < 170:
    print("WARN: <170GB total VRAM. The 72B/78B will not fit; expect OOM on those.",
          file=sys.stderr)
PY

missing=0
for M in $MODELS; do
  [[ -d "$ROOT/models/$M" ]] || { echo "MISSING checkpoint: $ROOT/models/$M" >&2; missing=1; }
done
[[ $missing -eq 0 ]] || { echo "fix the above, or set MODELS= to what you have" >&2; exit 1; }

# A DIRECTORY IS NOT A CHECKPOINT. A killed download leaves one that passes the
# -d test above at 54GB of 64GB, then dies inside from_pretrained on a missing
# shard -- as the 32B did, AFTER two other models had already run. The weight
# index names every required shard, so verify all of them up front: this loop is
# ordered smallest-first precisely so failures are cheap, and a checkpoint that
# cannot load is the cheapest failure of all.
python - "$ROOT/models" $MODELS <<'PY' || exit 1
import json, sys
from pathlib import Path

root, names = Path(sys.argv[1]), sys.argv[2:]
bad = []
for name in names:
    d = root / name
    idx = d / "model.safetensors.index.json"
    if idx.is_file():
        try:
            want = set(json.loads(idx.read_text()).get("weight_map", {}).values())
        except Exception as e:
            bad.append(f"{name}: unreadable weight index ({e})")
            continue
        miss = sorted(f for f in want if not (d / f).is_file())
        if miss:
            bad.append(f"{name}: {len(miss)} of {len(want)} shards missing "
                       f"(first: {miss[0]})")
    elif not any(d.glob("*.safetensors")) and not any(d.glob("*.bin")):
        bad.append(f"{name}: no weight files at all")

if bad:
    print("INCOMPLETE CHECKPOINT(S) -- refusing to start:", file=sys.stderr)
    for b in bad:
        print(f"  {b}", file=sys.stderr)
    print("\n  Finish the download from a LOGIN node (finished shards are kept):",
          file=sys.stderr)
    print("    sbatch scripts/13_fetch_models.sbatch --only <key>", file=sys.stderr)
    print("  Or drop it for now:  MODELS=\"...\" scripts/28_run_all_models.sh ...",
          file=sys.stderr)
    sys.exit(1)
print(f"checkpoints: {len(names)} verified, all shards present")
PY

# InternVL's vendored code hard-imports these. Checking here rather than 20
# minutes into a weight load -- and pip cannot fix it on a compute node anyway,
# which is exactly why it is worth failing early and loudly.
if echo "$MODELS" | grep -qi internvl; then
  python -c "import einops, timm" 2>/dev/null || {
    echo "ERROR: InternVL needs einops + timm, which are not installed." >&2
    echo "       Compute nodes have no internet, so install from a LOGIN node:" >&2
    echo "         ssh bigpurple-ln3 && conda activate ibdp && pip install einops 'timm>=0.9'" >&2
    exit 1; }
fi

# Same reasoning for Omni: a different utils package, and a transformers new
# enough to know the architecture at all. Both are unfixable from here.
if echo "$MODELS" | grep -qi omni; then
  python - <<'PY' || exit 1
import sys
try:
    import qwen_omni_utils, soundfile        # noqa: F401
except ImportError as e:
    sys.exit(f"ERROR: Qwen3-Omni needs qwen-omni-utils + soundfile ({e}).\n"
             "       Compute nodes have no internet -- install from a LOGIN node:\n"
             "         ssh bigpurple-ln3 && conda activate ibdp && "
             "pip install qwen-omni-utils soundfile librosa")
try:
    from transformers import Qwen3OmniMoeForConditionalGeneration  # noqa: F401
except ImportError:
    import transformers
    sys.exit(f"ERROR: transformers {transformers.__version__} does not have "
             "Qwen3OmniMoeForConditionalGeneration.\n"
             "       ssh bigpurple-ln3 && conda activate ibdp && "
             "pip install -U 'transformers>=4.57'")
PY
fi

# VIDEO=all: every video with splits, in one pass per model. Scenes mode only
# -- "all" on the 10s grid would be the entire 195-video corpus, which is an
# array-job problem, not a for-loop one.
if [[ "$VIDEO" == "all" && -z "$SCENES_DIR" ]]; then
  echo "ERROR: 'all' needs SCENES mode -- SCENES=1 $0 all" >&2
  exit 2
fi

# Splits must exist BEFORE a 60-150GB model is loaded to discover they don't.
if [[ -n "$SCENES_DIR" ]]; then
  pat="$SCENES_DIR/$VIDEO/split_*/clip.mp4"
  [[ "$VIDEO" == "all" ]] && pat="$SCENES_DIR/*/split_*/clip.mp4"
  if ! compgen -G "$pat" > /dev/null; then
    echo "ERROR: SCENES mode, but no splits matching $pat" >&2
    echo "       Cut them first (seconds of GPU, minutes of ffmpeg):" >&2
    echo "         sbatch scripts/32_split_scenes.sbatch --only $VIDEO --wav --thumbs" >&2
    exit 1
  fi
  n_splits=$(compgen -G "$pat" | wc -l | tr -d ' ')
  n_vids=$(compgen -G "$pat" | sed 's|/split_[^/]*/clip.mp4||' | sort -u | wc -l | tr -d ' ')
fi

# --transcribe loads Whisper from disk like everything else here.
if [[ "$AUDIO" == "1" ]]; then
  [[ -d "$ROOT/models/whisper-large-v3" ]] || {
    echo "ERROR: AUDIO=1 needs $ROOT/models/whisper-large-v3" >&2
    echo "       fetch it on a LOGIN node:" >&2
    echo "         scripts/13_fetch_models.sh --only whisper-large-v3" >&2
    exit 1; }
fi

export HF_HUB_OFFLINE=1                 # weights are local; fail fast, don't hang
export FORCE_QWENVL_VIDEO_READER=torchcodec
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Some shells on this cluster carry an LD_PRELOADed NVBLAS shim with NO CPU
# BLAS fallback configured ("CPU Blas library need to be provided"). A numpy
# BLAS call routed to that missing fallback jumps through a NULL pointer:
#   Caught signal 11 ... address not mapped to object at address (nil)
# It killed the Omni runs specifically because audio feature extraction is
# the one numpy-BLAS-heavy stage. The sbatch wrappers already unset it; this
# script is also run INTERACTIVELY, so unset it here too.
unset LD_PRELOAD

# ---------------------------------------------------------------------------
# W&B: ONE run PER MODEL (the default), named after the model, each carrying
# that model's results table and system metrics -- plus the compare run at the
# end. That is 7 runs for a 6-model sweep, grouped by video id in the UI.
#
# WANDB_SEPARATE_RUNS=0 collapses them into one shared run instead: exporting a
# shared run id makes every python process attach to the same run -- scripts/26
# sees WANDB_RUN_ID, prefixes its metrics with the model name, and drops its
# explicit step counter. The id carries RUN_TAG, so even the shared run is
# unique per invocation and never resumes a previous sweep's; WANDB_RESUME=allow
# only lets the six processes of THIS sweep attach to it one after another.
# ---------------------------------------------------------------------------
if [[ "${WANDB_SEPARATE_RUNS:-1}" != "1" ]]; then
  for a in ${EXTRA[@]+"${EXTRA[@]}"}; do
    if [[ "$a" == "--wandb" ]]; then
      export WANDB_RUN_ID="allmodels-${VIDEO}-${RUN_TAG}"
      export WANDB_RESUME=allow
      echo "wandb  : single shared run '$WANDB_RUN_ID' for all models"
      break
    fi
  done
fi

echo
echo "video  : $VIDEO"
if [[ "$VIDEO" == "all" ]]; then
  echo "segments: $n_splits scene splits across $n_vids videos under $SCENES_DIR"
elif [[ -n "$SCENES_DIR" ]]; then
  echo "segments: $n_splits scene splits from $SCENES_DIR/$VIDEO"
else
  echo "segments: fixed 10s grid over the full video"
fi
echo "models : $MODELS"
echo "audio  : $([[ "$AUDIO" == "1" ]] && echo "transcript in prompt (Omni: native)" || echo "off (video only)")"
if [[ "${MAX_SECONDS:-0}" != "0" ]]; then
  echo "window : FIRST ${MAX_SECONDS}s ONLY -- shakedown run, NOT the full video"
  echo "         (MAX_SECONDS=0 for the whole thing)"
else
  echo "window : whole video"
fi
echo "outdir : $OUTDIR"
echo "extra  : ${EXTRA[*]:-(none)}"
echo "wandb  : tag $RUN_TAG -- fresh runs, one per model + compare"
echo

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
declare -a OK_MODELS=() BAD_MODELS=()
t_all=$SECONDS

for M in $MODELS; do
  log="$LOGDIR/${VIDEO}_${M}.log"
  echo "==============================================================="
  echo "=== $M"
  echo "=== log: $log"
  echo "==============================================================="
  t0=$SECONDS

  # Qwen3-Omni takes the audio natively, so handing it a transcript as well
  # would let it answer from text it was supposed to hear -- and the run would
  # no longer measure what native audio is worth.
  declare -a AUDIO_ARGS=()
  if [[ "$AUDIO" == "1" && "$M" != *Omni* ]]; then
    AUDIO_ARGS=(--transcribe)
  fi

  # Shakedown window, when MAX_SECONDS is non-zero. Built per iteration rather
  # than once, so `set -u` cannot trip over an unset array on the first model.
  declare -a LIMIT_ARG=()
  if [[ "${MAX_SECONDS:-0}" != "0" ]]; then
    LIMIT_ARG=(--max-seconds "$MAX_SECONDS")
  fi

  declare -a RESUME_ARG=(--resume)
  if [[ "${NO_RESUME:-0}" == "1" ]]; then
    RESUME_ARG=()
  fi

  declare -a SCENES_ARG=()
  if [[ -n "$SCENES_DIR" ]]; then
    SCENES_ARG=(--scenes "$SCENES_DIR")
  fi

  # VIDEO=all: no --only, so scripts/26 walks every video dir under --scenes.
  declare -a ONLY_ARG=(--only "$VIDEO")
  [[ "$VIDEO" == "all" ]] && ONLY_ARG=()

  python scripts/26_describe_segments_hf.py \
      --model "$ROOT/models/$M" \
      ${ONLY_ARG[@]+"${ONLY_ARG[@]}"} \
      --outdir "$OUTDIR" \
      ${SCENES_ARG[@]+"${SCENES_ARG[@]}"} \
      ${RESUME_ARG[@]+"${RESUME_ARG[@]}"} \
      ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"} \
      ${AUDIO_ARGS[@]+"${AUDIO_ARGS[@]}"} \
      ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$log"

  # PIPESTATUS[0], not $?: $? is tee's status, which is 0 even when python died.
  if [[ "${PIPESTATUS[0]}" -eq 0 ]]; then
    OK_MODELS+=("$M")
    echo "--- $M done in $(( (SECONDS - t0) / 60 ))m $(( (SECONDS - t0) % 60 ))s"
  else
    BAD_MODELS+=("$M")
    echo "--- $M FAILED after $(( (SECONDS - t0) / 60 ))m -- see $log" >&2
  fi
  echo
done

echo "==============================================================="
echo "ran ${#OK_MODELS[@]} ok, ${#BAD_MODELS[@]} failed, in $(( (SECONDS - t_all) / 60 ))m"
[[ ${#BAD_MODELS[@]} -gt 0 ]] && echo "failed: ${BAD_MODELS[*]}" >&2

# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
# 27 compares ONE video's records; an all-videos sweep skips it rather than
# pretending. The per-video command is printed so it is a paste, not a hunt.
#
# BUT a sweep that carried --splits is a PROBE, and a probe's deliverable is
# the bundle: one JSON with every model's answer for the chosen split(s), side
# by side, logged to W&B (table + versioned artifact) like the model runs.
if [[ "$VIDEO" == "all" ]]; then
  SPLITS_VAL=""
  _prev=""
  for a in ${EXTRA[@]+"${EXTRA[@]}"}; do
    [[ "$_prev" == "--splits" ]] && SPLITS_VAL="$a"
    [[ "$a" == --splits=* ]] && SPLITS_VAL="${a#*=}"
    _prev="$a"
  done
  if [[ -n "$SPLITS_VAL" && -n "$SCENES_DIR" ]]; then
    VIDS=$(ls -d "$SCENES_DIR"/*/ 2>/dev/null | xargs -n1 basename | paste -sd,)
    declare -a PROBE_WANDB=()
    [[ "$WANDB" == "1" ]] && PROBE_WANDB=(--wandb)
    echo
    echo "=== probe bundle ==="
    python scripts/33_sample_outputs.py \
        --videos "$VIDS" \
        --split "$SPLITS_VAL" \
        --out "$OUTDIR/probe_${RUN_TAG}.json" \
        ${PROBE_WANDB[@]+"${PROBE_WANDB[@]}"}
  else
    echo
    echo "no comparison for an all-videos sweep. Per video, when you want one:"
    echo "  python scripts/27_compare_models.py --outdir $OUTDIR --video <VIDEO_ID> \\"
    echo "      --segmentation scenes --csv $OUTDIR/compare_<VIDEO_ID>.csv"
  fi
  exit 0
fi
if [[ ${#OK_MODELS[@]} -lt 2 ]]; then
  echo "fewer than 2 models produced output -- nothing to compare." >&2
  exit 1
fi

csv="$OUTDIR/compare_${VIDEO}.csv"
txt="$OUTDIR/compare_${VIDEO}.txt"
echo
echo "=== comparison ==="
# Only the models that actually succeeded: passing a failed one would abort the
# comparison over a file that was never written.
joined="$(IFS=,; echo "${OK_MODELS[*]}")"

# --wandb is aimed at the PYTHON annotator, so it arrives in EXTRA. The compare
# step is a separate process and would not see it -- forward it explicitly, or
# the models land in W&B and the comparison that explains them does not.
declare -a CMP_WANDB=()
for a in ${EXTRA[@]+"${EXTRA[@]}"}; do
  [[ "$a" == "--wandb" ]] && CMP_WANDB=(--wandb)
done
# Tables-only mode stripped --wandb from EXTRA above, so force the compare
# step's logging here -- it is the only thing that logs in that mode.
[[ "$WANDB_TABLES_ONLY" == "1" ]] && CMP_WANDB=(--wandb)

# Pinned to THIS sweep's segmentation, not left to 27's majority vote: with
# grid records already on disk for the same video, "most common" would be the
# OLD mode until a scenes sweep out-produces it -- and the comparison would
# quietly describe the wrong run.
python scripts/27_compare_models.py \
    --outdir "$OUTDIR" --video "$VIDEO" --models "$joined" --csv "$csv" \
    --segmentation "$([[ -n "$SCENES_DIR" ]] && echo scenes || echo fixed)" \
    ${CMP_WANDB[@]+"${CMP_WANDB[@]}"} \
    2>&1 | tee "$txt"

echo
echo "report : $txt"
echo "table  : $csv"

# ---------------------------------------------------------------------------
# W&B: offline runs are inert files until synced, and this node cannot do it.
# ---------------------------------------------------------------------------
if compgen -G "$OUTDIR/wandb/offline-run-*" > /dev/null; then
  n_runs=$(compgen -G "$OUTDIR/wandb/offline-run-*" | wc -l | tr -d ' ')
  echo
  echo "wandb  : $n_runs offline run(s) not yet uploaded."
  echo "         From a LOGIN node (this one has no internet):"
  echo "           ssh bigpurple-ln3 && conda activate ibdp"
  echo "           wandb login                      # once"
  echo "           wandb sync $OUTDIR/wandb/offline-run-*"
fi
