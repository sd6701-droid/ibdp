#!/usr/bin/env bash
#
# Pull model weights into $ROOT/models.
#
# MUST run on a LOGIN NODE (bigpurple-ln*). Compute nodes have no outbound
# internet -- the same reason 10_fetch_youtube.sh is not an sbatch job, and the
# reason every sbatch here exports HF_HUB_OFFLINE=1.
#
# Resumable twice over: hf download resumes a partial file by itself, and a
# finished repo gets a .complete marker so a rerun skips it outright. Kill it
# whenever; rerun to continue.
#
# DISK IS THE REAL CONSTRAINT. The three models below are ~377GB together, on
# top of the ~61GB Qwen3-VL-30B-A3B-Instruct already sitting there. The script
# refuses to start unless the filesystem has room for everything selected, with
# 10% headroom -- because a truncated shard does not announce itself, it just
# fails at load time hours later with an opaque safetensors error.
#
# Usage:
#   scripts/13_fetch_models.sh --list
#   scripts/13_fetch_models.sh --only qwen25-vl-72b
#   scripts/13_fetch_models.sh                        # everything in the registry
#   scripts/13_fetch_models.sh --dry-run
#
# Run it under tmux. 377GB is not a foreground task:
#   tmux new -s models
#   ./scripts/13_fetch_models.sh
#   # detach ctrl-b then d ; reattach: tmux attach -t models
#
set -euo pipefail

ROOT="${ROOT:-/gpfs/scratch/sd6701/personal/ibdp}"
MODELS="${MODELS:-$ROOT/models}"
LOG_DIR="$MODELS/_logs"

# key|repo_id|local dirname|approx GB|A100-80s needed at bf16|note
REGISTRY="
qwen25-vl-72b|Qwen/Qwen2.5-VL-72B-Instruct|Qwen2.5-VL-72B-Instruct|145|4|qwen_vl_utils works as-is
internvl3-78b|OpenGVLab/InternVL3-78B|InternVL3-78B|156|4|needs trust_remote_code + new video path
internvl3-38b|OpenGVLab/InternVL3-38B|InternVL3-38B|76|2|needs trust_remote_code + new video path
whisper-large-v3|openai/whisper-large-v3|whisper-large-v3|3|1|audio only; scripts/31
qwen3-omni-30b|Qwen/Qwen3-Omni-30B-A3B-Instruct|Qwen3-Omni-30B-A3B-Instruct|71|2|NATIVE AUDIO+VIDEO; needs qwen-omni-utils
"

ONLY=""; DRY=0; LIST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)    ONLY="${2:?--only needs a comma-separated list of keys}"; shift 2 ;;
    --only=*)  ONLY="${1#*=}"; shift ;;
    --dry-run) DRY=1; shift ;;
    --list)    LIST=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

row_for() { printf '%s\n' "$REGISTRY" | awk -F'|' -v k="$1" '$1==k{print; exit}'; }
keys()    { printf '%s\n' "$REGISTRY" | awk -F'|' 'NF>1{print $1}'; }

# ---------------------------------------------------------------------------
# Which CLI: `hf` is current, `huggingface-cli` is the deprecated alias. Both
# take the same download args, so pick whichever exists rather than making the
# user care.
# ---------------------------------------------------------------------------
#
# NB: written as if/elif, NOT `command -v hf && HF_BIN=hf`. Under `set -e` that
# one-liner form exits the whole script when the command is simply absent --
# silently, before the friendly "install huggingface_hub" message below can run.
HF_BIN=""
if command -v hf >/dev/null 2>&1; then
  HF_BIN="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_BIN="huggingface-cli"
fi

TARGETS="$(keys)"
if [[ -n "$ONLY" ]]; then
  TARGETS="$(echo "$ONLY" | tr ',' ' ')"
  for t in $TARGETS; do
    [[ -n "$(row_for "$t")" ]] || { echo "unknown model key: $t" >&2
                                    echo "known: $(keys | tr '\n' ' ')" >&2; exit 2; }
  done
fi

if [[ $LIST -eq 1 ]]; then
  printf '%-16s %-38s %7s %6s  %s\n' KEY REPO GB GPUS STATE
  for k in $TARGETS; do
    IFS='|' read -r _ repo dir gb gpus _ <<< "$(row_for "$k")"
    state=pending; [[ -f "$MODELS/$dir/.complete" ]] && state=complete
    printf '%-16s %-38s %7s %6s  %s\n' "$k" "$repo" "$gb" "$gpus" "$state"
  done
  exit 0
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
[[ -n "$HF_BIN" ]] || {
  echo "ERROR: neither 'hf' nor 'huggingface-cli' on PATH." >&2
  echo "       conda activate ibdp && pip install -U huggingface_hub" >&2
  exit 1; }

# The sbatch files export this and it is inherited if you sourced one. Offline
# mode makes hf download fail with a confusing 'not found in cache' rather than
# saying 'you told me not to use the network'.
if [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
  echo "NOTE: HF_HUB_OFFLINE=1 is set -- unsetting it for this download." >&2
  unset HF_HUB_OFFLINE
fi

# HF_HOME still matters: hf writes lock/metadata under it. Keep it on gpfs, not
# in $HOME, which has a small quota.
export HF_HOME="${HF_HOME:-$ROOT/.hf}"

if [[ $DRY -eq 0 ]] && ! curl -sSf -m 10 -o /dev/null https://huggingface.co; then
  echo "ERROR: no route to huggingface.co. You are on a compute node." >&2
  echo "       ssh to a login node (bigpurple-ln3) and run this there." >&2
  exit 1
fi

mkdir -p "$MODELS" "$LOG_DIR"

# ---- space check ----------------------------------------------------------
need_gb=0
for k in $TARGETS; do
  IFS='|' read -r _ _ dir gb _ _ <<< "$(row_for "$k")"
  [[ -f "$MODELS/$dir/.complete" ]] && continue      # already have it
  need_gb=$(( need_gb + gb ))
done
# 10% headroom: hf stages incomplete files alongside the finished ones, so peak
# usage sits above the sum of the final sizes.
need_gb=$(( need_gb + need_gb / 10 ))

# `|| true` on both: --output=avail is GNU-only, and under `set -e` + pipefail a
# failing df inside a command substitution takes the whole script down before
# the portable fallback below ever runs.
avail_gb="$(df -BG --output=avail "$MODELS" 2>/dev/null | tail -1 | tr -dc '0-9' || true)"
if [[ -z "${avail_gb:-}" ]]; then
  avail_gb="$(df -Pk "$MODELS" 2>/dev/null | awk 'NR==2{print int($4/1048576)}' || true)"
fi
# Still nothing (unreadable mount): say so rather than comparing against "" and
# letting the guard silently pass.
if [[ -z "${avail_gb:-}" ]]; then
  echo "WARN: cannot read free space on $MODELS -- skipping the space check." >&2
  avail_gb=-1
fi

echo "models dir : $MODELS"
echo "need       : ~${need_gb} GB (incl. 10% headroom)"
echo "available  : ${avail_gb} GB"
if [[ $DRY -eq 0 && "$need_gb" -gt 0 && "$avail_gb" -ge 0 && "$avail_gb" -lt "$need_gb" ]]; then
  echo >&2
  echo "ERROR: not enough space. Free some, or fetch fewer models with --only." >&2
  echo "       A run that dies mid-shard leaves a file that LOOKS present and" >&2
  echo "       fails at load time instead." >&2
  exit 1
fi
echo

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
for k in $TARGETS; do
  IFS='|' read -r _ repo dir gb gpus note <<< "$(row_for "$k")"
  dest="$MODELS/$dir"
  echo "=== $k  ($repo, ~${gb}GB, ${gpus}x A100-80 at bf16) ==="
  echo "    $note"

  if [[ -f "$dest/.complete" ]]; then
    echo "    already complete (rm $dest/.complete to force)"
    echo; continue
  fi

  if [[ $DRY -eq 1 ]]; then
    echo "    [dry] $HF_BIN download $repo --local-dir $dest"
    echo; continue
  fi

  # --local-dir puts real files in the destination rather than cache symlinks,
  # so $MODELS is self-contained and survives an HF_HOME cleanup.
  if $HF_BIN download "$repo" --local-dir "$dest" 2>&1 | tee -a "$LOG_DIR/$k.log"; then
    touch "$dest/.complete"
    echo "    done: $(du -sh "$dest" | cut -f1)"
  else
    echo "    FAILED -- see $LOG_DIR/$k.log" >&2
    echo "    gated repo? accept the licence on https://huggingface.co/$repo" >&2
    echo "    then: $HF_BIN auth login" >&2
  fi
  echo
done

echo "---------------------------------------------------------------"
du -sh "$MODELS"/*/ 2>/dev/null | sort -h || true
echo
df -h "$MODELS" | tail -1
