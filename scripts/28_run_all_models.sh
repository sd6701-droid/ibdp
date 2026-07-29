#!/usr/bin/env bash
#
# Annotate ONE video with several models back to back, then compare them.
#
#   scripts/28_run_all_models.sh 0HkcGRBsPUM
#   scripts/28_run_all_models.sh 0HkcGRBsPUM --limit-segments 3   # quick shakedown
#   MODELS="InternVL3-38B Qwen2.5-VL-72B-Instruct" scripts/28_run_all_models.sh VID
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

VIDEO="${1:-}"
shift 2>/dev/null || true
EXTRA=("$@")              # everything else is forwarded to the python script

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

echo
echo "video  : $VIDEO"
echo "models : $MODELS"
echo "audio  : $([[ "$AUDIO" == "1" ]] && echo "transcript in prompt (Omni: native)" || echo "off (video only)")"
echo "outdir : $OUTDIR"
echo "extra  : ${EXTRA[*]:-(none)}"
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

  python scripts/26_describe_segments_hf.py \
      --model "$ROOT/models/$M" \
      --only "$VIDEO" \
      --outdir "$OUTDIR" \
      --resume \
      "${AUDIO_ARGS[@]}" \
      "${EXTRA[@]}" 2>&1 | tee "$log"

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
for a in "${EXTRA[@]:-}"; do
  [[ "$a" == "--wandb" ]] && CMP_WANDB=(--wandb)
done

python scripts/27_compare_models.py \
    --outdir "$OUTDIR" --video "$VIDEO" --models "$joined" --csv "$csv" \
    "${CMP_WANDB[@]}" \
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
