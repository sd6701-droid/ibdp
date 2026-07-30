#!/usr/bin/env bash
#
# Probe ONE split of EVERY video through EVERY model, then bundle all the
# answers into a single JSON.
#
#   scripts/34_run_split_all_models.sh split_10
#   MODELS="Qwen3-VL-30B-A3B-Instruct" scripts/34_run_split_all_models.sh split_07
#
# This is the targeted cousin of 28_run_all_models.sh: instead of a corpus
# pass, each model annotates exactly one split per video (5 clips, seconds of
# GPU each) -- the "what does every model say about THIS clip" experiment.
# The result lands in ONE file:
#
#   $ROOT/outputs/probe_<split>_all_models.json
#
# NEEDS A GPU ALLOCATION that fits the biggest model in MODELS -- 3x A100 for
# the 72B/78B (with the --max-frames 16 cap this passes), 1 is enough if you
# override MODELS to the small ones. Run inside an interactive srun session or
# wrap in sbatch.
#
# W&B, SAME SHAPE AS 28_run_all_models.sh: online by default, one SEPARATE
# run per model (named <model>--<group>--<tag>, tagged uniquely per
# invocation so two probes of the same split never collide or resume into
# each other), plus one probe run at the end carrying the video x model table
# and the bundled JSON as a versioned artifact. WANDB=0 turns all of it off;
# WANDB_MODE=offline for a node with no internet route.
# No resume -- the point of a probe is a FRESH answer; records still append
# to new annotations_*.jsonl files, so nothing already on disk is touched,
# and 33 picks the newest record per (model, segment) when bundling.
set -uo pipefail          # NOT -e: one failing model should not kill the rest

ROOT="${ROOT:-/gpfs/scratch/sd6701/personal/ibdp}"
SCENES_DIR="${SCENES_DIR:-$ROOT/outputs/scenes}"

SPLIT="${1:?usage: $0 split_NN [extra args forwarded to 26]}"
shift 2>/dev/null || true
EXTRA=("$@")

# Same roster and the same smallest-first order as 28: harness mistakes cost
# minutes on the 30B, not an hour into the 78B.
MODELS="${MODELS:-Qwen3-VL-30B-A3B-Instruct Qwen3-Omni-30B-A3B-Instruct Qwen3-VL-32B-Instruct InternVL3-38B Qwen2.5-VL-72B-Instruct InternVL3-78B}"

# --- W&B: mirrored from 28_run_all_models.sh ---------------------------------
WANDB="${WANDB:-1}"
declare -a WANDB_ARG=()
if [[ "$WANDB" == "1" ]]; then
  WANDB_ARG=(--wandb)
  export WANDB_MODE="${WANDB_MODE:-online}"
fi
# Fresh runs, always: a unique tag lands in every run name via scripts/26,
# and any resume state inherited from the shell is dropped -- a probe NEVER
# attaches to a previous sweep's runs.
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
export RUN_TAG
unset WANDB_RUN_ID WANDB_RESUME

# WRONG-ENV GUARD. This was first run from the vitpose env (python 3.9):
# scripts/26 is 3.10+ syntax and needs the VLM stack, so every model "failed"
# with a TypeError before touching a GPU. Catch that in one line instead.
python - <<'PY' || { echo "       fix: conda activate ibdp" >&2; exit 1; }
import sys
if sys.version_info < (3, 10):
    sys.exit(f"ERROR: python {sys.version.split()[0]} -- scripts/26 needs the "
             f"ibdp env (3.12), not this one.")
try:
    import transformers  # noqa: F401
except ImportError:
    sys.exit("ERROR: transformers not importable -- this is not the ibdp env.")
PY

if ! compgen -G "$SCENES_DIR/*/$SPLIT/clip.mp4" > /dev/null; then
  echo "ERROR: no video has a $SPLIT under $SCENES_DIR" >&2
  echo "       (splits are zero-padded: split_07, not split_7)" >&2
  exit 1
fi
n_vids=$(compgen -G "$SCENES_DIR/*/$SPLIT/clip.mp4" | wc -l | tr -d ' ')
echo "probing $SPLIT of $n_vids video(s) with: $MODELS"
echo

declare -a OK=() BAD=()
for M in $MODELS; do
  echo "=== $M ==="
  python scripts/26_describe_segments_hf.py \
      --model "$ROOT/models/$M" \
      --scenes "$SCENES_DIR" \
      --splits "$SPLIT" \
      --max-new-tokens 512 \
      --max-frames 16 \
      ${WANDB_ARG[@]+"${WANDB_ARG[@]}"} \
      ${EXTRA[@]+"${EXTRA[@]}"}
  if [[ $? -eq 0 ]]; then OK+=("$M"); else BAD+=("$M"); echo "--- $M FAILED" >&2; fi
  echo
done

echo "==============================================================="
echo "annotated: ${#OK[@]} models ok, ${#BAD[@]} failed"
[[ ${#BAD[@]} -gt 0 ]] && echo "failed: ${BAD[*]}" >&2

# Every video dir that has this split, comma-joined for 33.
VIDS=$(compgen -G "$SCENES_DIR/*/$SPLIT/clip.mp4" \
       | sed "s|$SCENES_DIR/||; s|/$SPLIT/clip.mp4||" | paste -sd,)

python scripts/33_sample_outputs.py \
    --videos "$VIDS" \
    --split "$SPLIT" \
    --out "$ROOT/outputs/probe_${SPLIT}_all_models.json" \
    ${WANDB_ARG[@]+"${WANDB_ARG[@]}"}
