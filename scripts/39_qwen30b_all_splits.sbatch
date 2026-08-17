#!/bin/bash
#SBATCH --job-name=ibdp-30b-splits
#SBATCH --partition=gl40s_long       # L40S nodes (8 per node); _long for the 24h walltime
#SBATCH --gres=gpu:l40s:2            # 2x 48GB L40S = 96GB, shards the ~61GB 30B-A3B across both
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/30b-splits-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/30b-splits-%j.out
#
# EVERY scene split of EVERY video through Qwen3-VL-30B-A3B-Instruct, then a
# condition-filtered CSV manifest of the splits that matched.
#
#   sbatch scripts/39_qwen30b_all_splits.sbatch
#
# Step 1 is scripts/28 in scenes mode ("all"), pinned to the one model. Resume
# is per (model, segment): a resubmit after a walltime kill re-does at most the
# segment in flight. W&B is on (online) exactly as in 28/29.
#
# Step 2 is scripts/38, filtered to this model's records: it writes
#
#   outputs/children_walking_segments.csv   splits where the model saw a child
#                                           walking ("walks" in child_actions)
#   outputs/infants_supine_segments.csv     splits with an infant lying supine
#
# each row = video_id, split, segment start/end sec, timestamp, youtube url,
# url_at (deep link to the split's start second), plus the model's counts and
# action lists so stricter conditions (e.g. ONLY children, no infants) are a
# spreadsheet filter away. The condition itself lives in ONE lambda near the
# bottom of scripts/38_walking_manifest.py -- edit there for a new cohort.
set -uo pipefail          # NOT -e: 28 handles a failing model itself

ROOT="${ROOT:-/gpfs/scratch/sd6701/personal/ibdp}"
MODEL="Qwen3-VL-30B-A3B-Instruct"

# conda.sh, not `module load` -- see scripts/29_sweep.sbatch for why.
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ibdp

cd "${SLURM_SUBMIT_DIR:-$ROOT}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} partition=${SLURM_JOB_PARTITION:-?}"
echo "gpus=${CUDA_VISIBLE_DEVICES:-?} model=$MODEL"

# --- 1. annotate every scene split with the one model -----------------------
# SCENES=1 MAX_SECONDS=0 = the real full pass, not a shakedown window.
# SUMMARY=0 skips the timeline-fusion step -- this job's deliverable is the
# manifest CSV, not per-video prose (drop the =0 if you want both).
MODELS="$MODEL" SCENES=1 MAX_SECONDS=0 SUMMARY=0 \
    scripts/28_run_all_models.sh all
sweep_rc=$?
[[ $sweep_rc -ne 0 ]] && echo "sweep exited rc=$sweep_rc -- building the manifest from whatever finished" >&2

# --- 2. condition -> CSV manifest -------------------------------------------
# Only THIS model's records: min-models stays 1 because there is only one
# model's opinion in play.
python scripts/38_walking_manifest.py \
    --outdir "$ROOT/outputs" \
    --segmentation scenes \
    --models "$MODEL" \
    --min-models 1

echo
echo "manifests:"
ls -lh "$ROOT/outputs/children_walking_segments.csv" \
       "$ROOT/outputs/infants_supine_segments.csv" 2>/dev/null

# Offline W&B runs are inert files until synced (compute nodes have no route).
if compgen -G "$ROOT/outputs/wandb/offline-run-*" > /dev/null; then
  echo
  echo "wandb offline runs pending -- from a LOGIN node:"
  echo "  wandb sync $ROOT/outputs/wandb/offline-run-*"
fi
