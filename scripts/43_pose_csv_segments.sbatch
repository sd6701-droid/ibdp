#!/usr/bin/env bash
#SBATCH --job-name=ibdp-poseseg
#SBATCH --partition=gl40s_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/outputs/poses_walk_stand/logs/poseseg-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/outputs/poses_walk_stand/logs/poseseg-%j.out
#
# ViTPose over the walk/stand cohort clips from scripts/38.
#
# These are SCENE CLIPS, not whole videos -- typically 3-15s each, so ~65 of
# them is a fraction of the 64-whole-video sweep and one card is plenty.
#
# SUBMIT:
#   mkdir -p /gpfs/scratch/sd6701/personal/ibdp/outputs/poses_walk_stand/logs
#   sbatch scripts/43_pose_csv_segments.sbatch
#
# Extra flags pass straight through to the driver:
#   sbatch scripts/43_pose_csv_segments.sbatch --skip-start 0     # no trim
#   sbatch scripts/43_pose_csv_segments.sbatch --limit 5          # smoke test
#   sbatch scripts/43_pose_csv_segments.sbatch \
#          --csv /gpfs/.../outputs/children_walking_segments.csv
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="${ROOT:-/gpfs/scratch/sd6701/personal/ibdp}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${POSE_ENV:-vitpose}"

cd "${SLURM_SUBMIT_DIR:-$ROOT}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} gpus=${CUDA_VISIBLE_DEVICES:-?}"

srun python scripts/43_pose_csv_segments.py "$@"
