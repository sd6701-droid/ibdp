#!/usr/bin/env bash
#SBATCH --job-name=ibdp-split
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/split-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/split-%j.out
#
# Shot-boundary split of the YouTube corpus. Every argument is forwarded to
# 32_split_scenes.py.
#
#   sbatch scripts/32_split_scenes.sbatch --only 8yDn1uFbs4s --wav --thumbs
#   sbatch scripts/32_split_scenes.sbatch --resume            # whole corpus
#
# ONE A100, AND A SMALL ONE WOULD DO. TransNetV2 is 3.5M parameters at
# 48x27 -- it is the ffmpeg re-encode that costs, which is why cpus-per-task is
# 16 and not 8. On CPU the detection is ~30fps (a 10-minute video takes ~8
# minutes of the 4h wall) so `--device cpu` on a cpu_short partition is a real
# option when the GPU queue is long:
#
#   sbatch -p cpu_short --gres=NONE scripts/32_split_scenes.sbatch --device cpu
#
# WALLTIME: --resume keeps every split whose recorded boundaries still match,
# so a job killed at the wall costs one re-encode, not the video. Detection
# itself is cheap to repeat; pass --scenes-json auto to skip even that and
# re-cut from the boundaries already on disk.
set -euo pipefail

ROOT=/gpfs/scratch/sd6701/personal/ibdp

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ibdp

export HF_HOME=$ROOT/.hf
export HF_HUB_OFFLINE=1        # no network on compute; fail fast, don't hang

cd "${SLURM_SUBMIT_DIR:-$ROOT}"
mkdir -p "$ROOT/outputs/logs"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "args: $*"
ffmpeg -version | head -1

# NO srun, deliberately. This is a single-task job, so srun adds nothing --
# and it actively breaks when the sbatch was submitted from INSIDE another
# job (an interactive session): sbatch exports the submitter's environment,
# the parent job's SLURM_CPU_BIND names CPUs on the parent's node, and srun
# then dies on whatever node this job landed on with
#   "CPU binding outside of job step allocation"
# Running python directly uses this job's own allocation and cannot inherit
# a stale binding.
python scripts/32_split_scenes.py "$@"
