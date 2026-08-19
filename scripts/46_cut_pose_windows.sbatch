#!/usr/bin/env bash
#SBATCH --job-name=ibdp-cutwin
#SBATCH --partition=cpu_short         # ffmpeg only. Asking for a GPU would just queue longer.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # libx264 threads
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/cutwin-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/cutwin-%j.out
#
# Cut every downloaded Chambers video to the window they actually analysed,
# per URL_pose_dataset.csv -> datasets/chambers_infant_youtube/clips_pose.
#
# NO GPU: this is ffmpeg re-encoding a few minutes of footage. It also needs no
# network, so unlike the downloader it is a normal batch job.
#
# SUBMIT:
#   mkdir -p /gpfs/scratch/sd6701/personal/ibdp/outputs/logs
#   sbatch scripts/46_cut_pose_windows.sbatch
#
# Flags pass through:
#   sbatch scripts/46_cut_pose_windows.sbatch --dry-run
#   sbatch scripts/46_cut_pose_windows.sbatch --force
#   sbatch scripts/46_cut_pose_windows.sbatch \
#          --csv /gpfs/.../URL_labelled_dataset.csv --dest /gpfs/.../clips
#
# NEXT STEP once this finishes -- pose the windows it produced:
#   sbatch --export=ALL,\
#   VIDEOS=/gpfs/scratch/sd6701/personal/ibdp/datasets/chambers_infant_youtube/clips_pose,\
#   OUTDIR=/gpfs/scratch/sd6701/personal/ibdp/outputs/poses_chambers_clips,\
#   GPUS=1 scripts/42_pose_chambers_one_node.sbatch
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="${ROOT:-/gpfs/scratch/sd6701/personal/ibdp}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-ibdp}"

command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg missing in $CONDA_PREFIX" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$ROOT}"
echo "node=$(hostname) job=${SLURM_JOB_ID:-?} env=$CONDA_PREFIX"

srun python scripts/46_cut_pose_windows.py "$@"
