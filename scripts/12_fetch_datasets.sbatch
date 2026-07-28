#!/usr/bin/env bash
#SBATCH --job-name=ibdp-datasets
#SBATCH --partition=data_mover        # see PARTITION NOTE below
##SBATCH --account=CHANGEME           # uncomment IF sacctmgr shows you need one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # network-bound; unzip wants one or two
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/datasets/_logs/fetch-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/datasets/_logs/fetch-%j.out
#
# CPU batch wrapper for 12_fetch_datasets.sh. No GPU: this is curl, git and
# unzip. Asking for an A100 here would only make you queue behind real work.
#
# ---------------------------------------------------------------------------
# PARTITION NOTE -- same lesson as 10_fetch_youtube.sbatch.
#
# The only thing this job needs is OUTBOUND INTERNET, which is exactly what
# compute partitions tend not to have. data_mover exists to move data in and
# out, so it is the default here rather than cpu_medium.
#
# Settle it in 30 seconds before queueing anything:
#
#   srun --partition=data_mover --time=00:05:00 --pty \
#     curl -sSf -m 10 -o /dev/null https://zenodo.org && echo HAS_NET || echo NO_NET
#
# If that says NO_NET, try cpu_medium; if BOTH say NO_NET, no batch job can do
# this and you run it on the login node under tmux:
#
#   tmux new -s ds
#   ./scripts/12_fetch_datasets.sh
#   # detach ctrl-b then d ; reattach: tmux attach -t ds
#
# ---------------------------------------------------------------------------
# WALLTIME: the script is resumable (per-dataset .complete markers, curl -C-),
# so hitting the wall is not a failure. Resubmit and it continues.
#
# SUBMIT:
#   mkdir -p /gpfs/scratch/sd6701/personal/ibdp/datasets/_logs
#   sbatch scripts/12_fetch_datasets.sbatch
#   squeue -u $USER
#
# Only some of them:
#   sbatch scripts/12_fetch_datasets.sbatch --only babypose,chambers
# ---------------------------------------------------------------------------
set -euo pipefail

# Use YOUR miniconda, not the anaconda3 module -- that module puts a different
# conda on PATH which knows nothing about the ibdp envs.
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate ibdp
else
  echo "WARN: no conda at $CONDA_BASE -- relying on system curl/git/python3" >&2
fi

for tool in curl git python3; do
  command -v "$tool" >/dev/null || { echo "ERROR: $tool missing" >&2; exit 1; }
done

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "node=$(hostname) partition=${SLURM_JOB_PARTITION:-?} job=${SLURM_JOB_ID:-?}"
mkdir -p /gpfs/scratch/sd6701/personal/ibdp/datasets/_logs

# The script does its own 10s connectivity preflight and exits 1 with a clear
# message on a firewalled node, so a misrouted job dies in seconds rather than
# hanging for the full walltime. "$@" forwards --only / --dry-run from sbatch.
srun scripts/12_fetch_datasets.sh "$@"
