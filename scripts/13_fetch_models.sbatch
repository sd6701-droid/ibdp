#!/usr/bin/env bash
#SBATCH --job-name=ibdp-models
#SBATCH --partition=data_mover        # see PARTITION NOTE below
##SBATCH --account=CHANGEME           # uncomment IF sacctmgr shows you need one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # network-bound; hf parallelises shards
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/models/_logs/fetch-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/models/_logs/fetch-%j.out
#
# CPU batch wrapper for 13_fetch_models.sh -- no tmux, no open SSH session, no
# 400GB transfer running on a shared login node. Submit and walk away; the job
# survives your laptop closing.
#
# ---------------------------------------------------------------------------
# PARTITION NOTE
#
# This job needs OUTBOUND INTERNET, which most compute partitions do not have.
# data_mover exists to move data in and out, so it is the default here. Confirm
# in 30 seconds before queueing 12 hours of walltime:
#
#   srun --partition=data_mover --time=00:05:00 --pty \
#     curl -sSf -m 10 -o /dev/null https://huggingface.co && echo HAS_NET || echo NO_NET
#
# If that says NO_NET, fall back to nohup on the login node -- see the bottom of
# this comment block.
#
# WALLTIME: 414GB is a lot of bytes. The script is resumable (per-model
# .complete markers, and hf download resumes partial shards), so hitting the
# wall is not a failure -- resubmit and it continues where it stopped.
#
# SUBMIT:
#   mkdir -p /gpfs/scratch/sd6701/personal/ibdp/models/_logs
#   sbatch scripts/13_fetch_models.sbatch
#   sbatch scripts/13_fetch_models.sbatch --only internvl3-38b     # flags forward
#   squeue -u $USER
#   tail -f /gpfs/scratch/sd6701/personal/ibdp/models/_logs/fetch-*.out
#
# NO-SLURM FALLBACK (login node, still no tmux -- nohup detaches it):
#   nohup ./scripts/13_fetch_models.sh > ~/models-fetch.log 2>&1 &
#   tail -f ~/models-fetch.log
# ---------------------------------------------------------------------------
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate ibdp
else
  echo "WARN: no conda at $CONDA_BASE -- relying on whatever hf is on PATH" >&2
fi

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "node=$(hostname) partition=${SLURM_JOB_PARTITION:-?} job=${SLURM_JOB_ID:-?}"
mkdir -p /gpfs/scratch/sd6701/personal/ibdp/models/_logs

# The script preflights connectivity, the hf CLI and free disk, exiting fast
# with a clear message on any of them -- so a misrouted job dies in seconds
# rather than burning 12h of walltime. "$@" forwards --only / --dry-run.
#
# RUN DIRECTLY, NOT UNDER srun. data_mover nodes hand the job a restricted CPU
# mask, and srun then tries to re-bind to it and dies before the script starts:
#   srun: error: CPU binding outside of job step allocation, allocated CPUs
#         are: 0x0000003802
#   srun: error: Unable to satisfy cpu bind request
# The job "succeeds" in the sense that sbatch returns, and nothing downloads.
# This is one task on one node -- srun added a failure mode and no capability.
scripts/13_fetch_models.sh "$@"
