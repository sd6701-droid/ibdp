#!/usr/bin/env bash
#SBATCH --job-name=chambers-yt
#SBATCH --partition=data_mover        # needs OUTBOUND INTERNET. See note below.
##SBATCH --account=CHANGEME           # uncomment IF sacctmgr shows you need one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # yt-dlp is network-bound; ffmpeg mux wants a few
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/datasets/chambers_infant_youtube/logs/chambers-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/datasets/chambers_infant_youtube/logs/chambers-%j.out
#
# Batch wrapper for scripts/39_fetch_chambers_youtube.py.
#
# WHY THIS EXISTS: running the downloader directly on a login node gets the
# process SIGKILLed part-way through ("Killed", no traceback) -- login nodes are
# policed against exactly this kind of long network+ffmpeg job. The download is
# resumable, so a kill costs no data, but it will never finish there. Run it as
# a batch job instead.
#
# ---------------------------------------------------------------------------
# PARTITION NOTE -- the ONLY thing this job needs is outbound internet, and
# that is exactly what compute partitions usually do not have. data_mover
# exists to move data in and out, so it is the first candidate. Settle it in
# 30 seconds before queueing anything:
#
#   srun --partition=data_mover --time=00:05:00 --pty \
#     curl -sSf -m 10 -o /dev/null https://www.youtube.com && echo HAS_NET || echo NO_NET
#   srun --partition=cpu_medium --time=00:05:00 --pty \
#     curl -sSf -m 10 -o /dev/null https://www.youtube.com && echo HAS_NET || echo NO_NET
#
# Whichever says HAS_NET, put in --partition above (or pass it at submit time:
# `sbatch --partition=cpu_medium scripts/39_fetch_chambers_youtube.sbatch`).
#
# If BOTH say NO_NET, no batch job can do this and the login node is the only
# route. Then run it in small resumable chunks so no single process lives long
# enough to be reaped -- see the CHUNKED FALLBACK block at the bottom.
#
# SUBMIT:
#   mkdir -p /gpfs/scratch/sd6701/personal/ibdp/datasets/chambers_infant_youtube/logs
#   sbatch scripts/39_fetch_chambers_youtube.sbatch
#   squeue -u $USER
#   tail -f /gpfs/scratch/.../datasets/chambers_infant_youtube/logs/chambers-<jobid>.out
#
# Extra flags pass straight through:
#   sbatch scripts/39_fetch_chambers_youtube.sbatch --player-client 'default,web_safari'
# ---------------------------------------------------------------------------
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"

if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "ERROR: no conda at $CONDA_BASE" >&2
  echo "       Run 'conda info --base' on the login node and set CONDA_BASE." >&2
  exit 1
fi

# conda.sh, not `module load` -- this is what makes `conda activate` work in a
# non-interactive batch shell.
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ibdp

command -v yt-dlp >/dev/null || { echo "ERROR: yt-dlp missing. pip install -U 'yt-dlp[default]'" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg missing. conda install -c conda-forge ffmpeg" >&2; exit 1; }

# The JS runtime matters as much as yt-dlp itself: without one, YouTube
# extraction falls back to the android_vr client and reports "Video
# unavailable" for videos that are perfectly fine. The python script warns too,
# but a batch job's warning scrolls past unread -- so say it here as well.
if ! command -v deno >/dev/null && ! command -v node >/dev/null; then
  echo "WARN: no deno/node on PATH -- expect spurious 'Video unavailable'." >&2
  echo "      conda install -c conda-forge deno   (then resubmit)" >&2
fi

echo "env=$CONDA_PREFIX  yt-dlp=$(yt-dlp --version)  node=$(hostname)"
echo "partition=${SLURM_JOB_PARTITION:-?} job=${SLURM_JOB_ID:-?}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# The python script does its own 10s connectivity preflight and exits with a
# clear message on a firewalled node -- so a misrouted job dies in seconds
# rather than sitting on the walltime doing nothing.
srun python scripts/39_fetch_chambers_youtube.py --clips "$@"

# ---------------------------------------------------------------------------
# CHUNKED FALLBACK -- login node only, when no partition has internet.
#
# --limit N takes the FIRST N unique ids, and anything already on disk is
# skipped, so a rising --limit walks the list forward. Each invocation is short
# enough to finish before the watchdog notices:
#
#   for n in 20 40 60 80 101; do
#     python scripts/39_fetch_chambers_youtube.py --limit $n || true
#     sleep 60
#   done
#
# `|| true` because a chunk that gets killed should not stop the loop -- the
# next one resumes from whatever landed on disk.
# ---------------------------------------------------------------------------
