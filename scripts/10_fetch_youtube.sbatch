#!/usr/bin/env bash
#SBATCH --job-name=ibdp-yt
#SBATCH --partition=cpu_medium        # CPU-only. See PARTITION NOTE below.
##SBATCH --account=CHANGEME           # uncomment IF sacctmgr shows you need one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # yt-dlp is network-bound; ffmpeg mux wants a few
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/youtube_dataset/logs/yt-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/youtube_dataset/logs/yt-%j.out
#
# CPU batch job wrapping the downloader. No GPU is requested: this is network
# + ffmpeg only, and asking for a GPU would just make you queue longer.
#
# ---------------------------------------------------------------------------
# PARTITION NOTE -- read once, then you never think about it again.
#
# The only thing this job needs is OUTBOUND INTERNET, and that is exactly what
# compute partitions usually do not have. From your `sinfo`:
#
#   cpu_dev / cpu_short / cpu_medium / cpu_long   compute. May be firewalled.
#   data_mover  (3 nodes, 120 CPUs, no GPUs)      EXISTS TO MOVE DATA IN/OUT.
#
# data_mover is almost certainly the right partition for a downloader. It is
# named for this job. If the preflight below fails on cpu_medium, do not debug
# it -- just switch:
#
#   sbatch --partition=data_mover scripts/10_fetch_youtube.sbatch
#
# Settle it in 30 seconds before queueing anything, on each candidate:
#
#   srun --partition=cpu_medium --time=00:05:00 --pty \
#     curl -sSf -m 10 -o /dev/null https://www.youtube.com && echo HAS_NET || echo NO_NET
#   srun --partition=data_mover --time=00:05:00 --pty \
#     curl -sSf -m 10 -o /dev/null https://www.youtube.com && echo HAS_NET || echo NO_NET
#
# If BOTH say NO_NET: no batch job can do this. Run the shell script directly
# on the login node (bigpurple-ln2, where you are now) inside tmux:
#
#   tmux new -s yt
#   ./scripts/10_fetch_youtube.sh
#   # detach with ctrl-b then d ; reattach later with: tmux attach -t yt
#
# ---------------------------------------------------------------------------
# ACCOUNT: BigPurple may not require --account. Check with
#   sacctmgr show assoc user=$USER format=account,partition
# If it lists an account, uncomment the #SBATCH --account line above and set it.
#
# WALLTIME: cpu_short caps out well below 12h on most clusters. If cpu_medium
# rejects 12:00:00, lower it -- the download is resumable, so a job that hits
# the wall is not a failure. Resubmit and it continues from the archive file.
#
# SUBMIT:
#   mkdir -p /gpfs/scratch/sd6701/personal/ibdp/youtube_dataset/logs
#   sbatch scripts/10_fetch_youtube.sbatch
#   squeue -u $USER
# ---------------------------------------------------------------------------
set -euo pipefail

module load anaconda3/gpu/2023.09     # substitute what `module avail anaconda` shows
source activate ibdp

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "node=$(hostname) partition=${SLURM_JOB_PARTITION:-?} job=${SLURM_JOB_ID:-?}"

# 10_fetch_youtube.sh does its own 10s connectivity preflight and exits 1 with a
# clear message if the node is firewalled -- so a misrouted job dies in seconds
# instead of hanging for the full 12h walltime.
srun scripts/10_fetch_youtube.sh
