#!/usr/bin/env bash
#SBATCH --job-name=ibdp-yt
#SBATCH --partition=CHANGEME          # a CPU partition -- from `sinfo`
#SBATCH --account=CHANGEME            # from `sacctmgr show assoc user=$USER`
#SBATCH --cpus-per-task=4             # yt-dlp is I/O bound; ffmpeg mux wants a few
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/youtube_dataset/logs/yt-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/youtube_dataset/logs/yt-%j.out
#
# CPU batch job wrapping the downloader. NO GPU: this is network + ffmpeg only.
#
# !! READ THIS BEFORE SUBMITTING !!
# BigPurple compute nodes generally have NO outbound internet. If that is true
# of your partition, this job cannot work and 10_fetch_youtube.sh will abort in
# its preflight check within ~10s (by design -- better than a 12h hang).
#
# Test first, interactively, before you queue anything:
#   srun --partition=CHANGEME --account=CHANGEME --time=00:05:00 --pty bash
#   curl -sSf -m 10 -o /dev/null https://www.youtube.com && echo "HAS INTERNET" || echo "NO INTERNET"
#
# If that prints NO INTERNET: do not use this file. Run the plain shell script
# on a login node inside tmux instead -- that is the supported path.
#
# Submit:
#   mkdir -p /gpfs/scratch/sd6701/personal/ibdp/youtube_dataset/logs
#   sbatch scripts/10_fetch_youtube.sbatch
#
set -euo pipefail

module load anaconda3/gpu/2023.09     # substitute what `module avail` shows
source activate ibdp

# The download is resumable, so a job that hits the walltime wall is not a
# failure -- just resubmit and it continues from the archive.
srun scripts/10_fetch_youtube.sh
