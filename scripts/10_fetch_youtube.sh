#!/usr/bin/env bash
#
# Build the YouTube video dataset.
#
# MUST run on a BigPurple LOGIN node. Compute nodes have no internet, so
# yt-dlp will stall there with no useful error. This is also why it is not an
# sbatch job.
#
# Resumable: every completed download is recorded in the archive file, so
# re-running picks up where it left off. Safe to kill and restart.
#
# Usage:
#   scripts/10_fetch_youtube.sh                 # list + download
#   scripts/10_fetch_youtube.sh --list-only     # just refresh the manifest
#
set -euo pipefail

CHANNEL="${CHANNEL:-https://www.youtube.com/channel/UCvznoyf6e8T1NRHLkfxwG9Q/videos}"
ROOT="${ROOT:-/gpfs/scratch/sd6701/personal/ibdp/youtube_dataset}"

VIDEO_DIR="$ROOT/videos"
META_DIR="$ROOT/metadata"
LOG_DIR="$ROOT/logs"
MANIFEST="$ROOT/manifest.tsv"
ARCHIVE="$ROOT/downloaded.txt"   # yt-dlp's record of what is already done

mkdir -p "$VIDEO_DIR" "$META_DIR" "$LOG_DIR"

command -v yt-dlp >/dev/null || { echo "yt-dlp not found. pip install -U yt-dlp" >&2; exit 1; }
command -v ffmpeg >/dev/null || echo "WARN: ffmpeg not on PATH; yt-dlp cannot merge separate video+audio streams." >&2

# Refuse to run somewhere with no network rather than hang for 20 minutes.
if ! curl -sSf -m 10 -o /dev/null https://www.youtube.com; then
  echo "ERROR: no route to youtube.com. You are probably on a compute node." >&2
  echo "       Log into a login node and run this there." >&2
  exit 1
fi

# ---- 1. Manifest -----------------------------------------------------------
# --flat-playlist: read the channel index only, do not touch each video page.
# NB: yt-dlp does NOT expand \t in --print -- it emits a literal backslash-t,
# which silently produces a one-column "TSV". TAB below is a real tab.
TAB=$'\t'
echo "Listing channel -> $MANIFEST"
yt-dlp --flat-playlist \
       --print "%(id)s${TAB}%(title)s${TAB}%(duration)s${TAB}https://www.youtube.com/watch?v=%(id)s" \
       "$CHANNEL" > "$MANIFEST"
echo "  $(wc -l < "$MANIFEST") videos listed."

[[ "${1:-}" == "--list-only" ]] && exit 0

# ---- 2. Download -----------------------------------------------------------
# Files are named by video id, not title: titles contain slashes, quotes and
# emoji, and we want a stable key to join against the manifest.
echo "Downloading -> $VIDEO_DIR"
yt-dlp \
  --download-archive "$ARCHIVE" \
  --paths "home:$VIDEO_DIR" \
  --paths "infojson:$META_DIR" \
  --paths "subtitle:$META_DIR" \
  --output "%(id)s.%(ext)s" \
  --format "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best" \
  --merge-output-format mp4 \
  --write-info-json \
  --write-subs --write-auto-subs --sub-langs "en.*" --sub-format "vtt" \
  --concurrent-fragments 4 \
  --retries 10 --fragment-retries 10 \
  --sleep-requests 1 --sleep-interval 2 --max-sleep-interval 6 \
  --ignore-errors \
  --no-overwrites \
  --progress \
  "$CHANNEL" 2>&1 | tee -a "$LOG_DIR/download.log"

echo
echo "Done. $(find "$VIDEO_DIR" -name '*.mp4' | wc -l) mp4 files in $VIDEO_DIR"
du -sh "$ROOT"
