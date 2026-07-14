#!/usr/bin/env bash
#
# Mirror the finished dataset off /gpfs/scratch onto permanent storage, and
# reconcile what was actually downloaded against the manifest.
#
# WHY THIS EXISTS: /gpfs/scratch is PURGED on a rolling window. A dataset that
# took two days to pull can evaporate. Scratch is the right place to *build*
# it and the wrong place to *keep* it.
#
# NOTE ON RESUME: this script does not prevent re-downloading -- yt-dlp's
# --download-archive already does that, by video id, inside 10_fetch_youtube.sh.
# rsync cannot help there: the source of truth is YouTube, not a directory you
# can diff against. What rsync gives you is a cheap, incremental,
# interrupt-safe copy from scratch to somewhere durable.
#
# Usage:
#   scripts/11_sync_dataset.sh            # dry run -- prints what WOULD copy
#   scripts/11_sync_dataset.sh --go       # actually copy
#
set -euo pipefail

SRC="${SRC:-/gpfs/scratch/sd6701/personal/ibdp/youtube_dataset}"
DEST="${DEST:-/gpfs/data/CHANGEME/ibdp/youtube_dataset}"   # your lab's permanent dir

MANIFEST="$SRC/manifest.tsv"
ARCHIVE="$SRC/downloaded.txt"

# ---- 1. Reconcile ----------------------------------------------------------
# Which videos in the manifest never landed? --ignore-errors means yt-dlp skips
# private/geoblocked/deleted videos silently, so the gap is expected and worth
# seeing explicitly rather than discovering downstream.
if [[ -f "$MANIFEST" && -f "$ARCHIVE" ]]; then
  want=$(cut -f1 "$MANIFEST" | sort -u)
  # archive lines look like: "youtube <id>"
  have=$(awk '{print $2}' "$ARCHIVE" | sort -u)
  missing=$(comm -23 <(echo "$want") <(echo "$have") || true)
  echo "manifest: $(echo "$want" | grep -c . ) videos"
  echo "archive:  $(echo "$have" | grep -c . ) downloaded"
  if [[ -n "$missing" ]]; then
    echo "MISSING ($(echo "$missing" | grep -c . )):"
    echo "$missing" | sed 's/^/  /'
    echo "$missing" > "$SRC/missing.txt"
    echo "  -> written to $SRC/missing.txt"
  fi
  echo
fi

# ---- 2. Mirror -------------------------------------------------------------
[[ "$DEST" == *CHANGEME* ]] && { echo "Set DEST to your real /gpfs/data path first." >&2; exit 1; }
mkdir -p "$DEST"

# --partial + --append-verify: an interrupted transfer of a 2GB mp4 resumes
#   instead of restarting.
# --exclude '*.part': yt-dlp's in-flight temp files. Never mirror those; a
#   half-written .part copied to DEST looks like a real file later.
# NOT using --delete: DEST is the archive of record. Nothing should be removed
#   from it just because scratch got purged.
FLAGS=(-a --info=progress2 --human-readable --partial --append-verify
       --exclude '*.part' --exclude '*.ytdl' --exclude 'logs/')

if [[ "${1:-}" == "--go" ]]; then
  rsync "${FLAGS[@]}" "$SRC/" "$DEST/"
  echo "Synced -> $DEST"
  du -sh "$DEST"
else
  echo "DRY RUN (pass --go to copy for real):"
  rsync "${FLAGS[@]}" --dry-run "$SRC/" "$DEST/"
fi
