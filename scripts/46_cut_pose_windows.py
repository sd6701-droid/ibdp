#!/usr/bin/env python3
"""
Cut each downloaded Chambers video down to the window Chambers et al. actually
analysed, per URL_pose_dataset.csv.

    python scripts/46_cut_pose_windows.py

WHY THIS EXISTS: we pose whole videos, they posed a window. A whole YouTube
video runs through title cards, cutaways, and long stretches where the infant
is off-screen and an adult is talking to camera -- ViTPose poses all of it,
because a person detector has no notion of infant vs adult. Kinematic features
over that mixture describe the wrong footage entirely.

AND NOT clips/ EITHER: datasets/chambers_infant_youtube/clips was cut from
URL_labelled_dataset.csv, a DIFFERENT window list. Of the 50 videos in both
lists, 42 windows agree and 8 do not -- dtEjZmOcu08 is 49-61s in the labelled
list but 0-25s in the pose list, and q33LLLzAS68 has one window there against
three here. For a like-for-like comparison with their published features, the
pose list is the one that counts.

Output: <dest>/<video_id>_<start>-<end>s.mp4, one per window. A video with two
windows yields two clips, which is why the count can exceed the video count.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")
YT_ID = re.compile(r"([0-9A-Za-z_-]{11})")


def windows(csv_path: Path):
    """CSV rows -> (video_id, start_sec, end_sec). Times are split across a
    float-minute and a float-second column (`6.0, 13.0` == 373s)."""
    out = []
    with csv_path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            m = YT_ID.search(r.get("url") or "")
            if not m:
                continue
            try:
                s = float(r["start_min"]) * 60 + float(r["start_sec"])
                e = float(r["end_min"]) * 60 + float(r["end_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if e > s:
                out.append((m.group(1), s, e))
    return out


def cut(src: Path, start: float, end: float, dst: Path) -> bool:
    """Re-encode rather than -c copy: these windows are seconds long, and a
    keyframe snap would shift the cut by a large fraction of the clip."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{end - start:.3f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-an", str(dst)]
    return subprocess.run(cmd).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=ROOT / "Chamber-etal-dataset/URL_pose_dataset.csv")
    ap.add_argument("--videos", type=Path,
                    default=ROOT / "datasets/chambers_infant_youtube/videos")
    ap.add_argument("--dest", type=Path,
                    default=ROOT / "datasets/chambers_infant_youtube/clips_pose")
    ap.add_argument("--min-seconds", type=float, default=2.0,
                    help="skip windows shorter than this; velocity and "
                         "acceleration IQRs over a handful of frames are noise "
                         "that looks like data")
    ap.add_argument("--force", action="store_true", help="recut existing clips")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"no such csv: {args.csv}")
    if not args.videos.is_dir():
        raise SystemExit(f"no such video dir: {args.videos}")
    args.dest.mkdir(parents=True, exist_ok=True)

    wins = windows(args.csv)
    print(f"{len(wins)} window(s) in {args.csv.name}")

    made = skipped = short = missing = failed = 0
    for vid, s, e in wins:
        src = args.videos / f"{vid}.mp4"
        if not src.is_file():
            missing += 1
            continue
        if e - s < args.min_seconds:
            print(f"  SHORT  {vid} {s:.0f}-{e:.0f}s ({e - s:.1f}s)", flush=True)
            short += 1
            continue
        dst = args.dest / f"{vid}_{s:.0f}-{e:.0f}s.mp4"
        if dst.is_file() and not args.force:
            skipped += 1
            continue
        print(f"  cut    {dst.name} ({e - s:.1f}s)", flush=True)
        if args.dry_run:
            continue
        if cut(src, s, e, dst):
            made += 1
        else:
            print(f"  FAILED {vid}", file=sys.stderr)
            failed += 1

    print(f"\nmade={made} skipped={skipped} too_short={short} "
          f"not_downloaded={missing} failed={failed}")
    print(f"-> {args.dest}")
    if missing:
        print(f"   ({missing} windows belong to videos we never downloaded -- "
              "they are in the pose list but not the labelled list)")


if __name__ == "__main__":
    main()
