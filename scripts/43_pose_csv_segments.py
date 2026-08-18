#!/usr/bin/env python3
"""
ViTPose over the scene clips listed in a cohort CSV from scripts/38.

    python scripts/43_pose_csv_segments.py \
        --csv outputs/children_walk_or_stand_segments.csv

Every row of those CSVs carries a `clip_path` -- a pre-cut scene clip on gpfs.
Output goes to <outdir>/<video_id>__<split>/{vis,pred}, the same layout the
whole-video sweep produces.

WHY A DRIVER AND NOT A FOR-LOOP: every clip on disk is named `clip.mp4`
(outputs/scenes/<video_id>/<split>/clip.mp4). 30_infant_pose.py names its
output directory after the file STEM, so feeding those in directly would send
every segment to <outdir>/clip/ and each would overwrite the last. This gives
each segment a unique name first.

TRIM: --skip-start drops the opening seconds of each clip before posing
(default 2.0). Scene cuts land on a shot boundary, so the first frames are
often a dissolve, a title card, or the camera still settling -- pose on those
is noise that then propagates into every velocity/acceleration feature. The
trim is a real ffmpeg re-encode into <outdir>/_clips, so the ORIGINAL scene
clips are never modified.

Clips shorter than --skip-start + --min-keep are SKIPPED, not silently
truncated: some scene splits are only 3-4s long, and 1s of pose is not worth
a feature row that looks as legitimate as any other.
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")


def safe(s: str) -> str:
    """Filesystem-safe token: splits and ids are tame, but never trust a CSV."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", (s or "").strip())


def duration(path: Path) -> float:
    """Seconds, or 0.0 if ffprobe cannot say."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def trim(src: Path, start: float, dst: Path) -> bool:
    """src minus its first `start` seconds -> dst. Re-encodes rather than
    -c copy so the cut is frame-accurate; on clips this short a keyframe snap
    could swallow a large fraction of the segment."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-i", str(src),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-an", str(dst)]
    return subprocess.run(cmd).returncode == 0


def load_rows(csv_path: Path, only: set[str] | None):
    """Cohort rows -> (name, clip) pairs, deduped. A video appears once per
    split; a split appears once per matching model."""
    rows, seen = [], set()
    with csv_path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            clip = (r.get("clip_path") or "").strip()
            vid = safe(r.get("video_id"))
            split = safe(r.get("split")) or f"seg{safe(r.get('segment_index'))}"
            if not clip or not vid:
                continue
            if only and vid not in only:
                continue
            name = f"{vid}__{split}"
            if name in seen:
                continue
            seen.add(name)
            rows.append((name, Path(clip)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=ROOT / "outputs/children_walk_or_stand_segments.csv")
    ap.add_argument("--outdir", type=Path, default=ROOT / "outputs/poses_walk_stand")
    ap.add_argument("--mm", type=Path,
                    default=Path("/gpfs/scratch/sd6701/personal/mmpose"))
    ap.add_argument("--skip-start", type=float, default=2.0,
                    help="drop this many opening seconds of each clip (0 = none)")
    ap.add_argument("--min-keep", type=float, default=1.0,
                    help="skip a clip if less than this remains after the trim")
    ap.add_argument("--only", help="comma-separated video ids")
    ap.add_argument("--limit", type=int, default=0, help="first N segments only")
    ap.add_argument("--force", action="store_true", help="redo segments already done")
    ap.add_argument("--dry-run", action="store_true", help="list the work, run nothing")
    args = ap.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"no such csv: {args.csv}")

    only = {x.strip() for x in args.only.split(",")} if args.only else None
    rows = load_rows(args.csv, only)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit(f"no usable rows (need a clip_path column) in {args.csv}")

    pose_cfg = (args.mm / "configs/body_2d_keypoint/topdown_heatmap/coco/"
                          "td-hm_ViTPose-large_8xb64-210e_coco-256x192.py")
    det_cfg = args.mm / "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
    det_w = next(iter(sorted(args.mm.glob("checkpoints/*rtmdet*.pth"))), None)
    for label, path in [("pose config", pose_cfg), ("det config", det_cfg),
                        ("det weights", det_w)]:
        if path is None or not Path(path).is_file():
            raise SystemExit(f"missing {label}: {path}\n(checked --mm {args.mm})")

    # Trimmed copies live beside the OUTPUT, never in the scene tree the clips
    # came from -- that tree is an input and stays untouched.
    work = args.outdir / "_clips"
    work.mkdir(parents=True, exist_ok=True)

    print(f"{len(rows)} segment(s) from {args.csv.name}", flush=True)
    if args.skip_start:
        print(f"trimming first {args.skip_start:g}s of each clip", flush=True)
    done = skipped = short = failed = 0

    for i, (name, clip) in enumerate(rows, 1):
        out = args.outdir / name
        if not args.force and any(out.glob("pred/*.json")):
            print(f"[{i}/{len(rows)}] skip {name} (already posed)", flush=True)
            skipped += 1
            continue
        if not clip.is_file():
            print(f"[{i}/{len(rows)}] MISSING clip for {name}: {clip}",
                  file=sys.stderr, flush=True)
            failed += 1
            continue

        dur = duration(clip)
        if args.skip_start and dur and dur - args.skip_start < args.min_keep:
            print(f"[{i}/{len(rows)}] TOO SHORT {name}: {dur:.1f}s clip, "
                  f"{dur - args.skip_start:.1f}s would remain", flush=True)
            short += 1
            continue

        src = work / f"{name}.mp4"
        if args.skip_start:
            if not args.dry_run and not trim(clip, args.skip_start, src):
                print(f"[{i}/{len(rows)}] ffmpeg trim FAILED {name}",
                      file=sys.stderr, flush=True)
                failed += 1
                continue
        else:
            if src.is_symlink() or src.exists():
                src.unlink()
            src.symlink_to(clip)

        print(f"[{i}/{len(rows)}] pose {name} ({dur:.1f}s clip)", flush=True)
        if args.dry_run:
            continue
        cmd = ["python", "scripts/30_infant_pose.py",
               "--video", str(src),
               "--seconds", "0",          # src is already the exact window
               "--config", str(pose_cfg),
               "--det-config", str(det_cfg),
               "--det-weights", str(det_w),
               "--outdir", str(args.outdir),
               "--draw-bbox"]
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            done += 1
        else:
            print(f"  FAILED {name} (rc={rc}) -- continuing", file=sys.stderr)
            failed += 1

    print(f"\ndone={done} skipped={skipped} too_short={short} failed={failed}")
    print(f"output -> {args.outdir}/<video_id>__<split>/{{vis,pred}}")


if __name__ == "__main__":
    main()
