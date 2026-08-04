#!/usr/bin/env python3
"""
Combine every <splits-dir>/<video_id>/scenes.csv (written by
scripts/32_split_scenes.py) into ONE csv of youtube url + split boundaries,
enough to re-download and re-cut every clip from source:

    python scripts/36_splits_manifest_csv.py \
        --splits-dir test_data_splits --out test_data_splits/splits_manifest.csv

Columns: video_id, youtube_url, split, start_sec, end_sec, duration_sec,
start_hms, end_hms, cut_confidence, url_at_start, clip_path.

The directory NAME is the youtube id -- that is scripts/32's contract
(outputs/scenes/<video_id>/...). Directories without a scenes.csv (hand-made
folders, lab videos) are skipped with a note rather than guessed at.
"""
import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def hms(seconds: float) -> str:
    s = float(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", type=Path,
                    default=REPO / "test_data_splits",
                    help="tree of <video_id>/scenes.csv from scripts/32")
    ap.add_argument("--out", type=Path, default=None,
                    help="output csv (default: <splits-dir>/splits_manifest.csv)")
    args = ap.parse_args()

    if not args.splits_dir.is_dir():
        raise SystemExit(f"no such directory: {args.splits_dir}")
    out = args.out or args.splits_dir / "splits_manifest.csv"

    rows, skipped = [], []
    for vdir in sorted(p for p in args.splits_dir.iterdir() if p.is_dir()):
        scenes = vdir / "scenes.csv"
        if not scenes.is_file():
            skipped.append(vdir.name)
            continue
        vid = vdir.name
        url = f"https://www.youtube.com/watch?v={vid}"
        with scenes.open() as fh:
            for r in csv.DictReader(fh):
                start, end = float(r["start"]), float(r["end"])
                clip = vdir / r["split"] / "clip.mp4"
                rows.append({
                    "video_id": vid,
                    "youtube_url": url,
                    "split": r["split"],
                    "start_sec": f"{start:.3f}",
                    "end_sec": f"{end:.3f}",
                    "duration_sec": f"{end - start:.3f}",
                    "start_hms": hms(start),
                    "end_hms": hms(end),
                    "cut_confidence": r.get("cut_confidence", ""),
                    "url_at_start": f"{url}&t={int(start)}s",
                    "clip_path": str(clip.relative_to(args.splits_dir))
                                 if clip.is_file() else "",
                })

    if not rows:
        raise SystemExit(f"no <video_id>/scenes.csv found under "
                         f"{args.splits_dir} -- rsync the splits tree first")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n_vids = len({r["video_id"] for r in rows})
    print(f"wrote {out}: {len(rows)} splits across {n_vids} videos")
    if skipped:
        print(f"skipped (no scenes.csv): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
