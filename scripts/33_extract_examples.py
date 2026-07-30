#!/usr/bin/env python
"""
Pull EXAMPLE scene-split annotations -- one split per video, all models side
by side -- into a single JSON, for eyeballing / sharing.

For each requested video it picks the split that the MOST models have
annotated (ties -> lowest split index), and emits every model's full record
for that split, keyed by model tag. The url_at link in each record jumps
straight to the split's timestamp on YouTube, so the JSON is self-adjudicating:
open the link, read the six descriptions, see who was right.

JSON goes to --out, or to STDOUT when --out is '-'. All logging goes to
stderr, so the stdout stream is clean JSON even through an ssh pipe:

    ssh bigpurple "cd /gpfs/scratch/sd6701/personal/ibdp && python3 -" \
        < scripts/33_extract_examples.py > vlm_examples.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")
LEGACY_MODEL_TAG = "Qwen3-VL-30B-A3B-Instruct"  # records from before --model

DEFAULT_VIDEOS = ["8yDn1uFbs4s", "gN3aRdFW45g", "H2JWku-kJCA"]
DEFAULT_MODELS = ["Qwen3-VL-30B-A3B-Instruct", "Qwen3-Omni-30B-A3B-Instruct",
                  "Qwen3-VL-32B-Instruct", "InternVL3-38B",
                  "Qwen2.5-VL-72B-Instruct", "InternVL3-78B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=ROOT / "outputs")
    ap.add_argument("--videos", default=",".join(DEFAULT_VIDEOS),
                    help="comma-separated video ids")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated model tags to include")
    ap.add_argument("--out", default="-",
                    help="output json path, '-' for stdout")
    args = ap.parse_args()

    videos = [v for v in args.videos.split(",") if v]
    models = [m for m in args.models.split(",") if m]

    # (video, model) -> {split_index: record}. Files in name order, last
    # record wins, mirroring scripts/27 -- a resumed re-run overwrites.
    recs = defaultdict(dict)
    files = sorted(args.outdir.glob("annotations_*.jsonl"))
    if not files:
        sys.exit(f"no annotations_*.jsonl under {args.outdir}")
    for f in files:
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("video_id") not in videos:
                    continue
                # Scene splits only: grid records number different stretches
                # of the same video and must never mix in (see scripts/27).
                if (r.get("segmentation") or "fixed") != "scenes":
                    continue
                m = r.get("model", LEGACY_MODEL_TAG)
                if m not in models:
                    continue
                recs[(r["video_id"], m)][r["segment_index"]] = r

    out = {"segmentation": "scenes", "models": models, "examples": []}
    for vid in videos:
        # Coverage per split: how many of the requested models annotated it.
        cover = defaultdict(list)
        for m in models:
            for idx in recs.get((vid, m), {}):
                cover[idx].append(m)
        if not cover:
            print(f"WARN: no scene-split records at all for {vid}", file=sys.stderr)
            continue
        best = max(sorted(cover), key=lambda i: len(cover[i]))
        got = cover[best]
        missing = [m for m in models if m not in got]
        if missing:
            print(f"WARN: {vid} split {best}: missing {', '.join(missing)}",
                  file=sys.stderr)
        any_rec = recs[(vid, got[0])][best]
        out["examples"].append({
            "video_id": vid,
            "video_name": any_rec.get("video_name"),
            "split_index": best,
            "timestamp": any_rec.get("timestamp"),
            "url": any_rec.get("url"),
            "url_at": any_rec.get("url_at"),
            "models_present": got,
            "outputs": {m: recs[(vid, m)][best] for m in models if m in got},
        })
        print(f"{vid}: split {best} ({any_rec.get('timestamp')}), "
              f"{len(got)}/{len(models)} models", file=sys.stderr)

    text = json.dumps(out, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
