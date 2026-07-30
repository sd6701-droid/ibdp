#!/usr/bin/env python
"""
Bundle every model's answer for ONE scene split per video into a single JSON.

    python scripts/33_sample_outputs.py --videos 8yDn1uFbs4s,gN3aRdFW45g,H2JWku-kJCA
    python scripts/33_sample_outputs.py --videos 8yDn1uFbs4s --segment 300

Reads the annotations_*.jsonl files that scripts/26 wrote and, for each
requested video, picks ONE segment and collects all models' records for it
side by side -- the "show me what every model said about this exact clip"
view that the per-model JSONL layout makes tedious by hand.

HOW THE SEGMENT IS PICKED (unless --segment forces one): best model COVERAGE
first, lowest index as the tie-break. Runs are not always complete -- a model
may still be mid-run, the Omni skips clips over --clip-max-sec entirely, and a
crashed run leaves holes -- so "the split every model answered" is a better
default exemplar than "the first split". Whoever is missing from the chosen
segment is listed in missing_models rather than silently absent.

SCENE RECORDS ONLY (segmentation == "scenes"), parse_ok only, and the raw
model text is stripped: this file is for reading, and a failed parse's raw
dump belongs in the source JSONL where its context is.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True,
                    help="comma-separated video id(s), one sampled split each")
    ap.add_argument("--outdir", type=Path, default=ROOT / "outputs",
                    help="where the annotations_*.jsonl live")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: <outdir>/sample_<n>videos_all_models.json)")
    ap.add_argument("--segment", type=int, default=None,
                    help="force this segment_index for every video instead of "
                         "picking by coverage (splits are index*100, so "
                         "split_03 whole = 300)")
    args = ap.parse_args()

    videos = [v.strip() for v in args.videos.split(",") if v.strip()]

    # (video_id, segment_index) -> {model: record}. Files are read in name
    # order and later records overwrite earlier ones, so a re-run of a segment
    # wins -- same rule as scripts/27.
    recs = defaultdict(dict)
    files = sorted(args.outdir.glob("annotations_*.jsonl"))
    if not files:
        raise SystemExit(f"no annotations_*.jsonl under {args.outdir}")
    for f in files:
        with f.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (r.get("segmentation") == "scenes"
                        and r.get("video_id") in videos
                        and r.get("parse_ok")):
                    r.pop("raw", None)
                    recs[(r["video_id"], r["segment_index"])][r["model"]] = r

    all_models = sorted({m for d in recs.values() for m in d})
    print(f"models found: {len(all_models)}: {all_models}\n")

    out = {}
    for vid in videos:
        if args.segment is not None:
            by_model = recs.get((vid, args.segment), {})
            seg = args.segment
            if not by_model:
                print(f"{vid}: no records for segment {args.segment}")
                continue
        else:
            cands = sorted(((k[1], d) for k, d in recs.items() if k[0] == vid),
                           key=lambda x: (-len(x[1]), x[0]))
            if not cands:
                print(f"{vid}: NO scene records found")
                continue
            seg, by_model = cands[0]

        any_r = next(iter(by_model.values()))
        out[vid] = {
            "split": any_r.get("split"),
            "segment_index": seg,
            "timestamp": any_r.get("timestamp"),
            "url": any_r.get("url"),
            "url_at": any_r.get("url_at"),
            "clip": any_r.get("video"),
            "n_models": len(by_model),
            "missing_models": sorted(set(all_models) - set(by_model)),
            "models": by_model,
        }
        print(f"{vid}: {any_r.get('split')} (seg {seg})  {any_r.get('url_at')}"
              f"  [{len(by_model)}/{len(all_models)} models]")

    if not out:
        raise SystemExit("nothing to write")

    dst = args.out or (args.outdir /
                       f"sample_{len(videos)}videos_all_models.json")
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
