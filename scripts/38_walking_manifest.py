#!/usr/bin/env python3
"""Create review cohorts from completed scene annotations.

This script does not run a model. It reads JSONL annotations and writes two
one-row-per-split cohorts: children seen walking and infants seen supine.
Rows record how many models assessed and matched the exact split. These are
candidates for later fidgety-movement review, not ground-truth labels.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")


def load_records(outdir: Path):
    """Keep the latest parsed record per model/segmentation/video/split."""
    best = {}
    for path in sorted(outdir.glob("annotations_*.jsonl")):
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not record.get("parse_ok"):
                    continue
                key = (record.get("model") or "?",
                       record.get("segmentation") or "fixed",
                       record.get("video_id"), record.get("segment_index"))
                best[key] = record
    return best


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "video_id", "split", "segment_index", "start_sec", "end_sec",
        "n_models_match", "n_models_seen", "models_match"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_cohort(records: dict, predicate, min_models: int):
    """Return one review row per split, with model-agreement columns."""
    seen, matches = defaultdict(list), defaultdict(list)
    for (model, segmentation, video_id, index), record in records.items():
        key = (segmentation, video_id, index)
        seen[key].append((model, record))
        if predicate(record):
            matches[key].append((model, record))

    rows = []
    for (segmentation, video_id, index), hit_records in sorted(matches.items()):
        if len(hit_records) < min_models:
            continue
        _, record = hit_records[0]
        key = (segmentation, video_id, index)
        rows.append({
            "video_id": video_id or "",
            "video_name": record.get("video_name") or "",
            "segmentation": segmentation,
            "split": record.get("split") or "",
            "chunk": record.get("chunk") or "",
            "segment_index": index,
            "start_sec": f"{(record.get('start_sec') or 0):.2f}",
            "end_sec": f"{(record.get('end_sec') or 0):.2f}",
            "timestamp": record.get("timestamp") or "",
            "url": record.get("url") or "",
            "url_at": record.get("url_at") or "",
            "clip_path": record.get("video") or "",
            "n_models_match": len(hit_records),
            "n_models_seen": len(seen[key]),
            "models_match": ";".join(sorted(model for model, _ in hit_records)),
            "models_seen": ";".join(sorted(model for model, _ in seen[key])),
            "num_infants": record.get("num_infants"),
            "num_children": record.get("num_children"),
            "infant_posture": record.get("infant_posture") or "",
            "infant_actions": ";".join(record.get("infant_actions") or []),
            "child_actions": ";".join(record.get("child_actions") or []),
            "description": (record.get("description") or "").strip(),
        })
    return rows, len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=ROOT / "outputs")
    ap.add_argument("--only", help="comma-separated video IDs")
    ap.add_argument("--models", help="comma-separated model tags")
    ap.add_argument("--segmentation", default="scenes",
                    choices=["all", "fixed", "scenes"])
    ap.add_argument("--min-models", type=int, default=1,
                    help="require this many models to match a split")
    args = ap.parse_args()

    if args.min_models < 1:
        raise SystemExit("--min-models must be at least 1")
    records = load_records(args.outdir)
    if args.segmentation != "all":
        records = {k: r for k, r in records.items() if k[1] == args.segmentation}
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        records = {k: r for k, r in records.items() if k[2] in wanted}
    if args.models:
        wanted = {x.strip() for x in args.models.split(",") if x.strip()}
        records = {k: r for k, r in records.items() if k[0] in wanted}
    if not records:
        raise SystemExit("no parsed annotations match the selected filters")

    # Old records predate child_actions and cannot be used as negative evidence.
    child_records = {k: r for k, r in records.items() if "child_actions" in r}
    walking, n_seen = make_cohort(
        child_records, lambda r: "walks" in (r.get("child_actions") or []),
        args.min_models)
    supine, _ = make_cohort(
        records,
        lambda r: r.get("num_infants", 0) > 0 and r.get("infant_posture") == "supine",
        args.min_models)

    walking_csv = args.outdir / "children_walking_segments.csv"
    supine_csv = args.outdir / "infants_supine_segments.csv"
    write_csv(walking_csv, walking)
    write_csv(supine_csv, supine)
    walking_ids = args.outdir / "children_walking_video_ids.txt"
    supine_ids = args.outdir / "infants_supine_video_ids.txt"
    walking_ids.write_text("\n".join(sorted({r["video_id"] for r in walking})) + ("\n" if walking else ""))
    supine_ids.write_text("\n".join(sorted({r["video_id"] for r in supine})) + ("\n" if supine else ""))

    print(f"annotated scene splits: {n_seen}")
    print(f"children walking: {len(walking)} -> {walking_csv}")
    print(f"infants supine : {len(supine)} -> {supine_csv}")
    if not walking:
        print("note: child walking needs new annotations containing child_actions")


if __name__ == "__main__":
    main()
