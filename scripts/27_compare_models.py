#!/usr/bin/env python
"""
Compare several models' annotations of the SAME video, segment by segment.

Reads every annotations_*.jsonl in --outdir, groups records by model, and lines
them up on (video_id, segment_index).

    python scripts/27_compare_models.py --video 0HkcGRBsPUM
    python scripts/27_compare_models.py --video VID --csv cmp.csv
    python scripts/27_compare_models.py --video VID --field num_adults

WHAT IT REPORTS, AND WHY EACH ONE IS HERE:

  agreement        pairwise, per field. The headline number. Two models that
                   agree on 95% of segments are interchangeable for this task
                   and you should run the cheaper one.
  disagreements    the actual segments, with each model's answer and its
                   description, so you can open the url_at link and see who was
                   right. An agreement rate with no way to adjudicate it is a
                   number you cannot act on.
  parse failures   a model whose JSON does not parse is not "slightly worse",
                   it is unusable at that rate -- counted separately and never
                   folded into agreement.
  self-consistency the model's own has_infant / num_infants / total conflicts,
                   already flagged per record by 26_describe_segments_hf.py.
  seconds/segment  from elapsed_sec. A 78B that is 4x slower for the same counts
                   is not worth the four GPUs.

RUNS ARE MERGED PER MODEL, LAST RECORD WINS. Resume means one model can span
several files; a re-annotated segment should count once, as its latest value.
Records under a different prompt hash are dropped with a warning rather than
silently compared -- two prompts are two different questions.
"""
import argparse, csv, json, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")
LEGACY_MODEL_TAG = "Qwen3-VL-30B-A3B-Instruct"

# Compared by default. Ordered: the counts first, since those are what the
# downstream aggregation actually uses.
FIELDS = ["num_infants", "num_adults", "num_children", "num_humans_total",
          "has_infant", "infant_visibility"]


def load(outdir: Path, video: str, want_sha: str | None):
    """-> ({model: {seg_index: record}}, prompt_shas seen, dropped count)."""
    by_model = defaultdict(dict)
    shas, dropped = set(), 0
    for f in sorted(outdir.glob("annotations_*.jsonl")):
        with f.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("video_id") != video:
                    continue
                shas.add(r.get("prompt_sha"))
                if want_sha and r.get("prompt_sha") != want_sha:
                    dropped += 1
                    continue
                # Files are read in name order and later records overwrite
                # earlier ones, so a re-run of a segment wins over the original.
                by_model[r.get("model", LEGACY_MODEL_TAG)][r["segment_index"]] = r
    return by_model, shas, dropped


def fmt(r, field):
    if r is None:
        return "-"
    if not r.get("parse_ok"):
        return "PARSE_FAIL"
    v = r.get(field)
    return str(v).lower() if isinstance(v, bool) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=ROOT / "outputs")
    ap.add_argument("--video", required=True, help="video id to compare")
    ap.add_argument("--models", default=None,
                    help="comma-separated model tags (default: all found)")
    ap.add_argument("--prompt-sha", default=None,
                    help="only compare records under this prompt hash "
                         "(default: the most common one present)")
    ap.add_argument("--field", default=None,
                    help=f"single field to tabulate (default: {FIELDS[0]})")
    ap.add_argument("--max-rows", type=int, default=40,
                    help="0 = print every segment")
    ap.add_argument("--csv", type=Path, default=None,
                    help="also write the full per-segment table here")
    args = ap.parse_args()

    if not args.outdir.is_dir():
        raise SystemExit(f"no such outdir: {args.outdir}")

    # Pick the dominant prompt hash first, then reload filtered to it. Comparing
    # across prompts silently would be comparing answers to different questions.
    probe, shas, _ = load(args.outdir, args.video, None)
    if not probe:
        raise SystemExit(f"no records for video {args.video} in {args.outdir}")
    if args.prompt_sha:
        sha = args.prompt_sha
    else:
        counts = defaultdict(int)
        for recs in probe.values():
            for r in recs.values():
                counts[r.get("prompt_sha")] += 1
        sha = max(counts, key=counts.get)

    by_model, _, dropped = load(args.outdir, args.video, sha)
    if dropped:
        print(f"NOTE: ignored {dropped} record(s) under a different prompt "
              f"hash (comparing only {sha})\n", file=sys.stderr)

    models = sorted(by_model)
    if args.models:
        want = [m.strip() for m in args.models.split(",") if m.strip()]
        missing = [m for m in want if m not in by_model]
        if missing:
            raise SystemExit(f"no records for: {', '.join(missing)}\n"
                             f"present: {', '.join(models)}")
        models = want
    if len(models) < 2:
        raise SystemExit(f"need >=2 models to compare, found: {models}")

    field = args.field or FIELDS[0]
    fields = [field] if args.field else FIELDS

    # Union, not intersection: a segment one model skipped or crashed on is
    # exactly the segment worth seeing. It shows as "-".
    segs = sorted({s for m in models for s in by_model[m]})

    # From the FILTERED set, and tolerant of a model that has no segment 0:
    # reading it off `probe` picks up records under other prompt hashes, which
    # is how this printed an empty title.
    title = next((r.get("video_name", "") for m in models
                  for _, r in sorted(by_model[m].items())
                  if r.get("video_name")), "")
    print(f"video    : {args.video}  {title[:60]}")
    print(f"prompt   : {sha}")
    print(f"segments : {len(segs)}")
    print(f"models   : {len(models)}\n")

    # ---- per-model health --------------------------------------------------
    print(f"{'model':32s} {'segs':>5s} {'parse_fail':>10s} "
          f"{'inconsistent':>12s} {'s/seg':>7s}")
    for m in models:
        recs = list(by_model[m].values())
        bad = sum(1 for r in recs if not r.get("parse_ok"))
        inc = sum(1 for r in recs if r.get("inconsistent"))
        el = [r["elapsed_sec"] for r in recs if r.get("elapsed_sec") is not None]
        el_s = f"{sum(el) / len(el):.1f}" if el else "n/a"
        print(f"{m:32s} {len(recs):5d} {bad:10d} {inc:12d} {el_s:>7s}")
    print()

    # ---- pairwise agreement ------------------------------------------------
    # Segments where EITHER model failed to parse are excluded from the
    # denominator: a parse failure is not a disagreement about content, and
    # counting it as one would flatter whichever model failed more often.
    print("pairwise agreement (parse failures excluded)")
    for f in fields:
        print(f"  {f}")
        for a, b in combinations(models, 2):
            same = tot = 0
            for s in segs:
                ra, rb = by_model[a].get(s), by_model[b].get(s)
                if not (ra and rb and ra.get("parse_ok") and rb.get("parse_ok")):
                    continue
                tot += 1
                same += ra.get(f) == rb.get(f)
            pct = f"{100 * same / tot:5.1f}%" if tot else "  n/a"
            print(f"    {a:30s} vs {b:30s} {pct}  ({same}/{tot})")
    print()

    # ---- disagreements -----------------------------------------------------
    rows = []
    for s in segs:
        recs = [by_model[m].get(s) for m in models]
        vals = [fmt(r, field) for r in recs]
        ok = [v for v in vals if v not in ("-", "PARSE_FAIL")]
        rows.append({"segment": s,
                     "timestamp": next((r["timestamp"] for r in recs if r), ""),
                     "url_at": next((r.get("url_at", "") for r in recs if r), ""),
                     "agree": len(set(ok)) <= 1,
                     "vals": vals,
                     "recs": recs})

    disagree = [r for r in rows if not r["agree"]]
    print(f"'{field}' disagreements: {len(disagree)} of {len(rows)} segments")
    shown = disagree if args.max_rows == 0 else disagree[: args.max_rows]
    w = max((len(m) for m in models), default=10)
    for r in shown:
        print(f"\n  seg {r['segment']:3d} [{r['timestamp']}]  {r['url_at']}")
        for m, v, rec in zip(models, r["vals"], r["recs"]):
            desc = (rec or {}).get("description", "") if rec else ""
            print(f"    {m:{w}s}  {field}={v:12s} {desc[:90]}")
    if len(disagree) > len(shown):
        print(f"\n  ... {len(disagree) - len(shown)} more "
              f"(--max-rows 0 for all)")

    # ---- csv ---------------------------------------------------------------
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            cols = ["segment", "timestamp", "url_at"]
            for m in models:
                cols += [f"{m}:{f}" for f in fields] + [f"{m}:description"]
            wtr = csv.DictWriter(fh, fieldnames=cols)
            wtr.writeheader()
            for r in rows:
                row = {"segment": r["segment"], "timestamp": r["timestamp"],
                       "url_at": r["url_at"]}
                for m, rec in zip(models, r["recs"]):
                    for f in fields:
                        row[f"{m}:{f}"] = fmt(rec, f)
                    row[f"{m}:description"] = (rec or {}).get("description", "")
                wtr.writerow(row)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
