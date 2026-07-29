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
          "has_infant", "infant_visibility",
          # Scene fields, closed-vocabulary strings compared by equality just
          # like infant_visibility. "objects" is deliberately absent: a free
          # list where "toy car" vs "toy" counts as total disagreement says
          # nothing useful about agreement.
          "location", "surface", "camera_distance", "lighting",
          "infant_clothing", "background_complexity", "camera_motion",
          "image_quality"]


def load(outdir: Path, video: str, want_sha: str | None,
         want_seg: str | None = None):
    """-> ({model: {seg_index: record}}, prompt_shas seen, segmentations seen,
    dropped-for-prompt count, dropped-for-segmentation count).

    want_seg filters on how segments were delimited -- "fixed" (10s grid) or
    "scenes" (shot-boundary splits from scripts/32). The two number segments
    DIFFERENTLY over the same video under the SAME prompt_sha: grid indices are
    0-based 10s windows, split indices 1-based edit-point clips. Joining across
    them on segment_index would compare answers about different stretches of
    video and call the disagreement a model property. Records from before the
    field existed are all grid runs, hence the "fixed" default.
    """
    by_model = defaultdict(dict)
    shas, segs = set(), set()
    dropped_sha = dropped_seg = 0
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
                segs.add(r.get("segmentation") or "fixed")
                if want_seg and (r.get("segmentation") or "fixed") != want_seg:
                    dropped_seg += 1
                    continue
                if want_sha and r.get("prompt_sha") != want_sha:
                    dropped_sha += 1
                    continue
                # Files are read in name order and later records overwrite
                # earlier ones, so a re-run of a segment wins over the original.
                by_model[r.get("model", LEGACY_MODEL_TAG)][r["segment_index"]] = r
    return by_model, shas, segs, dropped_sha, dropped_seg


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
    ap.add_argument("--segmentation", default="auto",
                    choices=["auto", "fixed", "scenes"],
                    help="which segmentation's records to compare: 'fixed' "
                         "(10s grid) or 'scenes' (shot-boundary splits). "
                         "'auto' picks the most common one present. Unlike "
                         "prompt hashes the two are NEVER mixed: their "
                         "segment indices describe different stretches of "
                         "the video.")
    ap.add_argument("--strict-prompt", action="store_true",
                    help="drop every record whose prompt hash is not the "
                         "dominant one. Off by default so Qwen3-Omni, whose "
                         "prompt has an audio addendum and therefore a "
                         "different hash, is still comparable on the shared "
                         "fields.")
    ap.add_argument("--field", default=None,
                    help=f"single field to tabulate (default: {FIELDS[0]})")
    ap.add_argument("--max-rows", type=int, default=40,
                    help="0 = print every segment")
    ap.add_argument("--csv", type=Path, default=None,
                    help="also write the full per-segment table here")
    ap.add_argument("--wandb", action="store_true",
                    help="log the comparison to W&B, in the same group as the "
                         "per-model runs")
    # Must match scripts/26's default, or the comparison lands in a different
    # W&B project than the runs it compares and looks like it never logged.
    ap.add_argument("--wandb-project", default="ibdp")
    ap.add_argument("--wandb-mode", default="online",
                    choices=["offline", "online", "disabled"])
    ap.add_argument("--wandb-dir", type=Path, default=None,
                    help="where offline runs are written (default: --outdir)")
    args = ap.parse_args()

    if not args.outdir.is_dir():
        raise SystemExit(f"no such outdir: {args.outdir}")

    # Pick the dominant prompt hash first, then reload filtered to it. Comparing
    # across prompts silently would be comparing answers to different questions.
    # seg_modes, not "segs": that name is taken further down for the joined
    # SEGMENT INDICES, and shadowing it here invites exactly the confusion
    # this filter exists to prevent.
    probe, shas, seg_modes, _, _ = load(args.outdir, args.video, None)
    if not probe:
        raise SystemExit(f"no records for video {args.video} in {args.outdir}")

    # Segmentation is resolved BEFORE the prompt hash and filtered HARD. Grid
    # and split records coexist for the same video under the same prompt (the
    # prompt did not change; how the video was diced did), and their
    # segment_index spaces overlap without corresponding.
    if args.segmentation != "auto":
        seg = args.segmentation
    else:
        seg_counts = defaultdict(int)
        for recs in probe.values():
            for r in recs.values():
                seg_counts[r.get("segmentation") or "fixed"] += 1
        seg = max(seg_counts, key=seg_counts.get)
    if len(seg_modes) > 1:
        print(f"NOTE: both segmentations present for {args.video}; comparing "
              f"'{seg}' only (--segmentation to pick the other)\n",
              file=sys.stderr)

    if args.prompt_sha:
        sha = args.prompt_sha
    else:
        counts = defaultdict(int)
        for recs in probe.values():
            for r in recs.values():
                if (r.get("segmentation") or "fixed") != seg:
                    continue   # the other segmentation must not vote
                counts[r.get("prompt_sha")] += 1
        if not counts:
            raise SystemExit(f"no '{seg}' records for {args.video}")
        sha = max(counts, key=counts.get)

    # Qwen3-Omni's prompt carries an audio addendum, so its prompt_sha differs
    # from every video-only model's BY DESIGN. Filtering to the dominant hash
    # therefore drops Omni entirely -- and since 28_run_all_models.sh passes
    # every model it ran via --models, the old behaviour was to abort the whole
    # comparison over it. The audio-vs-video question is the reason Omni is in
    # the list at all, so mixed prompts are allowed by default and the shared
    # fields are compared; --strict-prompt restores hash-exact filtering.
    by_model, _, _, dropped, dropped_seg = load(
        args.outdir, args.video, sha if args.strict_prompt else None, seg)
    if dropped_seg:
        print(f"NOTE: ignored {dropped_seg} record(s) from the other "
              f"segmentation (comparing only '{seg}')\n", file=sys.stderr)
    if dropped:
        print(f"NOTE: ignored {dropped} record(s) under a different prompt "
              f"hash (comparing only {sha})\n", file=sys.stderr)

    # Which prompt each model actually answered. Printed below, because a
    # comparison across two prompts is only honest if it says so.
    model_sha = {m: sorted({r.get("prompt_sha") for r in recs.values()})
                 for m, recs in by_model.items()}

    models = sorted(by_model)
    if args.models:
        want = [m.strip() for m in args.models.split(",") if m.strip()]
        missing = [m for m in want if m not in by_model]
        if missing:
            # Warn and continue. A model that produced nothing is a reason to
            # exclude that model, not to throw away the other five.
            print(f"WARNING: no records for {', '.join(missing)} -- excluded.",
                  file=sys.stderr)
            if args.strict_prompt:
                print(f"         (--strict-prompt is on; they may simply be "
                      f"under a different prompt hash)", file=sys.stderr)
            want = [m for m in want if m in by_model]
        if len(want) < 2:
            raise SystemExit(f"only {len(want)} model(s) left after exclusions; "
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
    shas_used = sorted({s for m in models for s in model_sha.get(m, [])})
    if len(shas_used) > 1:
        print(f"prompt   : MIXED -- " + ", ".join(
            f"{m}={'/'.join(x or '?' for x in model_sha[m])}" for m in models))
        print(f"           shared fields are comparable; the differing prompt "
              f"is an addendum, not a different question")
    else:
        print(f"prompt   : {sha}")
    print(f"segments : {len(segs)}")
    print(f"models   : {len(models)}\n")

    # ---- per-model health --------------------------------------------------
    health = []
    print(f"{'model':32s} {'segs':>5s} {'parse_fail':>10s} "
          f"{'inconsistent':>12s} {'s/seg':>7s}")
    for m in models:
        recs = list(by_model[m].values())
        bad = sum(1 for r in recs if not r.get("parse_ok"))
        inc = sum(1 for r in recs if r.get("inconsistent"))
        el = [r["elapsed_sec"] for r in recs if r.get("elapsed_sec") is not None]
        sec = sum(el) / len(el) if el else None
        el_s = f"{sec:.1f}" if sec is not None else "n/a"
        print(f"{m:32s} {len(recs):5d} {bad:10d} {inc:12d} {el_s:>7s}")
        health.append([m, len(recs), bad, inc, sec])
    print()

    # ---- pairwise agreement ------------------------------------------------
    # Segments where EITHER model failed to parse are excluded from the
    # denominator: a parse failure is not a disagreement about content, and
    # counting it as one would flatter whichever model failed more often.
    print("pairwise agreement (parse failures excluded)")
    agree_rows = []
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
            agree_rows.append([f, a, b, 100 * same / tot if tot else None,
                               same, tot])
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

    # ---- wandb -------------------------------------------------------------
    if args.wandb:
        log_to_wandb(args, sha, seg, models, segs, rows, health, agree_rows, fields,
                     field)


def log_to_wandb(args, sha, seg, models, segs, rows, health, agree_rows, fields,
                 field):
    """One run per comparison, in the SAME group as the per-model runs.

    Group is the video id, so in the W&B UI the comparison sits alongside the
    models it compares rather than in a separate project you have to correlate
    by hand. job_type="compare" keeps it distinguishable from the annotation
    runs it summarises.

    Offline by default for the same reason as scripts/26: online mode does not
    fail fast on a firewalled node, it blocks and then drops the data."""
    try:
        import wandb
    except ImportError:
        raise SystemExit(
            "--wandb needs the wandb package.\n"
            "  ssh bigpurple-ln3 && conda activate ibdp && pip install wandb")
    import os

    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    os.environ.setdefault("WANDB_DIR", str(args.wandb_dir or args.outdir))

    tag = os.environ.get("RUN_TAG")
    default_name = f"compare--{args.video}" + (f"--{tag}" if tag else "")
    run = wandb.init(
        project=args.wandb_project,
        name=os.environ.get("WANDB_NAME", default_name),
        group=args.video,
        job_type="compare",
        config={"video": args.video, "prompt_sha": sha, "models": models,
                "segmentation": seg,
                "n_segments": len(segs), "fields": fields},
    )

    health_tbl = wandb.Table(
        columns=["model", "segments", "parse_fail", "inconsistent", "sec_per_segment"],
        data=[list(r) for r in health])
    agree_tbl = wandb.Table(
        columns=["field", "model_a", "model_b", "agreement_pct", "n_same", "n_compared"],
        data=[list(r) for r in agree_rows])

    seg_cols = ["segment", "timestamp", "url_at", "agree"]
    for m in models:
        seg_cols += [f"{m}:{field}", f"{m}:description"]
    seg_data = []
    for r in rows:
        d = [r["segment"], r["timestamp"], r["url_at"], bool(r["agree"])]
        for rec, v in zip(r["recs"], r["vals"]):
            d += [v, (rec or {}).get("description", "")]
        seg_data.append(d)
    seg_tbl = wandb.Table(columns=seg_cols, data=seg_data)

    # ---- one table, every model -------------------------------------------
    # LONG format: one row per (model, segment), with `model` as the first
    # column -- rather than the wide `segments` table above, which puts each
    # model in its own set of columns.
    #
    # Long is what you want for a single combined view: you can sort by
    # num_infants across all models at once, filter to one model, or group by
    # model, none of which a wide table supports. Wide is better only when you
    # are eyeballing one segment across models side by side, which is why both
    # are logged.
    #
    # audio_mode is carried explicitly so a blank audio column reads as "this
    # model had no audio" rather than "this model heard nothing" -- the same
    # trap as a null field in the JSONL.
    audio_keys = ["audio_events", "infant_vocalising", "speech_present",
                  "audio_inconsistent", "audio_description"]
    has_audio = any(any(k in (rec or {}) for k in audio_keys)
                    for r in rows for rec in r["recs"])

    long_cols = ["model", "audio_mode", "segment", "timestamp", "url_at",
                 "parse_ok", "num_infants", "num_children", "num_adults",
                 "num_humans_total", "infant_visibility", "visible_parts",
                 "inconsistent", "description"]
    if has_audio:
        long_cols += audio_keys
    long_cols += ["elapsed_sec", "prompt_sha"]

    long_data = []
    for r in rows:
        for model, rec in zip(models, r["recs"]):
            if rec is None:          # this model skipped or crashed on it
                continue
            parts = rec.get("visible_infant_parts") or []
            row = [
                model,
                "native" if rec.get("backend") == "qwen-omni" else
                ("transcript" if rec.get("audio_mode") == "transcript" else "none"),
                rec.get("segment_index"), rec.get("timestamp"),
                rec.get("url_at"), bool(rec.get("parse_ok")),
                rec.get("num_infants"), rec.get("num_children"),
                rec.get("num_adults"), rec.get("num_humans_total"),
                rec.get("infant_visibility"),
                ", ".join(parts) if isinstance(parts, list) else str(parts),
                bool(rec.get("inconsistent")),
                rec.get("description") or (rec.get("raw") or "")[:500],
            ]
            if has_audio:
                ev = rec.get("audio_events")
                row += [", ".join(ev) if isinstance(ev, list) else ev,
                        rec.get("infant_vocalising"), rec.get("speech_present"),
                        rec.get("audio_inconsistent"), rec.get("audio_description")]
            row += [rec.get("elapsed_sec"), rec.get("prompt_sha")]
            long_data.append(row)

    all_tbl = wandb.Table(columns=long_cols, data=long_data)

    run.log({"all_models": all_tbl, "health": health_tbl,
             "agreement": agree_tbl, "segments": seg_tbl})

    # Scalars too, not just tables: summary values are what W&B can sort and
    # chart across runs. A table alone cannot be compared between videos.
    summary = {"n_segments": len(segs),
               "n_disagreements": sum(1 for r in rows if not r["agree"])}
    for f in fields:
        pcts = [r[3] for r in agree_rows if r[0] == f and r[3] is not None]
        if pcts:
            summary[f"agreement/{f}_mean_pct"] = sum(pcts) / len(pcts)
            summary[f"agreement/{f}_min_pct"] = min(pcts)
    for m, n, bad, inc, sec in health:
        summary[f"model/{m}/parse_fail"] = bad
        summary[f"model/{m}/inconsistent"] = inc
        if sec is not None:
            summary[f"model/{m}/sec_per_segment"] = sec
    run.summary.update(summary)
    run_name = run.name
    run.finish()

    print(f"\nwandb: logged comparison as run {run_name}")
    if os.environ.get("WANDB_MODE") == "offline":
        print(f"  sync from a login node:  wandb sync "
              f"{args.wandb_dir or args.outdir}/wandb/offline-run-*")


if __name__ == "__main__":
    main()
