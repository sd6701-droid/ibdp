#!/usr/bin/env python
"""ONE description per (video, model): club every chunk's description together.

26_describe_segments_hf.py writes one record per chunk, so a video ends up as
50 scattered descriptions (and, for Qwen3-Omni, 50 audio descriptions too).
This script reassembles them: for each (model, video) it orders the chunks by
time, folds the visual and audio descriptions into one time-stamped timeline,
and writes a single record per video.

TWO LEVELS OF "SINGLE DESCRIPTION", because they answer different needs:

  timeline   Deterministic: the chunk descriptions concatenated in time order,
             duplicates collapsed. No model, no GPU, no way for it to invent
             anything -- what the annotators actually said, in one string.
             Always produced.

  summary    One coherent paragraph, produced by passing the timeline through
             a local LLM (--fuse <checkpoint>). Reads far better, but it is a
             GENERATED text: treat it as prose about the annotations, not as
             an annotation itself. Only produced when --fuse is given.

The two are separate fields in the same record, so downstream code can pick
the one whose failure mode it can live with.

Records join back to the per-chunk files on (model, video_id): nothing here
modifies or replaces the chunk-level corpus.

Usage:
    python scripts/35_video_summary.py                          # timelines only
    python scripts/35_video_summary.py --only 8yDn1uFbs4s
    python scripts/35_video_summary.py --models Qwen3-Omni-30B-A3B-Instruct
    python scripts/35_video_summary.py --fuse $ROOT/models/Qwen3-VL-30B-A3B-Instruct
                                                                # + fused summary
                                                                # (GPU node)
"""
import argparse, json, os, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")


def hhmmss(t: float) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"


# ---------------------------------------------------------------------------
# Collect chunks
# ---------------------------------------------------------------------------

def load_chunks(outdir: Path):
    """(model, segmentation, video_id) -> [records], time-ordered.

    Later files win on a (model, video, segment) collision: files sort by run
    counter, so a re-annotated segment supersedes its older self -- the same
    "latest run is the truth" rule --resume uses.

    Grid and scenes records are kept apart. They number different stretches of
    the same video with overlapping indices; folding them into one timeline
    would interleave two different segmentations of the same footage and
    describe half of it twice.
    """
    best = {}
    for f in sorted(outdir.glob("annotations_*.jsonl")):
        with f.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("parse_ok"):
                    continue
                key = (r.get("model", "?"),
                       r.get("segmentation") or "fixed",
                       r.get("video_id"),
                       r.get("segment_index"))
                best[key] = r

    groups = {}
    for (model, seg, vid, _), r in best.items():
        groups.setdefault((model, seg, vid), []).append(r)
    for recs in groups.values():
        recs.sort(key=lambda r: (r.get("start_sec") or 0,
                                 r.get("segment_index") or 0))
    return groups


# ---------------------------------------------------------------------------
# Timeline: the deterministic clubbed description
# ---------------------------------------------------------------------------

def build_timeline(recs: list) -> tuple[str, str]:
    """(visual timeline, audio timeline) as time-stamped lines.

    Consecutive chunks with an IDENTICAL description collapse into one line
    spanning both windows -- static footage makes the model repeat itself
    verbatim, and 'the baby is still splashing' five times is noise, not
    information. Non-identical near-duplicates are kept: deciding two
    sentences "mean the same" is a judgement call that belongs to --fuse,
    not to string handling.
    """
    def collapse(items):
        out = []
        for start, end, text in items:
            if out and out[-1][2] == text:
                out[-1] = (out[-1][0], end, text)
            else:
                out.append((start, end, text))
        return "\n".join(f"[{hhmmss(a)}-{hhmmss(b)}] {t}" for a, b, t in out)

    vis = [(r.get("start_sec") or 0, r.get("end_sec") or 0, d)
           for r in recs if (d := (r.get("description") or "").strip())]
    aud = [(r.get("start_sec") or 0, r.get("end_sec") or 0, d)
           for r in recs if (d := (r.get("audio_description") or "").strip())]
    return collapse(vis), collapse(aud)


# ---------------------------------------------------------------------------
# Optional LLM fusion
# ---------------------------------------------------------------------------

FUSE_PROMPT = """Below is a time-stamped log of what was observed in consecutive \
chunks of ONE video{audio_note}. Merge it into a single description of the whole \
video.

Rules:
- 3 to 6 sentences{audio_rule}.
- Only combine what the log states. Do NOT add details, moods or guesses that \
appear nowhere in it.
- Do not mention chunks, segments, timestamps or the log itself.
- Keep concrete specifics (who, where, actions, objects); drop repetition.

VISUAL LOG:
{visual}
{audio_block}"""


class Fuser:
    """Text-only pass through a local VL checkpoint. The vision tower is dead
    weight here, but reusing a model already on disk beats fetching a text-only
    one just for this."""

    def __init__(self, model_dir: Path, max_new_tokens: int):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if not torch.cuda.is_available():
            raise SystemExit("--fuse needs a GPU node; run without it for "
                             "timelines only.")
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        print(f"loading fuse model: {model_dir.name}", flush=True)
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir), dtype="auto",
            device_map="auto" if torch.cuda.device_count() > 1 else "cuda:0")
        self.model.eval()

    def fuse(self, visual: str, audio: str) -> str:
        prompt = FUSE_PROMPT.format(
            audio_note=(" (with notes on its soundtrack)" if audio else ""),
            audio_rule=(", plus 1-2 sentences on what is heard" if audio else ""),
            visual=visual,
            audio_block=(f"\nAUDIO LOG:\n{audio}" if audio else ""))
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt"
                                ).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, repetition_penalty=1.05)
        return self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=ROOT / "outputs")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <outdir>/video_summaries.jsonl")
    ap.add_argument("--only", default=None,
                    help="comma-separated video id(s)")
    ap.add_argument("--models", default=None,
                    help="comma-separated model tag(s); default all found")
    ap.add_argument("--fuse", type=Path, default=None,
                    help="local checkpoint for the fused paragraph (GPU node)")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--overwrite", action="store_true",
                    help="redo (model, video) pairs already in --out")
    args = ap.parse_args()

    out = args.out or (args.outdir / "video_summaries.jsonl")

    groups = load_chunks(args.outdir)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        groups = {k: v for k, v in groups.items() if k[2] in wanted}
    if args.models:
        wanted = {s.strip() for s in args.models.split(",") if s.strip()}
        groups = {k: v for k, v in groups.items() if k[0] in wanted}
    if not groups:
        raise SystemExit(f"no parsed chunk records under {args.outdir} match")

    # Resume by key. Fused and unfused records are DIFFERENT outputs of the
    # same input, so a timeline-only record does not block a later --fuse run.
    done = set()
    if out.exists() and not args.overwrite:
        with out.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r.get("model"), r.get("segmentation"),
                          r.get("video_id"), bool(r.get("fuse_model"))))

    fuser = Fuser(args.fuse, args.max_new_tokens) if args.fuse else None

    todo = [(k, v) for k, v in sorted(groups.items())
            if (k[0], k[1], k[2], fuser is not None) not in done]
    print(f"{len(todo)} (model, video) pairs to summarise "
          f"({len(groups) - len(todo)} already in {out.name})", flush=True)

    n_written = 0
    with out.open("a") as fout:
        for (model, seg, vid), recs in todo:
            t0 = time.time()
            visual, audio = build_timeline(recs)
            if not visual and not audio:
                continue

            summary = None
            if fuser is not None:
                try:
                    summary = fuser.fuse(visual, audio)
                except Exception as e:
                    # The timeline is still worth writing; the record just says
                    # the fusion failed rather than pretending it was not asked.
                    print(f"FUSE FAIL {model}/{vid}: {e}", flush=True)

            starts = [r.get("start_sec") or 0 for r in recs]
            ends = [r.get("end_sec") or 0 for r in recs]
            fout.write(json.dumps({
                "model": model,
                "segmentation": seg,
                "video_id": vid,
                "video_name": recs[0].get("video_name"),
                "url": recs[0].get("url"),
                "n_chunks": len(recs),
                # Omni skips >15s splits, so its span can be gappy: covered_sec
                # against span_sec is the honest coverage number.
                "span_sec": round(max(ends) - min(starts), 2),
                "covered_sec": round(sum(e - s for s, e in zip(starts, ends)), 2),
                "timeline": visual,
                "audio_timeline": audio or None,
                "summary": summary,
                "fuse_model": (args.fuse.name if fuser and summary else None),
                "elapsed_sec": round(time.time() - t0, 2),
            }) + "\n")
            fout.flush()
            n_written += 1
            print(f"  {model}  {vid}  {len(recs)} chunks -> 1"
                  f"{'  [fused]' if summary else ''}", flush=True)

    print(f"\nwrote {n_written} records -> {out}", flush=True)


if __name__ == "__main__":
    main()
