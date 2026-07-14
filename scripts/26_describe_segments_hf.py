#!/usr/bin/env python
"""
Segment-wise STRUCTURED annotation of the YouTube dataset via transformers.

Each video is cut into fixed-length windows (default 10s) and each window is
annotated independently. Windows are NOT extracted to disk: qwen_vl_utils takes
video_start/video_end and decodes only the requested range out of the full mp4.

The model is asked for JSON, not prose. Fields per segment:

    has_infant          bool
    num_infants         int
    has_adult           bool
    num_adults          int
    num_humans_total    int     (infants + adults; the model's own total)
    infant_visibility   "full_body" | "partial_body" | "not_visible"
    visible_infant_parts  list of head/face/torso/arms/hands/legs/feet
    description         one or two sentences

WHY JSON AND NOT PROSE: a free-text "describe this video" prompt produced
100-200 word narrations for a 10-second clip, most of the ~12s/segment being
generation rather than decode -- and it degenerated into repetition loops under
greedy decoding (one segment repeated "the baby is still splashing" until it hit
the token cap). Structured output is shorter, faster, parseable, and cannot
ramble.

Resume is per SEGMENT, so a walltime kill costs nothing. Rerun to continue.

    python scripts/26_describe_segments_hf.py --limit 2    # measure first
    python scripts/26_describe_segments_hf.py              # then the rest

NOTE: torchcodec needs the CUDA 12 NPP libs on LD_LIBRARY_PATH. Per shell:
    SITE=$(python -c "import site; print(site.getsitepackages()[0])")
    export LD_LIBRARY_PATH="$SITE/nvidia/npp/lib:$SITE/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
"""
import argparse, hashlib, json, os, re, time
from datetime import datetime
from pathlib import Path

# Must precede the qwen_vl_utils import. decord hangs on decode and is
# unmaintained; Qwen recommends torchcodec.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")
MODEL = ROOT / "models/Qwen3-VL-30B-A3B-Instruct"

# "infant" is defined explicitly. Left to itself the model drifts between
# "baby", "toddler" and "young child" across segments, which makes the counts
# useless to aggregate.
PROMPT = """Analyse this video clip and answer with JSON ONLY -- no markdown, no \
commentary, no code fences.

An INFANT is a child who cannot yet walk unaided: roughly under 18 months. A \
walking toddler or older child is NOT an infant; count them as an adult only if \
they are clearly an adult, otherwise exclude them from both counts.

Return exactly this shape:
{
  "has_infant": true or false,
  "num_infants": integer,
  "has_adult": true or false,
  "num_adults": integer,
  "num_humans_total": integer,
  "infant_visibility": "full_body" or "partial_body" or "not_visible",
  "visible_infant_parts": ["head","face","torso","arms","hands","legs","feet"],
  "description": "one or two dense sentences -- see below"
}

Rules:
- Count DISTINCT people visible anywhere in the clip, not per frame.
- "full_body": the whole infant, head to feet, is visible at some point.
- "partial_body": only some of the infant is in frame (e.g. head and torso).
- "not_visible": no infant in the clip. Then visible_infant_parts is [].
- visible_infant_parts lists only parts you actually see; use [] if no infant.
- If there is no infant, has_infant is false and num_infants is 0.

The "description" must be AT MOST TWO SENTENCES, but every clause must carry
information. Prefer concrete specifics -- who, where, what they physically do,
what they touch -- over adjectives and mood. Name the infant's posture (lying,
sitting, crawling, standing, held) and the main action. Report only what is
actually visible; never guess or invent. Do not pad, do not editorialise, do
not repeat yourself. Two tight sentences, then stop.

Good: "An infant in a red shirt sits on a concrete driveway, gripping a green
garden hose with both hands while water sprays over their legs. A white dog
approaches from the left and the infant turns their head to watch it."

Bad: "The video begins with a heartwarming scene. The atmosphere is playful and
cheerful as the adorable child enjoys a wonderful moment outdoors." """

VALID_VISIBILITY = {"full_body", "partial_body", "not_visible"}
VALID_PARTS = {"head", "face", "torso", "arms", "hands", "legs", "feet"}


def hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def load_manifest(path: Path) -> dict:
    """id -> {title, duration, url}. yt-dlp prints NA for missing durations."""
    rows = {}
    with path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            vid, title, dur, url = parts[0], parts[1], parts[2], parts[3]
            try:
                duration = float(dur)
            except ValueError:
                duration = None
            rows[vid] = {"title": title, "duration": duration, "url": url}
    return rows


def segments(duration: float, window: float, min_tail: float):
    """[(start, end)] covering [0, duration). A runt tail merges into the previous
    window rather than becoming a 0.4s clip the model can say nothing about."""
    out, start = [], 0.0
    while start < duration:
        end = min(start + window, duration)
        if duration - end < min_tail and end < duration:
            end = duration
        out.append((start, end))
        start = end
    return out


def parse_annotation(raw: str) -> dict:
    """Model text -> validated dict. Never raises: a segment that returns junk
    records parse_ok=false and keeps its raw text, rather than killing the run or
    silently writing zeros that look like real observations."""
    out = {"parse_ok": False, "raw": raw}

    # It is told not to fence the JSON, but it sometimes does anyway.
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return out
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out

    def as_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1")
        return bool(v)

    def as_int(v):
        try:
            return max(0, int(v))
        except (TypeError, ValueError):
            return 0

    vis = str(d.get("infant_visibility", "")).strip().lower()
    if vis not in VALID_VISIBILITY:
        vis = None

    parts = d.get("visible_infant_parts") or []
    if not isinstance(parts, list):
        parts = []
    parts = [p for p in (str(x).strip().lower() for x in parts) if p in VALID_PARTS]

    ann = {
        "has_infant": as_bool(d.get("has_infant")),
        "num_infants": as_int(d.get("num_infants")),
        "has_adult": as_bool(d.get("has_adult")),
        "num_adults": as_int(d.get("num_adults")),
        "num_humans_total": as_int(d.get("num_humans_total")),
        "infant_visibility": vis,
        "visible_infant_parts": parts,
        "description": str(d.get("description", "")).strip(),
    }

    # Self-consistency: the model sometimes says has_infant=true, num_infants=0.
    # Trust the count and flag the disagreement rather than quietly picking one.
    ann["inconsistent"] = bool(
        ann["has_infant"] != (ann["num_infants"] > 0)
        or ann["has_adult"] != (ann["num_adults"] > 0)
        or (not ann["has_infant"] and ann["visible_infant_parts"])
    )
    ann["parse_ok"] = True
    return ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=ROOT / "youtube_dataset/videos")
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "youtube_dataset/manifest.tsv")
    # Default: a NEW timestamped file per run, so a run never mutates an older
    # one. Pass --out explicitly to override.
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=ROOT / "outputs")
    # Resume scans every annotations_*.jsonl in --outdir, but only reuses records
    # whose prompt hash matches the CURRENT prompt. Change the prompt and stale
    # records are ignored and regenerated, rather than silently kept.
    ap.add_argument("--resume", action="store_true",
                    help="skip segments already done under the same prompt")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all VIDEOS (not segments). Use 2 to measure speed.")
    ap.add_argument("--seconds", type=float, default=10.0, help="window length")
    ap.add_argument("--min-tail", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=2.0)
    # 256: the structured fields are tiny and `description` is capped at two
    # sentences, so this is ample. It must stay comfortably ABOVE the real
    # output length -- a description truncated mid-sentence yields JSON with no
    # closing brace, parse_ok goes false, and the whole segment is lost, counts
    # included. Generation dominates per-segment cost, so this is also the
    # cheapest lever on total runtime.
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU -- needs an A100 allocation, not a login node.")

    # Stamped into every record. Two runs with different prompts are then
    # distinguishable after the fact, and --resume can tell fresh from stale.
    prompt_sha = hashlib.sha256(PROMPT.encode()).hexdigest()[:8]

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = args.outdir / f"annotations_{stamp}_{prompt_sha}.jsonl"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    meta = load_manifest(args.manifest)
    print(f"manifest: {len(meta)} videos", flush=True)
    print(f"prompt:   {prompt_sha}", flush=True)
    print(f"writing:  {args.out}", flush=True)

    clips = sorted(args.videos.glob("*.mp4"))
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"no .mp4 under {args.videos}")

    # Resume across ALL prior runs, not just one file -- but only honour records
    # written under the SAME prompt. A changed prompt makes old records stale,
    # and silently skipping them would leave a corpus that is half one prompt
    # and half another with nothing in the data to say which.
    done = set()
    if args.resume:
        stale = 0
        for prev in sorted(args.outdir.glob("annotations_*.jsonl")):
            with prev.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("prompt_sha") != prompt_sha:
                        stale += 1
                        continue
                    if r.get("parse_ok"):   # retry anything that failed to parse
                        done.add((r["video_id"], r["segment_index"]))
        print(f"resume:   {len(done)} done, {stale} stale (different prompt)",
              flush=True)

    work, skipped = [], []
    for clip in clips:
        vid = clip.stem
        info = meta.get(vid)
        if info is None:
            skipped.append(f"{vid} (not in manifest)")
            continue
        if not info["duration"]:
            skipped.append(f"{vid} (no duration)")
            continue
        for i, (start, end) in enumerate(
                segments(info["duration"], args.seconds, args.min_tail)):
            if (vid, i) not in done:
                work.append((clip, vid, info["title"], info["url"], i, start, end))

    if skipped:
        print(f"SKIPPED {len(skipped)}: {', '.join(skipped[:5])}"
              f"{' ...' if len(skipped) > 5 else ''}", flush=True)
    print(f"{len(work)} segments to process ({len(done)} already done)", flush=True)
    if not work:
        return

    processor = AutoProcessor.from_pretrained(str(MODEL))
    print("loading weights (~53GB off GPFS, ~1min)...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL), dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    print(f"ready on {torch.cuda.get_device_name(0)}\n", flush=True)

    n = len(work)
    t_start = time.time()
    ok = bad = 0

    with args.out.open("a") as fout:
        for k, (clip, vid, title, url, i, start, end) in enumerate(work, 1):
            msgs = [{
                "role": "user",
                "content": [
                    # total_pixels is the real memory knob, not fps. If a clip
                    # OOMs, lower total_pixels first -- you keep temporal
                    # coverage and give up spatial detail, the better trade.
                    {"type": "video", "video": str(clip), "fps": args.fps,
                     "video_start": start, "video_end": end,
                     "total_pixels": 20480 * 32 * 32},
                    {"type": "text", "text": PROMPT},
                ],
            }]
            t0 = time.time()
            try:
                images, videos, video_kwargs = process_vision_info(
                    msgs, return_video_kwargs=True)

                # fps comes back as a LIST (one per video in the batch) -- the
                # processor validates it as a scalar. It feeds MRoPE's temporal
                # positions, so use the ACTUAL sampled rate, not args.fps.
                fps = video_kwargs.get("fps")
                if isinstance(fps, (list, tuple)):
                    fps = fps[0] if fps else args.fps

                text = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    images=images if images else None,
                    videos=videos,
                    fps=fps,
                    return_tensors="pt",
                ).to(model.device)

                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,          # deterministic: this is extraction
                        repetition_penalty=1.05,  # greedy decoding looped without it
                    )
                raw = processor.batch_decode(
                    out[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True)[0].strip()
            except Exception as e:
                print(f"[{k}/{n}] FAIL {vid} seg {i}: {e}", flush=True)
                continue

            ann = parse_annotation(raw)
            if ann["parse_ok"]:
                ok += 1
            else:
                bad += 1

            rec = {
                "prompt_sha": prompt_sha,
                "video_id": vid,
                "video_name": title,
                "url": url,
                # Deep link straight to this segment -- makes spot-checking an
                # annotation a click instead of a scrub.
                "url_at": f"{url}&t={int(start)}s",
                "video": str(clip),
                "segment_index": i,
                "start_sec": round(start, 2),
                "end_sec": round(end, 2),
                "timestamp": f"{hhmmss(start)}-{hhmmss(end)}",
                **ann,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()   # survive a walltime kill

            dt = time.time() - t0
            rate = (time.time() - t_start) / max(ok + bad, 1)
            if ann["parse_ok"]:
                flag = " INCONSISTENT" if ann["inconsistent"] else ""
                summary = (f"infant={ann['num_infants']} adult={ann['num_adults']} "
                           f"vis={ann['infant_visibility']}{flag}")
            else:
                summary = "PARSE FAILED"
            print(f"[{k}/{n}] {vid} seg {i} [{hhmmss(start)}-{hhmmss(end)}] "
                  f"{dt:.1f}s (avg {rate:.1f}s/seg, "
                  f"eta {(n - k) * rate / 3600:.1f}h): {summary}", flush=True)

    total = time.time() - t_start
    print(f"\n{ok} parsed, {bad} unparseable, of {n} segments "
          f"in {total/60:.1f} min ({total/max(ok+bad,1):.1f}s each)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
