#!/usr/bin/env python
"""
Segment-wise video description via plain transformers -- the vLLM-free twin of
21_describe_videos.py.

Same segment logic, same manifest join, same resume file, SAME JSONL schema, so
the two are interchangeable and share progress. When the CUDA 12 vLLM wheel
lands, switch back to 21 -- it will pick up exactly where this left off.

WHY: vLLM's wheels are CUDA 13; the BigPurple driver caps at 12.9, so
`import vllm` dies on libcudart.so.13. transformers runs on the cu128 torch that
already works.

THE COST: transformers has no continuous batching. Expect this to be several
times slower per segment than vLLM would be. DO NOT guess the total -- run with
--limit 1 first and read the seconds/segment it prints. That number times the
segment count is your real budget, and it decides whether this plan is viable
as-is or whether the window needs to be longer.

    python scripts/26_describe_segments_hf.py --limit 1     # measure first
    python scripts/26_describe_segments_hf.py               # then the rest
"""
import argparse, json, os, time
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

PROMPT = "Describe what happens in this video. Be specific and concrete."


def hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def load_manifest(path: Path) -> dict:
    """id -> {title, duration}. yt-dlp prints NA for missing durations."""
    rows = {}
    with path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            vid, title, dur = parts[0], parts[1], parts[2]
            try:
                duration = float(dur)
            except ValueError:
                duration = None
            rows[vid] = {"title": title, "duration": duration}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=ROOT / "youtube_dataset/videos")
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "youtube_dataset/manifest.tsv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/descriptions.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all VIDEOS (not segments). Use 1 to measure speed.")
    ap.add_argument("--seconds", type=float, default=10.0, help="window length")
    ap.add_argument("--min-tail", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU -- needs an A100 allocation, not a login node.")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    meta = load_manifest(args.manifest)
    print(f"manifest: {len(meta)} videos", flush=True)

    clips = sorted(args.videos.glob("*.mp4"))
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"no .mp4 under {args.videos}")

    # Resume is per SEGMENT, not per video: an 11k-segment run will be killed by
    # the walltime repeatedly, and re-describing is pure waste.
    done = set()
    if args.out.exists():
        with args.out.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["video_id"], r["segment_index"]))

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
                work.append((clip, vid, info["title"], i, start, end))

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
    ok = 0

    with args.out.open("a") as fout:
        for k, (clip, vid, title, i, start, end) in enumerate(work, 1):
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
                # process_vision_info decodes ONLY [start, end) out of the full
                # mp4 -- no temp files, no re-muxing 11k clips to disk.
                images, videos, video_kwargs = process_vision_info(
                    msgs, return_video_kwargs=True)
                inputs = processor.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                    video=videos, **video_kwargs,
                ).to(model.device)

                with torch.inference_mode():
                    out = model.generate(**inputs,
                                         max_new_tokens=args.max_new_tokens,
                                         do_sample=False)
                desc = processor.batch_decode(
                    out[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True)[0].strip()
            except Exception as e:
                # One corrupt clip must not kill a multi-hour job.
                print(f"[{k}/{n}] FAIL {vid} seg {i}: {e}", flush=True)
                continue

            dt = time.time() - t0
            ok += 1

            fout.write(json.dumps({
                "video_id": vid,
                "video_name": title,
                "video": str(clip),
                "segment_index": i,
                "start_sec": round(start, 2),
                "end_sec": round(end, 2),
                "timestamp": f"{hhmmss(start)}-{hhmmss(end)}",
                "description": desc,
            }) + "\n")
            fout.flush()   # survive a walltime kill

            # The honest number: measured, not guessed. Extrapolate from THIS.
            rate = (time.time() - t_start) / ok
            print(f"[{k}/{n}] {vid} seg {i} [{hhmmss(start)}-{hhmmss(end)}] "
                  f"{dt:.1f}s (avg {rate:.1f}s/seg, "
                  f"eta {(n - k) * rate / 3600:.1f}h): {desc[:60]}...", flush=True)

    total = time.time() - t_start
    print(f"\n{ok}/{n} segments in {total/60:.1f} min "
          f"({total/max(ok,1):.1f}s per segment)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
