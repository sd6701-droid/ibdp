#!/usr/bin/env python
"""
Qwen3-VL-30B-A3B over the YouTube dataset, segment by segment. Offline vLLM --
no server, no port, no tunnel. One process, reads videos, writes JSONL.

Each video is cut into fixed-length windows (default 10s) and each window is
described independently. Windows are NOT extracted to disk: qwen_vl_utils takes
video_start/video_end and decodes only the requested range out of the full mp4.
A corpus of ~194 ten-minute videos is ~11.6k segments; writing those as temp
files would be pure waste.

Video titles come from manifest.tsv (id \t title \t duration \t url), which is
the same join key the fetcher used to name the mp4s.

Run inside the ibdp env on an A100-80. See 21_describe_videos.sbatch.

Sizing note: the 30B is an MoE -- 3.3B active, but all 30B of experts stay
resident. bf16 weights are ~61GB, so on an 80GB card there is ~19GB left for
KV cache and vision activations. That is why max_model_len is 32k here and not
the 128k in models.yaml: a 128k KV cache does not fit alongside the weights on
a single card. Raise it only if you drop to 2 GPUs.

WALLTIME: at ~11.6k segments this will not finish in one 4h job. That is fine
-- resume is per segment, so just resubmit until it stops printing "to process".
"""
import argparse, json, os
from pathlib import Path

# Must be set BEFORE qwen_vl_utils is imported. decord hangs on decode and is
# unmaintained; Qwen recommends torchcodec.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
# Compute nodes have no network. Fail loudly instead of hanging on a HF call.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")
MODEL = ROOT / "models/Qwen3-VL-30B-A3B-Instruct"

PROMPT = "Describe what happens in this video. Be specific and concrete."


def hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def load_manifest(path: Path) -> dict:
    """id -> {title, duration}. Duration may be missing; yt-dlp prints NA."""
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
                duration = None          # fall back to ffprobe-free skip below
            rows[vid] = {"title": title, "duration": duration}
    return rows


def segments(duration: float, window: float, min_tail: float):
    """[(start, end)] covering [0, duration). Runt tail merges into the previous
    window rather than becoming a 0.4s clip the model can say nothing about."""
    out = []
    start = 0.0
    while start < duration:
        end = min(start + window, duration)
        if duration - end < min_tail and end < duration:
            end = duration           # absorb the runt
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
                    help="0 = all VIDEOS (not segments). Use 1 for a smoke test.")
    ap.add_argument("--seconds", type=float, default=10.0, help="window length")
    ap.add_argument("--min-tail", type=float, default=2.0,
                    help="tails shorter than this merge into the previous window")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4,
                    help="segments per vLLM call. Higher = faster, more VRAM. "
                         "Drop to 1 if you OOM.")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    meta = load_manifest(args.manifest)
    print(f"manifest: {len(meta)} videos", flush=True)

    clips = sorted(args.videos.glob("*.mp4"))
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"no .mp4 under {args.videos}")

    # Resume: skip segments already written. The unit of work is a segment, not
    # a video -- an 11.6k-segment run will be killed by the walltime repeatedly,
    # and re-describing is pure waste.
    done = set()
    if args.out.exists():
        with args.out.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["video_id"], r["segment_index"]))

    # Build the full work list up front so the count in the log is honest.
    work = []          # (clip_path, video_id, title, seg_index, start, end)
    skipped = []
    for clip in clips:
        vid = clip.stem
        info = meta.get(vid)
        if info is None:
            skipped.append(f"{vid} (not in manifest)")
            continue
        if not info["duration"]:
            skipped.append(f"{vid} (no duration in manifest)")
            continue
        for i, (start, end) in enumerate(
                segments(info["duration"], args.seconds, args.min_tail)):
            if (vid, i) not in done:
                work.append((clip, vid, info["title"], i, start, end))

    if skipped:
        print(f"SKIPPED {len(skipped)} videos: {', '.join(skipped[:5])}"
              f"{' ...' if len(skipped) > 5 else ''}", flush=True)
    print(f"{len(work)} segments to process ({len(done)} already done)", flush=True)
    if not work:
        return

    processor = AutoProcessor.from_pretrained(str(MODEL))
    llm = LLM(
        model=str(MODEL),
        tensor_parallel_size=1,          # single A100-80. Was 2 in models.yaml (40GB cards).
        max_model_len=32768,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"video": 1},
        dtype="bfloat16",
    )
    sampling = SamplingParams(temperature=0.2, max_tokens=args.max_tokens)

    def build(clip, start, end):
        msgs = [{
            "role": "user",
            "content": [
                # total_pixels is the real memory knob, not fps. If a long
                # clip OOMs, lower total_pixels first -- you keep temporal
                # coverage and give up spatial detail, the better trade.
                {"type": "video", "video": str(clip), "fps": args.fps,
                 "video_start": start, "video_end": end,
                 "total_pixels": 20480 * 32 * 32},
                {"type": "text", "text": PROMPT},
            ],
        }]
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        _, video_inputs, video_kwargs = process_vision_info(
            msgs, return_video_kwargs=True)
        return {"prompt": text,
                "multi_modal_data": {"video": video_inputs[0]},
                "mm_processor_kwargs": video_kwargs}

    n = len(work)
    with args.out.open("a") as fout:
        for b0 in range(0, n, args.batch):
            batch = work[b0: b0 + args.batch]

            # Decode first: a corrupt clip must not poison the whole batch.
            reqs, keep = [], []
            for item in batch:
                clip, vid, title, i, start, end = item
                try:
                    reqs.append(build(clip, start, end))
                    keep.append(item)
                except Exception as e:
                    print(f"FAIL decode {vid} seg {i} [{start:.0f}-{end:.0f}s]: {e}",
                          flush=True)

            if not reqs:
                continue
            try:
                outs = llm.generate(reqs, sampling)
            except Exception as e:
                # One bad batch must not kill a 12h job. Retry the members one
                # at a time so a single poison clip costs one segment, not four.
                print(f"FAIL batch @{b0}: {e} -- retrying singly", flush=True)
                outs = []
                for r, item in zip(reqs, keep):
                    try:
                        outs.append(llm.generate([r], sampling)[0])
                    except Exception as e2:
                        print(f"  FAIL {item[1]} seg {item[3]}: {e2}", flush=True)
                        outs.append(None)

            for item, out in zip(keep, outs):
                if out is None:
                    continue
                clip, vid, title, i, start, end = item
                desc = out.outputs[0].text.strip()
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
                print(f"[{b0 + len(keep)}/{n}] {vid} seg {i} "
                      f"[{hhmmss(start)}-{hhmmss(end)}]: {desc[:60]}...", flush=True)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
