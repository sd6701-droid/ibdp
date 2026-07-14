#!/usr/bin/env python
"""
Same job as 21_describe_videos.py -- describe every clip, append JSONL -- but
against the vLLM server from 22_serve_vllm.sbatch instead of loading the model
in-process.

Needs no GPU. Run it from the login node (or any node that can reach the
compute node): the server does the work, this just feeds it paths.

    sbatch scripts/22_serve_vllm.sbatch      # takes a few min to load 61GB
    python scripts/23_describe_via_api.py --limit 2

Videos are passed as file:// URLs, not base64 -- the server reads the same GPFS
mount we do. That only works because 22_serve_vllm.sbatch passes
--allowed-local-media-path.
"""
import argparse, json, time
from pathlib import Path

from openai import OpenAI

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")
ENDPOINT_FILE = ROOT / "outputs/endpoint.txt"

PROMPT = "Describe what happens in this video. Be specific and concrete."


def wait_for_server(base_url: str, timeout: float) -> OpenAI:
    """Loading 61GB of weights takes minutes. Poll /models until it answers."""
    client = OpenAI(base_url=base_url, api_key="EMPTY")   # vLLM ignores the key
    deadline = time.time() + timeout
    while True:
        try:
            client.models.list()
            return client
        except Exception as e:
            if time.time() > deadline:
                raise SystemExit(
                    f"server at {base_url} never came up ({timeout:.0f}s): {e}\n"
                    "check the job is running (squeue -u $USER) and that its log "
                    "shows no OOM.")
            print("waiting for server...", flush=True)
            time.sleep(15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=ROOT / "youtube_dataset/videos")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/descriptions.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all. Use 2 for a smoke test.")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--base-url", default=None,
                    help="override; default reads outputs/endpoint.txt")
    ap.add_argument("--wait", type=float, default=900,
                    help="seconds to wait for the server to finish loading")
    args = ap.parse_args()

    base_url = args.base_url
    if not base_url:
        if not ENDPOINT_FILE.exists():
            raise SystemExit(f"no {ENDPOINT_FILE} -- is the serve job running?")
        base_url = ENDPOINT_FILE.read_text().strip()
    print(f"endpoint: {base_url}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    clips = sorted(args.videos.glob("*.mp4"))
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"no .mp4 under {args.videos}")

    # Resume, same as the offline script: the server's job has a walltime too.
    done = set()
    if args.out.exists():
        with args.out.open() as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    clips = [c for c in clips if c.stem not in done]
    print(f"{len(clips)} videos to process ({len(done)} already done)", flush=True)
    if not clips:
        return

    client = wait_for_server(base_url, args.wait)

    with args.out.open("a") as fout:
        for i, clip in enumerate(clips, 1):
            try:
                resp = client.chat.completions.create(
                    model="qwen3-vl",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "video_url",
                             "video_url": {"url": f"file://{clip.resolve()}"}},
                            {"type": "text", "text": PROMPT},
                        ],
                    }],
                    temperature=0.2,
                    max_tokens=args.max_tokens,
                    # Per-request sampling of the clip. If the server rejects
                    # this field, drop it and set --mm-processor-kwargs on the
                    # serve side instead -- support is vLLM-version dependent.
                    extra_body={"mm_processor_kwargs": {
                        "fps": args.fps,
                        "total_pixels": 20480 * 32 * 32,
                    }},
                    timeout=600,
                )
                desc = resp.choices[0].message.content.strip()
            except Exception as e:
                # One bad clip must not kill the run. Same rationale as 21_.
                print(f"[{i}/{len(clips)}] FAIL {clip.stem}: {e}", flush=True)
                continue

            fout.write(json.dumps({"id": clip.stem, "video": str(clip),
                                   "description": desc}) + "\n")
            fout.flush()
            print(f"[{i}/{len(clips)}] {clip.stem}: {desc[:80]}...", flush=True)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
