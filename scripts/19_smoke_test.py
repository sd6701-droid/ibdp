#!/usr/bin/env python
"""
Smoke test: load Qwen3-VL-30B-A3B and say hi. No video, no manifest, no batching.

The point is to isolate the setup from the pipeline. If this passes, then the
env, the weights on /gpfs, and the GPU are all good -- and any later failure is
in the video path, not the plumbing. If this fails, nothing downstream is worth
debugging yet.

Run inside the ibdp env, on a node that actually has a GPU:

    python scripts/19_smoke_test.py
    python scripts/19_smoke_test.py --prompt "what model are you?"

Expect a few quiet minutes on startup: vLLM reads ~53GB of weights off GPFS and
runs a memory-profiling pass before it emits a single token.
"""
import argparse
import os
from pathlib import Path

# Compute nodes have no network. Without this, a missing file turns into a hang
# on a HuggingFace call instead of an immediate, readable error.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from vllm import LLM, SamplingParams

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")
MODEL = ROOT / "models/Qwen3-VL-30B-A3B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--prompt", default="hi")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    if not (args.model / "config.json").exists():
        raise SystemExit(
            f"no model at {args.model}\n"
            "Weights are staged from a LOGIN node (compute nodes have no "
            "internet):\n"
            "  hf download Qwen/Qwen3-VL-30B-A3B-Instruct \\\n"
            f"    --local-dir {args.model}"
        )

    llm = LLM(
        model=str(args.model),
        # 8k, not the 32k the real pipeline uses: this is a smoke test and a
        # smaller KV cache means a faster start. It is not a tuning decision.
        max_model_len=8192,
        gpu_memory_utilization=0.90,
    )

    # .chat() applies the model's chat template for us -- no hand-built prompt
    # string, no special tokens to get subtly wrong.
    out = llm.chat(
        [{"role": "user", "content": args.prompt}],
        SamplingParams(temperature=0.7, max_tokens=args.max_tokens),
    )

    print("\n=== PROMPT ===")
    print(args.prompt)
    print("\n=== REPLY ===")
    print(out[0].outputs[0].text.strip())


if __name__ == "__main__":
    main()
