#!/usr/bin/env python
"""
Smoke test via plain transformers -- NO vLLM. Load Qwen3-VL-30B-A3B, say hi.

WHY THIS EXISTS: vLLM's PyPI wheels are compiled against CUDA 13, but the
BigPurple driver (575.57.08) only supports CUDA 12.9, so `import vllm` dies on
`libcudart.so.13: cannot open shared object file`. torch itself is fine -- the
cu128 build sees the A100 -- so transformers runs today while the vLLM wheel
situation gets sorted.

This is SLOW compared to vLLM (no paged attention, no continuous batching) and
is not what you want for the 11.6k-segment pipeline. It is here to prove the
weights + GPU + env are good, and to unblock experimenting with prompts.

    python scripts/18_smoke_hf.py
    python scripts/18_smoke_hf.py --prompt "what model are you?"
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # no network on compute nodes

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")
MODEL = ROOT / "models/Qwen3-VL-30B-A3B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--prompt", default="hi")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    if not (args.model / "config.json").exists():
        raise SystemExit(f"no model at {args.model}")

    if not torch.cuda.is_available():
        raise SystemExit(
            "no GPU visible. torch reports CUDA unavailable -- are you on a "
            "login/CPU node? This needs an A100 allocation."
        )
    print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"torch: {torch.__version__} (CUDA {torch.version.cuda})", flush=True)

    processor = AutoProcessor.from_pretrained(str(args.model))

    # bf16, not fp16: A100 is compute-capability 8.0 and has native bf16. The
    # 30B MoE is ~53GB of weights, which fits an 80GB card with room to spare.
    print("loading weights (~53GB off GPFS, this takes a few minutes)...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    # Text-only turn -- the VL model handles a bare text message fine.
    msgs = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    inputs = processor.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=False)

    # Strip the prompt: generate() returns prompt + completion.
    reply = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()

    print("\n=== PROMPT ===")
    print(args.prompt)
    print("\n=== REPLY ===")
    print(reply)
    print(f"\npeak GPU: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
