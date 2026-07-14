#!/usr/bin/env python
"""
Interactive chat with Qwen3-VL-30B-A3B. Loads the model ONCE, then loops.

The whole point: weights take ~65s to come off GPFS, generation takes seconds.
Re-running a script per question pays that 65s every time. This pays it once and
stays resident for as long as you hold the GPU allocation.

Multi-turn: history is kept, so follow-ups ("explain that more") work. The
context is capped -- see MAX_TURNS -- because history grows the prompt and a
30B on an 80GB card has finite KV room.

    python scripts/24_chat.py
    python scripts/24_chat.py --image /path/to/frame.png   # it IS a VL model
    python scripts/24_chat.py --video youtube_dataset/videos/<id>.mp4

In-chat commands:
    /reset   drop the conversation history, keep the model loaded
    /exit    quit
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")
MODEL = ROOT / "models/Qwen3-VL-30B-A3B-Instruct"

# Trim oldest turns past this. Each turn adds to the prompt, and an unbounded
# history will eventually OOM the KV cache mid-conversation -- a confusing way
# to lose a session you have been building context in for 20 minutes.
MAX_TURNS = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--image", type=Path, help="attach to the FIRST message")
    ap.add_argument("--video", type=Path, help="attach to the FIRST message")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU -- this needs an A100 allocation, not a login node.")

    processor = AutoProcessor.from_pretrained(str(args.model))
    print(f"loading {args.model.name} (~53GB, ~1min)...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model), dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    print(f"ready on {torch.cuda.get_device_name(0)}. "
          f"/reset to clear history, /exit to quit.\n", flush=True)

    history = []
    pending_media = []
    if args.image:
        pending_media.append({"type": "image", "image": str(args.image)})
    if args.video:
        pending_media.append({"type": "video", "video": str(args.video), "fps": 2.0})

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            history = []
            print("[history cleared]\n")
            continue

        # Media rides along with the first user turn only; after that it is in
        # the model's context and re-attaching would re-encode it every turn.
        content = pending_media + [{"type": "text", "text": user}]
        pending_media = []
        history.append({"role": "user", "content": content})

        if len(history) > MAX_TURNS * 2:
            history = history[-MAX_TURNS * 2:]

        inputs = processor.apply_chat_template(
            history, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
            )

        reply = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )[0].strip()

        print(f"\nqwen> {reply}\n", flush=True)
        history.append({"role": "assistant",
                        "content": [{"type": "text", "text": reply}]})


if __name__ == "__main__":
    main()
