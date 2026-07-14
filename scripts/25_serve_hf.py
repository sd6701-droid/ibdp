#!/usr/bin/env python
"""
OpenAI-compatible HTTP server for Qwen3-VL-30B-A3B, on plain transformers.

Binds 127.0.0.1:8000 on whatever GPU node it runs on. Anything on THAT SAME NODE
can then hit http://localhost:8000/v1 like any normal OpenAI endpoint.

  weights on disk:  /gpfs/scratch/sd6701/personal/ibdp/models/Qwen3-VL-30B-A3B-Instruct
  served at:        http://127.0.0.1:8000/v1   (on the node running this process)

To use it, you need a SECOND shell on the same node -- the server holds the
first one. Attach to your existing allocation:

    srun --jobid <your-jobid> --pty bash
    curl http://localhost:8000/health

WHY NOT vllm serve: vLLM's wheels are compiled against CUDA 13; the BigPurple
driver (575.57.08) tops out at CUDA 12.9, so `import vllm` dies on
libcudart.so.13. When a CUDA 12 wheel lands, delete this file and use
`vllm serve` -- it does paged attention and continuous batching, neither of
which this does.

WHAT THIS IS NOT: it serves ONE request at a time, under a lock. Concurrent
clients queue rather than share the GPU. Fine for interactive chat and prompt
iteration; do NOT run the 11.6k-segment video pipeline through it.

    pip install fastapi uvicorn        # pure python, no CUDA compile
    python scripts/25_serve_hf.py
"""
import argparse
import os
import threading
import time
import uuid
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")
MODEL = ROOT / "models/Qwen3-VL-30B-A3B-Instruct"
SERVED_NAME = "qwen3-vl-30b-a3b"

app = FastAPI(title="ibdp qwen3-vl")
STATE = {}

# The GPU is one resource and generate() is not reentrant. Without this lock,
# two concurrent requests interleave and corrupt each other's KV cache.
GPU_LOCK = threading.Lock()


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list
    max_tokens: int = 512
    temperature: float = 0.7


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": SERVED_NAME,
        "weights": str(MODEL),
        "node": os.environ.get("SLURMD_NODENAME", "unknown"),
        "gpu": torch.cuda.get_device_name(0),
    }


@app.get("/v1/models")
def models():
    return {"object": "list",
            "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "ibdp"}]}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(400, "messages is empty")

    processor, model = STATE["processor"], STATE["model"]

    # OpenAI permits `content` as a bare string; the VL chat template wants the
    # list-of-parts form. Normalise so both client styles work.
    msgs = []
    for m in req.messages:
        content = m.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        msgs.append({"role": m["role"], "content": content})

    with GPU_LOCK:
        inputs = processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=req.temperature > 0,
                temperature=req.temperature if req.temperature > 0 else None,
            )

        n_prompt = inputs["input_ids"].shape[1]
        reply = processor.batch_decode(
            out[:, n_prompt:], skip_special_tokens=True)[0].strip()
        n_completion = out.shape[1] - n_prompt

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": SERVED_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": int(n_prompt),
            "completion_tokens": int(n_completion),
            "total_tokens": int(n_prompt + n_completion),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODEL)
    # 127.0.0.1 = reachable only from this node. That is the whole point: the
    # weights and any PHI stay on the node, nothing is exposed to the network.
    # Use --host 0.0.0.0 ONLY if you intend to SSH-tunnel in from off-node.
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU -- needs an A100 allocation, not a login node.")

    print(f"loading {args.model.name} (~53GB, ~1min)...", flush=True)
    STATE["processor"] = AutoProcessor.from_pretrained(str(args.model))
    STATE["model"] = AutoModelForImageTextToText.from_pretrained(
        str(args.model), dtype=torch.bfloat16, device_map="cuda:0")
    STATE["model"].eval()

    node = os.environ.get("SLURMD_NODENAME", "this node")
    job = os.environ.get("SLURM_JOB_ID", "<jobid>")
    print(f"\nready on {torch.cuda.get_device_name(0)} ({node})", flush=True)
    print(f"endpoint: http://{args.host}:{args.port}/v1", flush=True)
    print(f"weights:  {args.model}", flush=True)
    print(f"\nfrom a second shell ON {node}:\n"
          f"  srun --jobid {job} --pty bash\n"
          f"  curl http://localhost:{args.port}/health\n", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
