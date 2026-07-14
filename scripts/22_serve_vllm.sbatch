#!/usr/bin/env bash
#SBATCH --job-name=ibdp-vl30b-serve
#SBATCH --partition=a100_short        # a100_long if you need >4h of uptime
#SBATCH --gres=gpu:a100:1             # ONE 80GB card is enough at TP=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/serve-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/serve-%j.out
#
# Same model, same memory math as 21_describe_videos.sbatch -- but as an
# OpenAI-compatible HTTP server instead of an in-process vLLM. The endpoint only
# exists for this job's walltime; when the job ends the server is gone.
#
# SUBMIT:
#   sbatch scripts/22_serve_vllm.sbatch
#   # then, from the login node, wait for the endpoint file to appear:
#   cat /gpfs/scratch/sd6701/personal/ibdp/outputs/endpoint.txt
#   python scripts/23_describe_via_api.py --limit 2
set -euo pipefail

ROOT=/gpfs/scratch/sd6701/personal/ibdp

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ibdp

export HF_HOME=$ROOT/.hf
export HF_HUB_OFFLINE=1               # no network here; fail fast, don't hang
export VLLM_WORKER_MULTIPROC_METHOD=spawn

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The client has no way to know which node SLURM gave us. Publish it.
# Written before the server is up; the client polls /health, not this file's
# existence, so a stale file from a dead job cannot silently mislead it.
mkdir -p "$ROOT/outputs"
echo "http://$(hostname -f):8000/v1" > "$ROOT/outputs/endpoint.txt"
echo "ENDPOINT: $(cat "$ROOT/outputs/endpoint.txt")"

# --allowed-local-media-path lets requests reference clips as file:///... URLs.
# Without it vLLM refuses local paths and you would have to base64 every mp4
# into the request body -- gigabytes of JSON for no reason, since the server can
# read the same GPFS mount the client can.
exec vllm serve "$ROOT/models/Qwen3-VL-30B-A3B-Instruct" \
  --served-model-name qwen3-vl \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt video=1 \
  --dtype bfloat16 \
  --allowed-local-media-path "$ROOT/youtube_dataset"
