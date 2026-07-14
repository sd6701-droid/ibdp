#!/usr/bin/env bash
#SBATCH --job-name=ibdp-vl30b
#SBATCH --partition=a100_short        # a100_long if you need >4h
#SBATCH --gres=gpu:a100:1             # ONE 80GB card is enough at TP=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/vl30b-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/vl30b-%j.out
#
# Qwen3-VL-30B-A3B over the YouTube dataset, offline vLLM, single A100-80.
#
# PREREQ (login node, once): scripts/20_setup_serve.sh equivalent --
#   conda env ibdp-serve exists, and weights are at
#   $ROOT/models/Qwen3-VL-30B-A3B-Instruct. This node has no internet.
#
# SUBMIT:
#   mkdir -p /gpfs/scratch/sd6701/personal/ibdp/outputs/logs
#   sbatch scripts/21_describe_videos.sbatch
set -euo pipefail

ROOT=/gpfs/scratch/sd6701/personal/ibdp

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ibdp

# NOTE: vllm lives in this env alongside yt-dlp. It brings its own pinned torch.
# If you later `pip install -r requirements/batch.txt` here, pip must pick ONE
# torch for both vllm and torchcodec/torchvision -- and whichever loses breaks
# at import with an opaque `undefined symbol` error. If that day comes, split
# the batch deps into a separate env rather than fighting the resolver.

export HF_HOME=$ROOT/.hf
export HF_HUB_OFFLINE=1               # no network here; fail fast, don't hang
export FORCE_QWENVL_VIDEO_READER=torchcodec
export VLLM_WORKER_MULTIPROC_METHOD=spawn

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "${SLURM_SUBMIT_DIR:-$ROOT}"

# --limit counts VIDEOS, not segments: one 10-minute video is already ~61
# segments, which is plenty to check the timestamps and titles look right.
# Drop it once the smoke test looks good -- but note the full corpus is ~11.6k
# segments and will NOT finish in one 4h job. Resume is per segment; resubmit
# until it stops printing "segments to process", or move to a100_long.
srun python scripts/21_describe_videos.py --limit 1
