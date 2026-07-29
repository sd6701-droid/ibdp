#!/usr/bin/env bash
#SBATCH --job-name=ibdp-seg
#SBATCH --partition=a100_short        # a100_long if you need >4h
#SBATCH --gres=gpu:a100:1             # OVERRIDE PER MODEL -- see table below
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/seg-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/seg-%j.out
#
# Segment-wise annotation, one model per job. Every argument is forwarded to
# 26_describe_segments_hf.py, so the model is chosen at submit time.
#
# ---------------------------------------------------------------------------
# GPUS PER MODEL. The #SBATCH line above is a default for the 30B; a
# --gres on the sbatch COMMAND LINE overrides it, which is why this file does
# not need one copy per model.
#
#   Qwen3-VL-30B-A3B-Instruct   ~61GB   --gres=gpu:a100:1
#   InternVL3-38B               ~76GB   --gres=gpu:a100:2
#   Qwen2.5-VL-72B-Instruct    ~145GB   --gres=gpu:a100:4
#   InternVL3-78B              ~156GB   --gres=gpu:a100:4
#
# Those are bf16 weights only. Video activations and KV cache come on top,
# which is why the 72B gets 4 cards (320GB) and not 2.
#
# SUBMIT -- three models over one video, three output files:
#   R=/gpfs/scratch/sd6701/personal/ibdp
#   V=0HkcGRBsPUM
#   sbatch --gres=gpu:a100:4 scripts/26_describe_segments.sbatch \
#          --model $R/models/Qwen2.5-VL-72B-Instruct --only $V
#   sbatch --gres=gpu:a100:4 scripts/26_describe_segments.sbatch \
#          --model $R/models/InternVL3-78B --only $V
#   sbatch --gres=gpu:a100:2 scripts/26_describe_segments.sbatch \
#          --model $R/models/InternVL3-38B --only $V
#
# They queue independently and run in whatever order the scheduler allows.
# Nothing is shared between them: separate output files, and --resume ignores
# other models' records.
#
# WALLTIME: resume is per (model, segment), so a job killed at the wall costs
# nothing. Resubmit the same line with --resume and it continues.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT=/gpfs/scratch/sd6701/personal/ibdp

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ibdp

export HF_HOME=$ROOT/.hf
export HF_HUB_OFFLINE=1               # no network on compute; fail fast, don't hang
export FORCE_QWENVL_VIDEO_READER=torchcodec

# sbatch exports the SUBMITTING shell's environment, and some login shells
# carry an LD_PRELOADed NVBLAS/UCX shim from an HPC module. Inside the job the
# shim is misconfigured (no nvblas.conf), fails every BLAS call
#   [NVBLAS] cublasXtSgemm failed with error=1
# and contributed to NaN audio features on the Omni run. Drop it: nothing in
# this pipeline wants a BLAS interposer.
unset LD_PRELOAD

# InternVL shards across cards via device_map="auto"; without this a fragmented
# 156GB load can trip allocator OOM even when the free VRAM total is sufficient.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "${SLURM_SUBMIT_DIR:-$ROOT}"
mkdir -p "$ROOT/outputs/logs"

echo "node=$(hostname) gpus=${SLURM_GPUS_ON_NODE:-?} job=${SLURM_JOB_ID:-?}"
echo "args: $*"

# NO srun: single-task job, and srun inherits SLURM_CPU_BIND from a parent
# job when this was submitted from inside an interactive allocation -- then
# dies on the new node with "CPU binding outside of job step allocation".
# (Same fix as 32_split_scenes.sbatch, where this actually happened.)
python scripts/26_describe_segments_hf.py "$@"
