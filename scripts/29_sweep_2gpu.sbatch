#!/bin/bash
#SBATCH --job-name=ibdp-sweep2
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:a100:2            # 2 cards: ONLY the <=76GB models fit
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/sweep2-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/sweep2-%j.out
#
# The 2-GPU half of the sweep: every model that fits on 2x A100-80.
#
# The 72B (145GB) and 78B (156GB) are EXCLUDED on purpose -- on 160GB of VRAM
# they either OOM at load or, worse, offload to CPU and turn 15s segments into
# minutes. Run those on 4 cards with 29_sweep.sbatch:
#
#   MODELS="Qwen2.5-VL-72B-Instruct InternVL3-78B" \
#     sbatch --export=ALL --gres=gpu:a100:4 scripts/29_sweep.sbatch
#
# Both jobs can queue at the same time: each model writes its own files and
# resume keys on (model, segment), so they never collide.
#
# Qwen3-VL-30B-A3B (already fully annotated) stays in the list -- resume makes
# it exit in seconds without loading weights, and keeping it means this file
# still does the right thing on a fresh corpus.
#
# SUBMIT:  sbatch scripts/29_sweep_2gpu.sbatch
# WATCH:   tail -f /gpfs/scratch/sd6701/personal/ibdp/outputs/logs/sweep2-<jobid>.out
# RESUME:  resubmit the same file; finished segments are skipped.
#
set -uo pipefail          # NOT -e: 28 handles a failing model itself

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ibdp

cd "${SLURM_SUBMIT_DIR:-/gpfs/scratch/sd6701/personal/ibdp}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} partition=${SLURM_JOB_PARTITION:-?}"
echo "gpus=${CUDA_VISIBLE_DEVICES:-?}"

# NO srun -- see 29_sweep.sbatch.
MODELS="Qwen3-VL-30B-A3B-Instruct Qwen3-Omni-30B-A3B-Instruct Qwen3-VL-32B-Instruct InternVL3-38B" \
SCENES=1 MAX_SECONDS=0 scripts/28_run_all_models.sh all
