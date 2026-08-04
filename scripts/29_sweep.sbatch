#!/bin/bash
#SBATCH --job-name=ibdp-sweep
#SBATCH --partition=a100_long        # the multi-model sweep does not fit a100_short's 4h
#SBATCH --gres=gpu:a100:3            # 4 cards so the 72B/78B fit; 2 is enough if MODELS excludes them
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/sweep-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/sweep-%j.out
#
# Batch wrapper for 28_run_all_models.sh -- because the sweep died once already
# to a dropped SSH session at segment 65/250. A foreground run belongs to the
# connection; an sbatch job belongs to the scheduler and survives your laptop
# sleeping, the VPN dropping, and the login node rebooting.
#
# SUBMIT:
#   sbatch scripts/29_sweep.sbatch                       # scenes sweep, all videos
#   MODELS="Qwen3-Omni-30B-A3B-Instruct" sbatch --export=ALL scripts/29_sweep.sbatch
#   sbatch --gres=gpu:a100:2 scripts/29_sweep.sbatch     # small-models-only queue faster
#
# WATCH:
#   squeue -u $USER
#   tail -f /gpfs/scratch/sd6701/personal/ibdp/outputs/logs/sweep-<jobid>.out
#
# WALLTIME KILLS ARE CHEAP. Resume is per (model, segment): resubmit this same
# file and every finished segment is skipped; at most the one in flight is
# redone. The only thing a kill costs is that the resubmitted job starts a
# fresh W&B run (new RUN_TAG) -- the JSONL on disk is the continuous record.
#
set -uo pipefail          # NOT -e: 28 handles a failing model itself

# conda.sh, not `module load` -- the module's conda is a different install and
# `conda activate` does not work in a non-interactive batch shell without this.
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ibdp

cd "${SLURM_SUBMIT_DIR:-/gpfs/scratch/sd6701/personal/ibdp}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} partition=${SLURM_JOB_PARTITION:-?}"
echo "gpus=${CUDA_VISIBLE_DEVICES:-?} models=${MODELS:-<default list>}"

# NO srun. It killed the data_mover fetch job outright ("Unable to satisfy cpu
# bind request") and buys nothing for one task on one node -- 28 is a plain
# sequential shell loop that manages its own python processes.
#
# SCENES=1     annotate the shot-boundary splits from scripts/32, not a 10s grid
# MAX_SECONDS=0 whole videos -- this is the real run, not a shakedown
# "all"        every video that has splits
SCENES=1 MAX_SECONDS=0 scripts/28_run_all_models.sh all
