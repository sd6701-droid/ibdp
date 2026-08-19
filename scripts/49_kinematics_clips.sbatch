#!/usr/bin/env bash
#SBATCH --job-name=ibdp-kin
#SBATCH --partition=cpu_medium        # CPU-only: no stage of this pipeline touches a GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/outputs/poses_chambers_clips/logs/kin-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/outputs/poses_chambers_clips/logs/kin-%j.out
#
# ViTPose pred JSONs -> Chambers kinematic features, as a BATCH job.
#
# WHY THIS EXISTS rather than an interactive srun: the pipeline is one long
# serial pandas job with no checkpointing inside a stage. Run under `srun
# --pty`, it dies with the allocation -- which is exactly what killed job
# 26583650 at its walltime, after ~all the work was done but before anything
# said so. A batch job survives your ssh session and writes a log either way.
#
# SUBMIT:
#   mkdir -p /gpfs/scratch/sd6701/personal/ibdp/outputs/poses_chambers_clips/logs
#   sbatch scripts/49_kinematics_clips.sbatch                    # the clip windows
#   sbatch --export=ALL,POSES=outputs/poses_chambers,DATASET=vitpose \
#          scripts/49_kinematics_clips.sbatch                    # whole videos
#
# KNOBS:
#   POSES=<dir>     pose tree to read      (default outputs/poses_chambers_clips)
#   DATASET=<name>  batch name under data/ (default vitpose_clips)
#   FORCE=1         redo stages 1 and 2 instead of reusing their outputs
#
# RESUMABLE: stages 1 (convert) and 2 (preprocess) are skipped when their .pkl
# already exists, so a resubmit after a timeout picks up where it stopped
# rather than redoing the expensive smoothing. FORCE=1 to override.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="${ROOT:-/gpfs/scratch/sd6701/personal/ibdp}"
POSES="${POSES:-outputs/poses_chambers_clips}"
DATASET="${DATASET:-vitpose_clips}"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${KIN_ENV:-ibdp}"

cd "$ROOT"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "poses=$POSES  data-set=$DATASET  force=${FORCE:-0}"
echo

ARGS=(--poses-root "$POSES" --data-set "$DATASET")
[[ "${FORCE:-0}" == "1" ]] && ARGS+=(--force)

python kinematics/run_pipeline.py "${ARGS[@]}"
rc=$?

echo
if [[ $rc -ne 0 ]]; then
  echo "run_pipeline FAILED (rc=$rc)" >&2
  exit $rc
fi

# Prove the artefact is loadable, not merely present: a job killed mid-write
# leaves a truncated pickle that `ls` reports as a perfectly good file.
python - "$DATASET" <<'PY'
import sys
import pandas as pd
from pathlib import Path
pkl = Path('kinematics/data/interim/features_' + sys.argv[1] + '.pkl')
df = pd.read_pickle(pkl)
print('features: ' + str(len(df)) + ' rows x ' + str(df.shape[1] - 1)
      + ' features -> ' + str(pkl))
print('videos: ' + str(df.video.nunique()))
PY
