#!/usr/bin/env bash
#SBATCH --job-name=ibdp-pose
#SBATCH --partition=gl40s_short       # gl40s_long if a single video needs >4h
#SBATCH --gres=gpu:l40s:1             # ViTPose-large is ~300M params. ONE card.
#SBATCH --cpus-per-task=4             # ffmpeg decode + dataloader
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-63%8                # one task per video, 8 at a time
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/outputs/poses/logs/pose-%A_%a.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/outputs/poses/logs/pose-%A_%a.out
#
# ViTPose over every downloaded Chambers video, one video per array task.
#
# A JOB ARRAY, not one long job: 30_infant_pose.py is single-GPU by design and
# each video is decode-bound, so throughput comes from running videos side by
# side. 64 one-card tasks also queue far faster than one multi-card job.
#
# SUBMIT:
#   mkdir -p /gpfs/scratch/sd6701/personal/ibdp/outputs/poses/logs
#   sbatch scripts/41_pose_chambers_all.sbatch
#   squeue -u $USER
#
# The array range MUST match the video count. Check it first and pass a
# matching --array if it has changed since this was written:
#   ls /gpfs/scratch/sd6701/personal/ibdp/datasets/chambers_infant_youtube/videos/*.mp4 | wc -l
#   sbatch --array=0-$(( $(ls .../videos/*.mp4 | wc -l) - 1 ))%8 scripts/41_pose_chambers_all.sbatch
#
# KNOBS (environment, at submit time):
#   SECONDS_PER_VIDEO=0   0 = whole video (default); 10 = first 10s smoke test
#   FORCE=1               re-pose videos that already have output
#
#   sbatch --export=ALL,SECONDS_PER_VIDEO=10 scripts/41_pose_chambers_all.sbatch
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="${ROOT:-/gpfs/scratch/sd6701/personal/ibdp}"
MM="${MM:-/gpfs/scratch/sd6701/personal/mmpose}"
VIDEOS="$ROOT/datasets/chambers_infant_youtube/videos"
OUTDIR="$ROOT/outputs/poses"
SECONDS_PER_VIDEO="${SECONDS_PER_VIDEO:-0}"     # 0 = whole video

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
# The DEDICATED pose env -- mmpose/mmcv pin their own torch and must not share
# the ibdp env (see the ENV note in scripts/30_infant_pose.py).
conda activate "${POSE_ENV:-vitpose}"

# ---- pick this task's video ------------------------------------------------
# Sorted listing indexed by array id, so task N always maps to the same file.
mapfile -t FILES < <(ls -1 "$VIDEOS"/*.mp4 2>/dev/null | sort)
N=${#FILES[@]}
if [[ $N -eq 0 ]]; then
  echo "ERROR: no mp4 files in $VIDEOS" >&2
  exit 1
fi
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ $IDX -ge $N ]]; then
  # Array range wider than the file list: exit clean rather than fail loudly.
  echo "task $IDX >= $N videos -- nothing to do"
  exit 0
fi
VIDEO="${FILES[$IDX]}"
VID=$(basename "$VIDEO" .mp4)

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} task=$IDX/$((N-1))"
echo "video=$VIDEO"
echo "gpus=${CUDA_VISIBLE_DEVICES:-?}"

# ---- resume ----------------------------------------------------------------
# pred/ holding JSON means this video is done. Cheap to check, and it makes a
# resubmit after a walltime kill skip everything that already finished.
if [[ "${FORCE:-0}" != "1" ]] && compgen -G "$OUTDIR/$VID/pred/*.json" > /dev/null; then
  echo "already posed -> $OUTDIR/$VID/pred (FORCE=1 to redo)"
  exit 0
fi

# ---- resolve the detector --------------------------------------------------
# Top-down pose cannot run without a person detector, and compute nodes cannot
# download one -- so the checkpoint must already be on disk. Resolve by glob so
# a version bump in the filename does not break this script.
DET_CFG="$MM/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
DET_W=$(ls "$MM"/checkpoints/*rtmdet*.pth 2>/dev/null | head -1)
POSE_CFG="$MM/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py"

for f in "$DET_CFG" "$DET_W" "$POSE_CFG"; do
  if [[ -z "$f" || ! -f "$f" ]]; then
    echo "ERROR: missing required file: '${f:-<no rtmdet .pth found>}'" >&2
    echo "       checked MM=$MM -- set MM=/path/to/mmpose if the clone moved." >&2
    exit 1
  fi
done
echo "detector=$DET_W"

python scripts/30_infant_pose.py \
  --video "$VIDEO" \
  --seconds "$SECONDS_PER_VIDEO" \
  --config "$POSE_CFG" \
  --det-config "$DET_CFG" \
  --det-weights "$DET_W" \
  --outdir "$OUTDIR" \
  --draw-bbox
rc=$?

echo "rc=$rc  ->  $OUTDIR/$VID"
exit $rc
