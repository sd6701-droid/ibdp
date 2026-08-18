#!/usr/bin/env bash
#SBATCH --job-name=ibdp-pose1
#SBATCH --partition=gl40s_long        # 24h: 64 videos on one node is a long haul
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1             # raise with GPUS=n (see below) for fan-out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/outputs/poses/logs/pose1-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/outputs/poses/logs/pose1-%j.out
#
# ViTPose over EVERY downloaded Chambers video inside ONE job on ONE node --
# the single-allocation alternative to the 64-task array in scripts/41.
#
# WHY YOU MIGHT WANT THIS over the array: it queues once instead of 64 times,
# holds its node for the whole sweep, and keeps one log. WHY YOU MIGHT NOT:
# the array finishes sooner, because 8 tasks run side by side while this walks
# the list with GPUS workers.
#
# SUBMIT:
#   mkdir -p /gpfs/scratch/sd6701/personal/ibdp/outputs/poses/logs
#   sbatch scripts/42_pose_chambers_one_node.sbatch                  # 1 GPU, sequential
#   sbatch --gres=gpu:l40s:4 --export=ALL,GPUS=4 \
#          scripts/42_pose_chambers_one_node.sbatch                  # 4 GPUs, 4 in parallel
#
# GPUS must MATCH the --gres count. gl40s nodes carry 8 cards, so GPUS=8 with
# --gres=gpu:l40s:8 takes a whole node -- fastest, but the longest queue wait.
#
# KNOBS:
#   SECONDS_PER_VIDEO=0   0 = whole video (default); 10 = first 10s smoke test
#   FORCE=1               re-pose videos that already have output
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="${ROOT:-/gpfs/scratch/sd6701/personal/ibdp}"
MM="${MM:-/gpfs/scratch/sd6701/personal/mmpose}"
VIDEOS="$ROOT/datasets/chambers_infant_youtube/videos"
OUTDIR="$ROOT/outputs/poses"
SECONDS_PER_VIDEO="${SECONDS_PER_VIDEO:-0}"
GPUS="${GPUS:-1}"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${POSE_ENV:-vitpose}"

cd "${SLURM_SUBMIT_DIR:-$ROOT}"

mapfile -t FILES < <(ls -1 "$VIDEOS"/*.mp4 2>/dev/null | sort)
N=${#FILES[@]}
[[ $N -eq 0 ]] && { echo "ERROR: no mp4 files in $VIDEOS" >&2; exit 1; }

DET_CFG="$MM/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
DET_W=$(ls "$MM"/checkpoints/*rtmdet*.pth 2>/dev/null | head -1)
POSE_CFG="$MM/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py"
for f in "$DET_CFG" "$DET_W" "$POSE_CFG"; do
  if [[ -z "$f" || ! -f "$f" ]]; then
    echo "ERROR: missing required file: '${f:-<no rtmdet .pth found>}'" >&2
    exit 1
  fi
done

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} videos=$N workers=$GPUS"
echo "gpus=${CUDA_VISIBLE_DEVICES:-?}  detector=$DET_W"
echo

# One worker per GPU, each striding through the list (worker w takes videos
# w, w+GPUS, w+2*GPUS, ...). A stride rather than a contiguous block so a slow
# stretch of long videos does not land entirely on one worker.
run_worker() {
  local worker=$1
  # Pin this worker to its own card. The script itself always asks for cuda:0,
  # so CUDA_VISIBLE_DEVICES is what actually separates the workers.
  export CUDA_VISIBLE_DEVICES=$worker
  local i video vid
  for (( i=worker; i<N; i+=GPUS )); do
    video="${FILES[$i]}"
    vid=$(basename "$video" .mp4)

    if [[ "${FORCE:-0}" != "1" ]] && compgen -G "$OUTDIR/$vid/pred/*.json" > /dev/null; then
      echo "[w$worker] $((i+1))/$N skip $vid (already posed)"
      continue
    fi

    echo "[w$worker] $((i+1))/$N pose $vid"
    python scripts/30_infant_pose.py \
      --video "$video" \
      --seconds "$SECONDS_PER_VIDEO" \
      --config "$POSE_CFG" \
      --det-config "$DET_CFG" \
      --det-weights "$DET_W" \
      --outdir "$OUTDIR" \
      --device cuda:0 \
      --draw-bbox \
      || echo "[w$worker] FAILED $vid (rc=$?) -- continuing" >&2
  done
  echo "[w$worker] done"
}

for (( w=0; w<GPUS; w++ )); do
  run_worker "$w" &
done
wait

echo
posed=$(ls -d "$OUTDIR"/*/ 2>/dev/null | wc -l)
echo "finished: $posed video dirs under $OUTDIR"
