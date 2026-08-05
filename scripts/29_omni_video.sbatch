#!/bin/bash
#SBATCH --job-name=ibdp-omni
#SBATCH --partition=a100_short       # one video ~1h; fits well inside the 4h cap
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/omni-%j.out
#SBATCH --error=/gpfs/scratch/sd6701/personal/ibdp/outputs/logs/omni-%j.out
#
# Qwen3-Omni over ONE video's scene splits -- the quick way to see a prompt
# change (infant_actions, audio_source) on real output without queueing the
# whole sweep.
#
# The model itself is pinned to a single card by the backend (talker-less it
# fits one A100-80); the extra cards just widen the allocation so the job
# schedules on a node with headroom. gpu:a100:1 works too if the queue is long.
#
# SUBMIT:
#   sbatch scripts/29_omni_video.sbatch                     # default video
#   VIDEO=8yDn1uFbs4s sbatch --export=ALL scripts/29_omni_video.sbatch
#   sbatch scripts/29_omni_video.sbatch --limit 1           # extra args forward
#
# WATCH:
#   tail -f /gpfs/scratch/sd6701/personal/ibdp/outputs/logs/omni-<jobid>.out
#
# Resume is per (model, segment) under the CURRENT prompt_sha -- records from
# an older prompt are stale and re-annotated, which after a prompt change is
# exactly what you want.
#
set -uo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ibdp

ROOT="${ROOT:-/gpfs/scratch/sd6701/personal/ibdp}"
VIDEO="${VIDEO:-0HkcGRBsPUM}"

cd "${SLURM_SUBMIT_DIR:-$ROOT}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} video=$VIDEO"

# NO srun -- see 29_sweep.sbatch.
python scripts/26_describe_segments_hf.py \
    --model "$ROOT/models/Qwen3-Omni-30B-A3B-Instruct" \
    --scenes "$ROOT/outputs/scenes" \
    --only "$VIDEO" \
    --resume \
    --wandb --wandb-mode online \
    "$@"

# Print the fresh records so the result is IN THIS LOG, not another command away.
python - "$VIDEO" <<'PY'
import json, sys
from pathlib import Path
vid = sys.argv[1]
files = sorted(Path("/gpfs/scratch/sd6701/personal/ibdp/outputs")
               .glob("annotations_*Omni*.jsonl"))
if files:
    print(f"\n=== {files[-1].name} : {vid} ===")
    for line in files[-1].open():
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        if r.get("video_id") == vid and r.get("parse_ok"):
            print(f"[{r['timestamp']}] actions={r.get('infant_actions')} "
                  f"audio_src={r.get('audio_source')} "
                  f"events={r.get('audio_events')}\n"
                  f"  video: {r.get('description')}\n"
                  f"  audio: {r.get('audio_description')}")
PY
