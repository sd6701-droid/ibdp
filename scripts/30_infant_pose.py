#!/usr/bin/env python
"""
Infant 2D pose on the first N seconds of one YouTube-dataset video, using the
retrained ViTPose-large checkpoints from Zenodo 14833182 (Jahn et al. 2025),
run through MMPose.

    python scripts/30_infant_pose.py --id -EiqT5BpcpI
    python scripts/30_infant_pose.py --url "https://youtu.be/-EiqT5BpcpI" --seconds 10

Outputs, under outputs/poses/<video_id>/:
    vis/     frames (and a video) with the skeleton drawn on
    pred/    per-frame JSON: 17 COCO keypoints [x, y, score] per detected person

TWO THINGS TO KNOW:
1. This is TOP-DOWN pose: a person detector finds every human, then ViTPose
   poses each box. It does NOT know infant from adult -- every person in frame
   gets a skeleton. Filtering to just the infant is a separate step (the
   26_*.jsonl annotations already say which segments contain one).
2. The COCO skeleton is 17 keypoints: nose, eyes, ears, shoulders, elbows,
   wrists, hips, knees, ankles. The retrained weights improve localisation on
   infants; the keypoint set is unchanged.

ENV: a DEDICATED env (mmpose/mmcv pin their own torch -- do not reuse ibdp).
See scripts/README or the install block in the chat. Needs mmpose, mmcv, mmdet.

OFFLINE NOTE: MMPoseInferencer's default person detector downloads weights on
first use. Compute nodes have no internet, so run this ONCE on a login node
first (any --id) to warm the cache, or pre-stage the detector. After that it
runs offline on the A100.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")
VIDEOS = ROOT / "youtube_dataset/videos"
WEIGHTS = ROOT / "models/vitpose_infant/retrainedViTPose_split1.pth"

# ViTPose-large on COCO, top-down heatmap. This path is inside a cloned mmpose
# repo. Set --config if yours lives elsewhere.
DEFAULT_CONFIG = (
    "mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/"
    "td-hm_ViTPose-large_8xb64-210e_coco-256x192.py"
)

YT_ID = re.compile(r"(?:v=|/shorts/|youtu\.be/|/watch\?v=)?([0-9A-Za-z_-]{11})")


def resolve_id(s: str) -> str:
    """Accept a bare 11-char id, a watch URL, or a youtu.be URL -> id."""
    s = s.strip()
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", s):
        return s
    m = YT_ID.search(s)
    if not m:
        raise SystemExit(f"could not extract a video id from: {s!r}")
    return m.group(1)


def cut_clip(src: Path, seconds: float, dst: Path):
    """First `seconds` of src -> dst. Re-encode (not -c copy) so the cut is
    frame-accurate rather than snapping to the nearest keyframe."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-t", str(seconds), "-an", str(dst)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", help="youtube video id (11 chars)")
    g.add_argument("--url", help="youtube watch or youtu.be url")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--outdir", type=Path, default=ROOT / "outputs/poses")
    ap.add_argument("--device", default=None, help="cuda:0 | cpu (auto if unset)")
    ap.add_argument("--kpt-thr", type=float, default=0.3,
                    help="hide keypoints below this confidence")
    args = ap.parse_args()

    vid = resolve_id(args.id or args.url)
    mp4 = VIDEOS / f"{vid}.mp4"
    if not mp4.exists():
        raise SystemExit(f"no video for id {vid} at {mp4}\n"
                         f"(is it in the dataset? ls {VIDEOS})")
    if not args.weights.exists():
        raise SystemExit(f"missing weights: {args.weights}\n"
                         "download split1 from Zenodo 14833182 on a login node.")
    if not args.config.exists():
        raise SystemExit(f"missing config: {args.config}\n"
                         "clone mmpose so the ViTPose-large COCO config is present.")

    # Import here so --help works without the heavy env installed.
    try:
        import torch
        from mmpose.apis import MMPoseInferencer
    except ImportError as e:
        raise SystemExit(f"mmpose/torch not importable in this env: {e}\n"
                         "activate the dedicated pose env first.")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: no GPU -- ViTPose-large on CPU is slow "
              "(seconds per frame). Fine for a short demo clip.", flush=True)

    out = args.outdir / vid
    (out / "vis").mkdir(parents=True, exist_ok=True)
    (out / "pred").mkdir(parents=True, exist_ok=True)

    clip = out / f"{vid}_first{int(args.seconds)}s.mp4"
    print(f"cutting first {args.seconds:.0f}s -> {clip}", flush=True)
    cut_clip(mp4, args.seconds, clip)

    print(f"loading ViTPose (weights={args.weights.name}) on {device}...", flush=True)
    inferencer = MMPoseInferencer(
        pose2d=str(args.config),
        pose2d_weights=str(args.weights),
        device=device,
    )

    print("running pose over the clip...", flush=True)
    n = 0
    for _ in inferencer(
        str(clip),
        vis_out_dir=str(out / "vis"),
        pred_out_dir=str(out / "pred"),
        kpt_thr=args.kpt_thr,
        return_vis=False,
    ):
        n += 1
        if n % 20 == 0:
            print(f"  {n} frames", flush=True)

    print(f"\ndone: {n} frames", flush=True)
    print(f"  overlays : {out / 'vis'}")
    print(f"  keypoints: {out / 'pred'}")


if __name__ == "__main__":
    main()
