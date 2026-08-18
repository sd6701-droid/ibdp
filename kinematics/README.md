# kinematics — pose estimates → kinematic features

Chambers et al. ("Computer vision to automatically assess infant neuromotor
risk") feature pipeline, copied from `chamber_et_al_src/` and adapted to run
directly on OUR ViTPose outputs (`outputs/poses/<video_id>/pred/*.json` from
`scripts/30_infant_pose.py`).

## Run

    python kinematics/run_pipeline.py --poses-root outputs/poses
    python kinematics/run_pipeline.py --poses-root outputs/poses --only 0LGLSB5BVi0
    # no source video next to pred/? supply the metadata yourself:
    python kinematics/run_pipeline.py --poses-root outputs/poses --fps 30 --pixel-x 360 --pixel-y 640

Needs: numpy<2, pandas>=2,<2.2, scipy, matplotlib (see requirements.txt).
ffmpeg (ffprobe) on PATH, or cv2, for reading fps/frame size — else pass
--fps/--pixel-x/--pixel-y.

## Pipeline (3 stages, resumable; --force reruns)

1. **convert** (`src/data/convert_vitpose_json.py`, ours)
   pred JSONs → `data/pose_estimates/<data_set>/py/pose_estimates.pkl`
   (long-form: video, frame, bp, x, y, part_idx, fps, pixel_x/y, time).
   Per frame, ONE person is kept: most keypoints ≥ 0.3 confidence, ties by
   mean score. Sub-threshold keypoints → NaN. Neck = shoulder midpoint
   (COCO-17 has none; the angle code needs it). Undetected frames are
   explicit NaN rows.

2. **preprocess** (`src/data/preprocess_pose_data.py`, Chambers)
   - normalise x,y by frame size (centered, aspect-ratio kept)
   - linear interpolation of missing frames
   - **rolling MEDIAN (0.5 s) then rolling MEAN (0.5 s)**, centered, per
     body part — the pose-noise smoothing
   - rotate + scale skeletons: trunk (shoulder-center↔hip-center) = 1 unit
   - joint angles (elbow/knee/hip/shoulder, L/R; Neck as shoulder parent)
   - velocity & acceleration for coords and angles, each smoothed AGAIN
     with a 0.25 s rolling mean (raw kept in *_raw columns)
   → `processed_pose_estimates_coords.pkl` / `..._angles.pkl`

3. **features** (`src/data/build_features.py`, Chambers)
   → `data/interim/features_<data_set>.pkl` + `.csv`, one row per video.

## Features (104 per video)

Positional, per body part in {LAnkle, RAnkle, LWrist, RWrist} (12 × 4 = 48):
  medianx/y, IQRx/y, medianvelx/y, IQRvelx/y, IQRaccx/y, meanent
  (units: trunk lengths, trunk lengths/s)

Angular, per joint in {L,R}×{Elbow, Knee, Hip, Shoulder} (6 × 8 = 48):
  mean_angle, stdev_angle (circular stats), entropy_angle,
  median_vel_angle, IQR_vel_angle, IQR_acc_angle (deg, deg/s)

Symmetry (8): lrCorr_x_<part> (L–R correlation of radial distance) and
  lrCorr_angle_<part> for part in {Ankle/Wrist-side parts, Elbow, Knee,
  Hip, Shoulder}.

## Not (yet) wired in, from the original repo

- `src/data/compute_surprise.py` — Chambers' Bayesian-surprise scoring of
  the processed pose data against their reference population (needs their
  reference-model pickles).
- `src/data/merge_data_sets.py` — merging youtube + clinical feature sets.
- `src/data/load_pose_data.py` — the ORIGINAL OpenPose-pkl loader that our
  convert stage replaces. Kept for reference, not called.
- `src/pose_model/` — their pose-model training/eval; superseded by our
  retrained ViTPose (scripts/30).
- `notebooks/` — their exploratory notebooks; paths assume their layout.

## Local changes vs the paper's code

- `convert_vitpose_json.py` (new): reads our bare-list pred JSON format.
- `util_data.py`: video libs (cv2/moviepy/skvideo) import lazily;
  `group_keys=False` on the 13 bp/video groupby-apply chains and
  `numeric_only=True` in get_joint_angles (pandas ≥2 compat).
- `build_features.py`: `.reset_index(drop=True)` on the xy-features
  groupby (same pandas ≥2 index/column collision).
None of these change any numbers — they restore the pandas-1 behaviour the
code was written against.
