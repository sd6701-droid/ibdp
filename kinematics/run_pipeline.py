#!/usr/bin/env python3
'''
End-to-end: ViTPose pred JSONs -> Chambers et al. kinematic features.

    python kinematics/run_pipeline.py --poses-root outputs/poses
    python kinematics/run_pipeline.py --poses-root outputs/poses --only 0LGLSB5BVi0
    python kinematics/run_pipeline.py --poses-root outputs/poses --data-set vitpose --force

Three stages, each skipped when its output already exists (--force reruns all):
  1. convert   pred/*.json -> data/pose_estimates/<data_set>/py/pose_estimates.pkl
  2. preprocess  interpolate -> rolling median+mean smooth -> trunk-normalise
                 -> joint angles -> velocity/acceleration (Chambers et al.,
                 untouched) -> processed_pose_estimates_{coords,angles}.pkl
  3. features    one row per video -> data/interim/features_<data_set>.pkl
                 (+ a .csv copy next to it)

The Chambers code addresses everything relative to src/data, so this runner
chdirs there; all --paths are resolved to absolute first.
'''
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--poses-root', type=Path,
                    default=Path('/gpfs/scratch/sd6701/personal/ibdp/outputs/poses'),
                    help='tree of <video_id>/pred/*.json from scripts/30')
    ap.add_argument('--data-set', default='vitpose',
                    help='name for this batch under data/pose_estimates/')
    ap.add_argument('--only', help='comma-separated video ids')
    ap.add_argument('--score-threshold', type=float, default=0.3)
    ap.add_argument('--fps', type=float, help='override when no source video')
    ap.add_argument('--pixel-x', type=float, help='frame width override')
    ap.add_argument('--pixel-y', type=float, help='frame height override')
    ap.add_argument('--force', action='store_true', help='rerun every stage')
    args = ap.parse_args()

    poses_root = args.poses_root.resolve()
    # cwd = kinematics/src, so the Chambers-style '../data/...' paths inside
    # the stage mains resolve to kinematics/data/. Imports use absolute paths
    # (the stages' own relative sys.path.insert calls then simply miss, which
    # is harmless).
    os.chdir(HERE / 'src')
    sys.path.insert(0, str(HERE / 'src' / 'data'))
    sys.path.insert(0, str(HERE / 'src' / 'modules'))

    est = Path('../data/pose_estimates') / args.data_set / 'py'
    interim = Path('../data/interim')
    interim.mkdir(parents=True, exist_ok=True)

    print('=== 1/3 convert', poses_root)
    if args.force or not (est / 'pose_estimates.pkl').exists():
        import convert_vitpose_json
        convert_vitpose_json.main(args.data_set, str(poses_root),
                                  score_threshold=args.score_threshold,
                                  fps=args.fps, pixel_x=args.pixel_x,
                                  pixel_y=args.pixel_y, only=args.only)
    else:
        print('exists, skipping (--force to redo)')

    print('=== 2/3 preprocess (interpolate, smooth, normalise, dynamics)')
    # BOTH outputs, not just coords: the stage writes coords then angles, and
    # stage 3 reads both. Checking only coords means a job killed between the
    # two writes skips this stage on resubmit and dies in stage 3 instead.
    stage2 = ['processed_pose_estimates_coords.pkl',
              'processed_pose_estimates_angles.pkl']
    if args.force or not all((est / f).exists() for f in stage2):
        import preprocess_pose_data
        preprocess_pose_data.main(args.data_set)
    else:
        print('exists, skipping (--force to redo)')

    print('=== 3/3 features')
    import build_features
    build_features.main(args.data_set)

    import pandas as pd
    pkl = interim / ('features_' + args.data_set + '.pkl')
    features = pd.read_pickle(pkl)
    csv = pkl.with_suffix('.csv')
    features.to_csv(csv, index=False)
    print('\nfeatures: ' + str(len(features)) + ' videos x '
          + str(features.shape[1] - 1) + ' features')
    print('  ' + str(pkl.resolve()))
    print('  ' + str(csv.resolve()))

    # Per-video copies NEXT TO pred/ and vis/, so each pose folder is
    # self-contained: <poses_root>/<video_id>/kinematics/ gets the feature row
    # and the per-frame time series, each as pkl + csv. The central copies
    # above stay the aggregate across videos.
    coords = pd.read_pickle(est / 'processed_pose_estimates_coords.pkl')
    angles = pd.read_pickle(est / 'processed_pose_estimates_angles.pkl')
    for vid, vrow in features.groupby('video'):
        vdir = poses_root / str(vid) / 'kinematics'
        if not vdir.parent.is_dir():
            continue        # video processed in an earlier run from another root
        vdir.mkdir(exist_ok=True)
        vrow.to_pickle(vdir / 'features.pkl')
        vrow.to_csv(vdir / 'features.csv', index=False)
        vc = coords[coords.video == vid]
        va = angles[angles.video == vid]
        vc.to_pickle(vdir / 'timeseries_coords.pkl')
        vc.to_csv(vdir / 'timeseries_coords.csv', index=False)
        va.to_pickle(vdir / 'timeseries_angles.pkl')
        va.to_csv(vdir / 'timeseries_angles.csv', index=False)
        print('  ' + str(vdir) + '  (features + timeseries, pkl + csv)')


if __name__ == '__main__':
    main()
