'''
Convert OUR ViTPose/MMPose pred JSONs (outputs/poses/<video_id>/pred/*.json,
written by scripts/30_infant_pose.py) into the pose_estimates.pkl format that
preprocess_pose_data.py consumes.

Differences from convert_mmpose_json.py, which this builds on:
 - Our pred file is a BARE LIST of {'frame_id': i, 'instances': [...]} with no
   'meta_info' wrapper, so the keypoint order is assumed COCO-17 (which is
   what ViTPose emits through MMPoseInferencer).
 - One file per video under <poses_root>/<video_id>/pred/, not per-segment
   results_seg_*.json folders.
 - fps and frame size come from the video sitting next to pred/ (ffprobe,
   falling back to cv2), or from --fps/--pixel-x/--pixel-y overrides.

Everything downstream is untouched Chambers et al. code: person selection per
frame (most confident keypoints), sub-threshold keypoints -> NaN, Neck
synthesised from the shoulder midpoint, dense frame grid so undetected frames
are explicit NaN rows.
'''
import sys
sys.path.insert(0, '../modules')
import os
import glob
import json
import subprocess
import numpy as np
import pandas as pd

from convert_mmpose_json import (bps, pick_instance, instance_to_rows)

# COCO-17 order, used when the JSON carries no meta_info/keypoint_id2name.
COCO17 = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

VIDEO_EXTS = ('mp4', 'avi', 'mov', 'mkv', 'wmv')


def probe_video(path):
    ''' (fps, width, height) via ffprobe; cv2 as fallback; Nones on failure. '''
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=avg_frame_rate,width,height',
             '-of', 'json', str(path)],
            capture_output=True, text=True, timeout=30)
        s = json.loads(out.stdout)['streams'][0]
        num, _, den = s['avg_frame_rate'].partition('/')
        fps = float(num) / float(den or 1)
        return (fps if fps > 0 else None), float(s['width']), float(s['height'])
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        return (fps if fps and fps > 0 else None), w, h
    except Exception:
        return None, None, None


def find_video(video_dir):
    ''' The source clip scripts/30 leaves next to pred/ (any common ext). '''
    for f in sorted(glob.glob(os.path.join(video_dir, '*.*'))):
        if f.rsplit('.', 1)[-1].lower() in VIDEO_EXTS:
            return f
    return None


def read_frames(pred_dir):
    ''' All frames of one video, tolerant of both pred layouts:
    one whole-video JSON (bare list or {'instance_info': [...]}) or many
    per-frame JSONs. Returns (frames, keypoint_names). '''
    files = sorted(glob.glob(os.path.join(pred_dir, '*.json')))
    if len(files) == 0:
        return [], COCO17
    names = COCO17
    if len(files) == 1:
        with open(files[0]) as f:
            data = json.load(f)
        if isinstance(data, dict):
            id2name = data.get('meta_info', {}).get('keypoint_id2name')
            if id2name:
                names = [id2name[str(i)] for i in range(len(id2name))]
            data = data.get('instance_info', [])
        return data, names
    # per-frame files: order by the number in the filename
    def frame_no(path):
        digits = ''.join(c for c in os.path.basename(path) if c.isdigit())
        return int(digits) if digits else 0
    frames = []
    for i, file in enumerate(sorted(files, key=frame_no)):
        with open(file) as f:
            data = json.load(f)
        insts = data.get('instances', data) if isinstance(data, dict) else data
        frames.append({'frame_id': i, 'instances': insts})
    return frames, names


def main(data_set, poses_root, score_threshold=0.3,
         fps=None, pixel_x=None, pixel_y=None, only=None):

    pose_estimates_path = '../data/pose_estimates/' + data_set + '/py'
    if os.path.exists(pose_estimates_path) == 0:
        os.makedirs(pose_estimates_path)

    video_dirs = sorted([d for d in glob.glob(os.path.join(poses_root, '*'))
                         if os.path.isdir(os.path.join(d, 'pred'))])
    if only:
        wanted = set(only.split(',')) if isinstance(only, str) else set(only)
        video_dirs = [d for d in video_dirs if os.path.basename(d) in wanted]
    assert len(video_dirs) > 0, 'no <video_id>/pred folders under ' + poses_root

    rows, info_rows = [], []
    for video_dir in video_dirs:
        video = os.path.basename(video_dir)
        frames, names = read_frames(os.path.join(video_dir, 'pred'))
        if len(frames) == 0:
            print('no frames for ' + video)
            continue

        source = find_video(video_dir)
        ifps, ipx, ipy = probe_video(source) if source else (None, None, None)
        ifps, ipx, ipy = ifps or fps, ipx or pixel_x, ipy or pixel_y
        if not (ifps and ipx and ipy):
            # One bad video must not abort a 60-video batch: skip loudly.
            print('SKIP ' + video + ': no readable source video next to pred/ '
                  '(rerun it alone with --fps/--pixel-x/--pixel-y)')
            continue
        info_rows.append([video, ifps, ipx, ipy])
        print(video + ': ' + str(len(frames)) + ' frames, fps=' + str(round(ifps, 3))
              + ', ' + str(int(ipx)) + 'x' + str(int(ipy)))

        for iframe in frames:
            instance = pick_instance(iframe.get('instances', []), score_threshold)
            if instance is None:
                continue
            coords = instance_to_rows(instance, names, score_threshold)
            for bp in bps:
                x, y = coords.get(bp, (np.nan, np.nan))
                rows.append([video, iframe['frame_id'], x, y, bp, bps.index(bp)])

    df = pd.DataFrame(rows, columns=['video', 'frame', 'x', 'y', 'bp', 'part_idx'])

    # dense grid of video x frame x bp, so undetected frames are explicit NaNs
    grid = []
    for _, irow in df.groupby('video').frame.max().reset_index().iterrows():
        for iframe in np.arange(0, int(irow['frame']) + 1):
            for bp in bps:
                grid.append([irow['video'], iframe, bp])
    grid = pd.DataFrame(grid, columns=['video', 'frame', 'bp'])
    df = pd.merge(df.drop('part_idx', axis=1), grid,
                  on=['video', 'frame', 'bp'], how='outer')
    df['part_idx'] = [bps.index(i) for i in df.bp]

    df_fps = pd.DataFrame(info_rows, columns=['video', 'fps', 'pixel_x', 'pixel_y'])
    df_fps.to_pickle(os.path.join(pose_estimates_path, 'video_info.pkl'))
    df = pd.merge(df, df_fps, on='video', how='left')
    df['time'] = df['frame'] / df['fps']

    df = df[['video', 'frame', 'x', 'y', 'bp', 'part_idx',
             'fps', 'pixel_x', 'pixel_y', 'time']]
    df.to_pickle(os.path.join(pose_estimates_path, 'pose_estimates.pkl'))
    print('wrote ' + os.path.join(pose_estimates_path, 'pose_estimates.pkl')
          + ' (' + str(len(df)) + ' rows, ' + str(df.video.nunique()) + ' videos)')
    return df


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
