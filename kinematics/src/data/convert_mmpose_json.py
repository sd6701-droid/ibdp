'''
Convert MMPose JSON output (COCO-17) to the pose_estimates.pkl format used by
preprocess_pose_data.py, replacing load_pose_data.main().

MMPose writes one JSON per video segment:
    <pose_dir>/<video>/results_seg_000.json
    {'meta_info': {'keypoint_id2name': {...}},
     'instance_info': [{'frame_id': i,
                        'instances': [{'keypoints': [[x, y] * 17],
                                       'keypoint_scores': [17],
                                       'bbox': [[x1, y1, x2, y2]],
                                       'bbox_score': float}, ...]}, ...]}

Differences from the OpenPose pkl path that this handles:
 - COCO-17 has no Neck; it is synthesised as the shoulder midpoint because
   get_joint_angles uses Neck as the vertex parent of both shoulder angles.
 - MMPose returns every keypoint every frame; low-score keypoints are set to
   NaN so that interpolate_df/smooth/get_joint_angles see them as undetected,
   as they would with thresholded OpenPose peaks.
 - The infant is selected per frame by number of confident keypoints
   (matching edit_df), tie-broken by mean keypoint score.
 - fps and frame size are read from the source video, since the JSON has neither.
'''

import sys
sys.path.insert(0, '../modules')
import os
import glob
import json
import numpy as np
import pandas as pd

# openpose part order, as expected by edit_df / get_joint_angles / normalise_skeletons
bps = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
       "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"]

coco_to_openpose = {
    'nose': 'Nose',
    'left_eye': 'LEye', 'right_eye': 'REye',
    'left_ear': 'LEar', 'right_ear': 'REar',
    'left_shoulder': 'LShoulder', 'right_shoulder': 'RShoulder',
    'left_elbow': 'LElbow', 'right_elbow': 'RElbow',
    'left_wrist': 'LWrist', 'right_wrist': 'RWrist',
    'left_hip': 'LHip', 'right_hip': 'RHip',
    'left_knee': 'LKnee', 'right_knee': 'RKnee',
    'left_ankle': 'LAnkle', 'right_ankle': 'RAnkle'}


def get_video_information_mmpose(video_path, video_names, fps=None, pixel_x=None, pixel_y=None):
    ''' Read fps and frame size from the source videos. fps/pixel_x/pixel_y
    override the video and are used directly when video_path is None. '''
    if video_path is None:
        assert fps and pixel_x and pixel_y, \
            'supply fps, pixel_x and pixel_y when there is no source video'
        return pd.DataFrame([[i, fps, pixel_x, pixel_y] for i in video_names],
                            columns=['video', 'fps', 'pixel_x', 'pixel_y'])
    import cv2
    rows = []
    for video in video_names:
        matches = [f for f in glob.glob(os.path.join(video_path, video + '.*'))
                   if f[-3:].lower() in ['mp4', 'avi', 'mov', 'mkv', 'wmv']]
        if len(matches) == 0:
            print('no source video found for ' + video + ', using supplied defaults')
            rows.append([video, fps, pixel_x, pixel_y])
            continue
        cap = cv2.VideoCapture(matches[0])
        ifps = cap.get(cv2.CAP_PROP_FPS)
        ipixel_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        ipixel_y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        if not ifps or np.isnan(ifps) or ifps <= 0:
            ifps = fps
        rows.append([video, ifps, ipixel_x, ipixel_y])
    df_fps = pd.DataFrame(rows, columns=['video', 'fps', 'pixel_x', 'pixel_y'])
    assert df_fps[['fps', 'pixel_x', 'pixel_y']].isnull().sum().sum() == 0, \
        'missing fps/frame size for: ' + str(list(df_fps.loc[df_fps.fps.isnull(), 'video']))
    return df_fps


def read_segments(video_dir):
    ''' Read every results_seg_*.json for one video in order, offsetting
    frame_id where a segment restarts its numbering. '''
    files = sorted(glob.glob(os.path.join(video_dir, 'results_seg_*.json')))
    if len(files) == 0:
        files = sorted(glob.glob(os.path.join(video_dir, '*.json')))
    frames = []
    id2name = {}
    offset = 0
    for file in files:
        with open(file) as f:
            data = json.load(f)
        if len(id2name) == 0:
            id2name = data['meta_info']['keypoint_id2name']
        seg = data['instance_info']
        seg_ids = [i['frame_id'] for i in seg]
        if len(frames) > 0 and min(seg_ids) <= max([i['frame_id'] for i in frames]):
            offset = max([i['frame_id'] for i in frames]) + 1 - min(seg_ids)
        for iframe in seg:
            frames.append({'frame_id': iframe['frame_id'] + offset,
                           'instances': iframe['instances']})
    return frames, id2name


def pick_instance(instances, score_threshold):
    ''' Select one person per frame: most keypoints above threshold, tie-broken
    by mean keypoint score. Mirrors edit_df, which keeps the person index with
    the most keypoints present. '''
    if len(instances) == 0:
        return None
    n_confident = [int(np.sum(np.array(i['keypoint_scores']) >= score_threshold))
                   for i in instances]
    mean_score = [float(np.mean(i['keypoint_scores'])) for i in instances]
    best = int(np.lexsort((mean_score, n_confident))[-1])
    if n_confident[best] == 0:
        return None
    return instances[best]


def instance_to_rows(instance, names, score_threshold):
    ''' One instance -> {bp: (x, y)}, with sub-threshold keypoints as NaN and
    Neck synthesised from the shoulders. '''
    kp = np.asarray(instance['keypoints'], dtype=float)
    sc = np.asarray(instance['keypoint_scores'], dtype=float)
    kp[sc < score_threshold, :] = np.nan

    coords = {}
    for i, name in enumerate(names):
        if name in coco_to_openpose:
            coords[coco_to_openpose[name]] = (kp[i, 0], kp[i, 1])

    # Neck is absent from COCO-17 but required by get_joint_angles
    ls = coords.get('LShoulder', (np.nan, np.nan))
    rs = coords.get('RShoulder', (np.nan, np.nan))
    coords['Neck'] = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    return coords


def main(data_set, mmpose_json_path, source_video_path, score_threshold=0.3,
         fps=None, pixel_x=None, pixel_y=None):

    pose_estimates_path = '../data/pose_estimates/' + data_set + '/py'
    if os.path.exists(pose_estimates_path) == 0:
        os.makedirs(pose_estimates_path)

    video_dirs = sorted([d for d in glob.glob(os.path.join(mmpose_json_path, '*'))
                         if os.path.isdir(d)])
    assert len(video_dirs) > 0, 'no per-video folders found in ' + mmpose_json_path

    rows = []
    for video_dir in video_dirs:
        video = os.path.basename(video_dir)
        frames, id2name = read_segments(video_dir)
        if len(frames) == 0:
            print('no frames for ' + video)
            continue
        names = [id2name[str(i)] if str(i) in id2name else id2name[i]
                 for i in sorted([int(k) for k in id2name.keys()])]
        print(video + ': ' + str(len(frames)) + ' frames')

        for iframe in frames:
            instance = pick_instance(iframe['instances'], score_threshold)
            if instance is None:
                continue
            coords = instance_to_rows(instance, names, score_threshold)
            for bp in bps:
                x, y = coords.get(bp, (np.nan, np.nan))
                rows.append([video, iframe['frame_id'], x, y, bp, bps.index(bp)])

    df = pd.DataFrame(rows, columns=['video', 'frame', 'x', 'y', 'bp', 'part_idx'])

    # dense grid of video x frame x bp, so undetected frames are explicit NaN rows
    max_frame = df.groupby('video').frame.max().reset_index()
    grid = []
    for _, irow in max_frame.iterrows():
        for iframe in np.arange(0, int(irow['frame']) + 1):
            for bp in bps:
                grid.append([irow['video'], iframe, bp])
    grid = pd.DataFrame(grid, columns=['video', 'frame', 'bp'])
    df = pd.merge(df.drop('part_idx', axis=1), grid, on=['video', 'frame', 'bp'], how='outer')
    df['part_idx'] = [bps.index(i) for i in df.bp]

    # fps and frame size
    df_fps = get_video_information_mmpose(source_video_path, df.video.unique(),
                                          fps, pixel_x, pixel_y)
    df_fps.to_pickle(os.path.join(pose_estimates_path, 'video_info.pkl'))
    df = pd.merge(df, df_fps, on='video', how='left')
    df['time'] = df['frame'] / df['fps']

    df = df[['video', 'frame', 'x', 'y', 'bp', 'part_idx', 'fps', 'pixel_x', 'pixel_y', 'time']]
    df.to_pickle(os.path.join(pose_estimates_path, 'pose_estimates.pkl'))
    print('wrote ' + os.path.join(pose_estimates_path, 'pose_estimates.pkl')
          + ' (' + str(len(df)) + ' rows, ' + str(df.video.nunique()) + ' videos)')
    return df


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
