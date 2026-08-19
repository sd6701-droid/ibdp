#!/usr/bin/env python3
"""
Pick the INFANT out of ViTPose's multi-person output, so the normative database
is built on infants and not on the adult holding them.

    # once: learn what an infant's body proportions look like
    python scripts/48_select_infant_track.py calibrate

    # then: per video, link tracks, score them, keep the infant
    python scripts/48_select_infant_track.py select \
        --poses-root outputs/poses --out-root outputs/poses_infant

THE PROBLEM
-----------
scripts/30 runs TOP-DOWN pose: a person detector finds every human, ViTPose
poses each box. Nothing in that pipeline knows infant from adult. On
kI9n00reJwM, 24% of frames carry 2-4 people.

The existing selector (`pick_instance` in convert_mmpose_json.py, inherited
from Chambers) takes, PER FRAME AND INDEPENDENTLY, the person with the most
above-threshold keypoints. That is wrong here in two compounding ways:

  1. It is biased TOWARD THE ADULT. Adults are larger, fully visible, upright,
     and squarely in the COCO training distribution, so they land more
     confident keypoints than a partially-occluded supine infant. The
     "most keypoints" rule hands the adult the frame.

  2. It has NO TEMPORAL LINKING. The choice is re-made every frame, so it can
     flip adult->infant->adult. Those flips are teleports of hundreds of
     pixels, and the feature set is built on velocity, acceleration and IQR --
     precisely the quantities a flip destroys. A single flip can dominate a
     video's IQRvel.

THE FIX, IN THREE STAGES
------------------------
  A. TRACK. Link per-frame detections into tracks by bbox IoU, so a person is
     one object over time, not a fresh guess each frame.

  B. MEASURE. Score each track on SCALE-FREE body proportions. We have no
     metric scale from monocular video, so absolute size is useless (a distant
     adult and a near infant have the same pixel height). Ratios are not:
     newborn head height is ~1/4 of body length against ~1/8 in adults, and
     infant limbs are proportionally much shorter. Every ratio below is
     normalised by torso length and is therefore invariant to camera distance.

  C. SELECT. Score each track against the INFANT distribution and keep the best
     match. The distribution is not guessed -- it is measured from
     Chamber-etal-dataset/pose_estimates_youtube_dataset.csv, which is 85
     videos of vetted, infant-only pose in the same 18-bodypart naming. That
     file is ground truth for "what an infant's proportions look like" on
     exactly this kind of footage.

WHAT THIS DOES NOT DO
---------------------
It cannot separate TWO INFANTS (twins, an older sibling near the same
proportions). Those are resolved by track persistence and centrality, which is
a weak signal -- so `select` always writes a QA contact sheet and a report
sorted by decision margin. Review the low-margin videos by eye before the
output reaches the normative database. An adult silently entering the
reference set is worse than dropping the video.

OUTPUT
------
  <out-root>/<video>/pred/<name>.json   one instance per frame (the infant),
                                        same schema as the input, so
                                        convert_vitpose_json.py consumes it
                                        with no changes
  <out-root>/<video>/<video>.mp4        symlink to the source clip, so that
                                        converter's find_video() still resolves
                                        fps and frame size
  <out-root>/report.csv                 per-video decision + margin + ratios
  <out-root>/qa/<video>.jpg             contact sheet of the kept skeleton
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# COCO-17, the order MMPoseInferencer emits. Indices are used directly below,
# so this must not be reordered.
# --------------------------------------------------------------------------
COCO17 = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
K = {n: i for i, n in enumerate(COCO17)}

# Chambers' 18-part naming, for reading the calibration CSV.
CH = {'nose': 'Nose', 'left_eye': 'LEye', 'right_eye': 'REye',
      'left_ear': 'LEar', 'right_ear': 'REar',
      'left_shoulder': 'LShoulder', 'right_shoulder': 'RShoulder',
      'left_elbow': 'LElbow', 'right_elbow': 'RElbow',
      'left_wrist': 'LWrist', 'right_wrist': 'RWrist',
      'left_hip': 'LHip', 'right_hip': 'RHip',
      'left_knee': 'LKnee', 'right_knee': 'RKnee',
      'left_ankle': 'LAnkle', 'right_ankle': 'RAnkle'}

# The three ratios that actually carry infant-vs-adult signal. Each is a length
# divided by torso length, so all are dimensionless and camera-distance free.
RATIOS = ('r_head', 'r_arm', 'r_leg')

DEFAULT_PRIOR = Path('outputs/infant_ratio_prior.json')


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def _d(p, q):
    """Euclidean distance, NaN if either endpoint is missing."""
    if p is None or q is None:
        return np.nan
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def ratios_from_points(pt):
    """pt: {coco_name: (x, y) or None} -> {'r_head','r_arm','r_leg'} (NaN where
    the needed keypoints are missing).

    torso = neck (shoulder midpoint) -> hip midpoint. Chosen as the denominator
    because both endpoints are among the most reliably detected keypoints and
    the segment is rigid: it does not change length as the subject moves, which
    a limb-based denominator would.
    """
    def mid(a, b):
        pa, pb = pt.get(a), pt.get(b)
        if pa is None or pb is None:
            return None
        return ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)

    neck = mid('left_shoulder', 'right_shoulder')
    hip = mid('left_hip', 'right_hip')
    torso = _d(neck, hip)
    if not (torso and torso > 1e-6 and np.isfinite(torso)):
        return {r: np.nan for r in RATIOS}

    # HEAD. Ear span is the primary measure -- it is a true head WIDTH and is
    # stable under head rotation in-plane. Eye span is the fallback (ears are
    # often occluded by a lying head against a mat); 1.9 converts eye span to
    # ear span, the ratio of the two medians in the calibration set.
    head = _d(pt.get('left_ear'), pt.get('right_ear'))
    if not np.isfinite(head):
        head = _d(pt.get('left_eye'), pt.get('right_eye')) * 1.9

    # LIMBS. Sum the two segments so a bent elbow/knee does not shrink the
    # measurement the way a wrist-to-shoulder straight line would. Average the
    # sides that are available rather than requiring both.
    def limb(prox, mid_, dist):
        out = []
        for side in ('left', 'right'):
            v = (_d(pt.get(side + '_' + prox), pt.get(side + '_' + mid_))
                 + _d(pt.get(side + '_' + mid_), pt.get(side + '_' + dist)))
            if np.isfinite(v):
                out.append(v)
        return float(np.mean(out)) if out else np.nan

    return {
        'r_head': head / torso,
        'r_arm': limb('shoulder', 'elbow', 'wrist') / torso,
        'r_leg': limb('hip', 'knee', 'ankle') / torso,
    }


def points_from_instance(inst, thr):
    """One MMPose instance -> {coco_name: (x, y)}, sub-threshold keypoints
    dropped (not zeroed -- a zeroed keypoint is a keypoint at the image
    corner, which would poison every distance it appears in)."""
    kp = np.asarray(inst['keypoints'], dtype=float)
    sc = np.asarray(inst['keypoint_scores'], dtype=float)
    return {n: (kp[i][0], kp[i][1]) for i, n in enumerate(COCO17)
            if i < len(sc) and sc[i] >= thr}


# --------------------------------------------------------------------------
# stage A -- tracking
# --------------------------------------------------------------------------
def _bbox(inst):
    """MMPose writes bbox as [[x1,y1,x2,y2]]; tolerate the unwrapped form."""
    b = inst.get('bbox')
    if b is None:
        return None
    if isinstance(b[0], (list, tuple)):
        b = b[0]
    return [float(v) for v in b[:4]]


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _weak_match(a, b):
    """Fallback association for when IoU is 0 but it is plainly the same person:
    boxes of similar size whose centres are within half a box-diagonal.

    Needed because IoU alone shatters tracks on handheld footage -- a fast pan
    moves the box clear of its own previous position in one frame, IoU drops to
    0, and one infant becomes hundreds of fragments. Scored below any real IoU
    match so it only ever fires when nothing better is on offer.
    """
    if a is None or b is None:
        return 0.0
    wa, ha = a[2] - a[0], a[3] - a[1]
    wb, hb = b[2] - b[0], b[3] - b[1]
    if min(wa, ha, wb, hb) <= 0:
        return 0.0
    scale = max(wa * ha, wb * hb) / max(1e-6, min(wa * ha, wb * hb))
    if scale > 4.0:                       # a very different-sized person
        return 0.0
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    diag = math.hypot((wa + wb) / 2, (ha + hb) / 2)
    return 0.15 if math.hypot(ca[0] - cb[0], ca[1] - cb[1]) < 0.5 * diag else 0.0


def link_tracks(frames, iou_thr=0.3, max_gap=15):
    """Greedy IoU tracker. Returns [{'id', 'frames': [(frame_idx, instance)]}].

    Greedy rather than Hungarian on purpose: at 1-4 people per frame the
    assignment is nearly always unambiguous, and greedy-by-descending-IoU
    gives the same answer without the dependency. max_gap keeps a track alive
    across brief detector dropouts (a hand passing over the infant), which
    otherwise shatter one infant into a dozen short tracks that each look too
    sparse to trust.
    """
    tracks, active = [], []   # active: (track, last_frame_idx, last_bbox)
    for fi, fr in enumerate(frames):
        insts = fr.get('instances', []) or []
        boxes = [_bbox(i) for i in insts]

        pairs = []
        for ai, (_, last_f, last_b) in enumerate(active):
            if fi - last_f > max_gap:
                continue
            for di, b in enumerate(boxes):
                v = _iou(last_b, b)
                if v < iou_thr:
                    v = _weak_match(last_b, b)
                if v > 0:
                    pairs.append((v, ai, di))
        pairs.sort(reverse=True)

        used_a, used_d = set(), set()
        for v, ai, di in pairs:
            if ai in used_a or di in used_d:
                continue
            used_a.add(ai)
            used_d.add(di)
            trk, _, _ = active[ai]
            trk['frames'].append((fi, insts[di]))
            active[ai] = (trk, fi, boxes[di])

        for di, inst in enumerate(insts):
            if di in used_d:
                continue
            trk = {'id': len(tracks), 'frames': [(fi, inst)]}
            tracks.append(trk)
            active.append((trk, fi, boxes[di]))

        active = [a for a in active if fi - a[1] <= max_gap]
    return tracks


# --------------------------------------------------------------------------
# stage B -- per-track proportions
# --------------------------------------------------------------------------
def track_profile(track, n_frames, thr):
    """Median ratios over the track, plus the bookkeeping the report needs.

    MEDIAN, not mean: a few frames will have a badly-placed keypoint, and one
    torso length collapsing toward zero sends a mean ratio to infinity. The
    median simply ignores it.
    """
    vals = {r: [] for r in RATIOS}
    conf = []
    for _, inst in track['frames']:
        pt = points_from_instance(inst, thr)
        rr = ratios_from_points(pt)
        for r in RATIOS:
            if np.isfinite(rr[r]):
                vals[r].append(rr[r])
        sc = np.asarray(inst['keypoint_scores'], dtype=float)
        conf.append(float(np.mean(sc)))

    prof = {r: (float(np.median(vals[r])) if vals[r] else np.nan)
            for r in RATIOS}
    prof['n_frames'] = len(track['frames'])
    prof['presence'] = len(track['frames']) / max(1, n_frames)
    prof['mean_conf'] = float(np.mean(conf)) if conf else 0.0
    prof['track_id'] = track['id']
    return prof


# --------------------------------------------------------------------------
# stage C -- scoring against the infant prior
# --------------------------------------------------------------------------
def infant_cost(prof, prior):
    """Mean squared z-distance from the infant distribution, over whichever
    ratios this track actually has. Lower is more infant-like.

    Averaging (not summing) over available ratios matters: a track with only
    r_head measurable must not beat a track with all three simply by having
    fewer terms in its sum.
    """
    zs = []
    for r in RATIOS:
        v = prof.get(r)
        if v is None or not np.isfinite(v):
            continue
        mu, sd = prior[r]['mean'], prior[r]['sd']
        if sd <= 0:
            continue
        zs.append(((v - mu) / sd) ** 2)
    if not zs:
        return np.inf, 0
    return float(np.mean(zs)), len(zs)


def select_track(profiles, prior, min_presence, min_ratios, max_cost):
    """-> (winner, cost, margin). margin is the runner-up's cost gap in units of
    the winner's cost.

    TWO INDEPENDENT CHECKS, and both matter:

      cost   -- ABSOLUTE. How infant-like the winner is on its own terms. A
                video where only one track clears the gates has no runner-up,
                so a relative test would wave through an obvious adult
                unopposed. max_cost is what stops that.

      margin -- RELATIVE. How much better the winner was than the next
                candidate. Low margin means two people in frame scored alike
                (an adult and an older child, or twins) and only a human can
                say which is the subject.
    """
    cands = []
    for p in profiles:
        if p['presence'] < min_presence:
            continue
        cost, n = infant_cost(p, prior)
        if n < min_ratios or not np.isfinite(cost):
            continue
        cands.append((cost, p))
    if not cands:
        return None, np.inf, 0.0
    cands.sort(key=lambda c: c[0])
    win_cost, win = cands[0]
    if win_cost > max_cost:
        # Best candidate is still not plausibly an infant. Dropping the video
        # is the right call: a normative database poisoned with adult
        # kinematics is worse than one built on fewer infants.
        return None, win_cost, 0.0
    margin = (float('inf') if len(cands) == 1
              else (cands[1][0] - win_cost) / max(win_cost, 1e-6))
    return win, win_cost, float(margin)


# --------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------
def cmd_calibrate(args):
    """Measure the infant ratio distribution from Chambers' vetted infant-only
    pose, one profile per video (NOT per frame -- 85 videos weighted equally,
    so a single 10-minute video cannot dominate the prior)."""
    inv = {v: k for k, v in CH.items()}
    per_video = {}
    with open(args.csv, newline='') as fh:
        for row in csv.DictReader(fh):
            bp = inv.get(row['bp'])
            if bp is None:                      # 'Neck' is synthesised, skip
                continue
            try:
                x, y = float(row['x']), float(row['y'])
            except (TypeError, ValueError):
                continue                        # NaN cell = keypoint not found
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            per_video.setdefault(row['video'], {}).setdefault(
                int(float(row['frame'])), {})[bp] = (x, y)

    rows = {r: [] for r in RATIOS}
    for video, frames in per_video.items():
        vals = {r: [] for r in RATIOS}
        for pt in frames.values():
            rr = ratios_from_points(pt)
            for r in RATIOS:
                if np.isfinite(rr[r]):
                    vals[r].append(rr[r])
        for r in RATIOS:
            if len(vals[r]) >= args.min_frames:
                rows[r].append(float(np.median(vals[r])))

    prior = {}
    for r in RATIOS:
        a = np.asarray(rows[r], dtype=float)
        if a.size < 5:
            raise SystemExit(
                'only %d videos yielded %s -- prior would be meaningless' %
                (a.size, r))
        # Robust spread: the IQR->sd conversion (1.349) shrugs off the one or
        # two videos where the released pose is itself off.
        q1, q3 = np.percentile(a, [25, 75])
        prior[r] = {'mean': float(np.median(a)),
                    'sd': float(max((q3 - q1) / 1.349, 1e-3)),
                    'n_videos': int(a.size),
                    'p05': float(np.percentile(a, 5)),
                    'p95': float(np.percentile(a, 95))}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {'source': str(args.csv), 'n_videos': len(per_video), 'ratios': prior},
        indent=2))
    print('infant prior from %d videos -> %s' % (len(per_video), args.out))
    for r in RATIOS:
        p = prior[r]
        print('  %-7s median=%.3f  sd=%.3f  [p05 %.3f, p95 %.3f]  n=%d'
              % (r, p['mean'], p['sd'], p['p05'], p['p95'], p['n_videos']))


# --------------------------------------------------------------------------
# select
# --------------------------------------------------------------------------
def read_frames(pred_dir):
    """Same two layouts convert_vitpose_json.py tolerates: one whole-video JSON
    (bare list or {'instance_info': [...]}) or many per-frame files."""
    files = sorted(glob.glob(os.path.join(pred_dir, '*.json')))
    if not files:
        return []
    if len(files) == 1:
        data = json.load(open(files[0]))
        if isinstance(data, dict):
            data = data.get('instance_info', [])
        return data

    def frame_no(p):
        digits = ''.join(c for c in os.path.basename(p) if c.isdigit())
        return int(digits) if digits else 0

    out = []
    for i, f in enumerate(sorted(files, key=frame_no)):
        d = json.load(open(f))
        insts = d.get('instances', d) if isinstance(d, dict) else d
        out.append({'frame_id': i, 'instances': insts})
    return out


def qa_sheet(video_dir, frames, track, out_path, n=6):
    """Contact sheet of the KEPT skeleton. Deliberately not optional-by-default:
    the whole point is that a human confirms an adult did not get through."""
    try:
        import cv2
    except ImportError:
        return False
    src = None
    for f in sorted(glob.glob(os.path.join(video_dir, '*.*'))):
        if f.rsplit('.', 1)[-1].lower() in ('mp4', 'avi', 'mov', 'mkv', 'wmv'):
            src = f
            break
    if src is None:
        return False

    keep = {fi: inst for fi, inst in track['frames']}
    picks = sorted(keep)[::max(1, len(keep) // n)][:n]
    cap = cv2.VideoCapture(src)
    tiles = []
    for fi in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        if not ok:
            continue
        kp = np.asarray(keep[fi]['keypoints'], dtype=float)
        sc = np.asarray(keep[fi]['keypoint_scores'], dtype=float)
        for i in range(min(len(kp), len(sc))):
            if sc[i] >= 0.3:
                cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), 4,
                           (0, 255, 0), -1)
        b = _bbox(keep[fi])
        if b:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (0, 255, 0), 2)
        cv2.putText(img, 'f%d' % fi, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)
        tiles.append(cv2.resize(img, (320, 240)))
    cap.release()
    if not tiles:
        return False
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    return True


def cmd_select(args):
    prior = json.loads(args.prior.read_text())['ratios']

    video_dirs = sorted(d for d in glob.glob(os.path.join(args.poses_root, '*'))
                        if os.path.isdir(os.path.join(d, 'pred')))
    if args.only:
        want = set(args.only.split(','))
        video_dirs = [d for d in video_dirs if os.path.basename(d) in want]
    if not video_dirs:
        raise SystemExit('no <video>/pred folders under ' + args.poses_root)

    args.out_root.mkdir(parents=True, exist_ok=True)
    report = []

    for vd in video_dirs:
        video = os.path.basename(vd)
        frames = read_frames(os.path.join(vd, 'pred'))
        if not frames:
            print('SKIP %s: no frames' % video)
            continue

        tracks = link_tracks(frames, args.iou, args.max_gap)
        profs = [track_profile(t, len(frames), args.kp_thr) for t in tracks]
        win, cost, margin = select_track(profs, prior, args.min_presence,
                                         args.min_ratios, args.max_cost)

        if win is None:
            why = ('best track cost=%.1f > max %.1f (not infant-like)'
                   % (cost, args.max_cost) if np.isfinite(cost)
                   else 'no track met presence/ratio minimums')
            print('DROP %-20s %s (%d tracks)' % (video, why, len(tracks)))
            report.append({'video': video, 'decision': 'dropped',
                           'n_tracks': len(tracks),
                           'cost': (round(cost, 2) if np.isfinite(cost) else ''),
                           'margin': '', 'coverage': '',
                           'r_head': '', 'r_arm': '', 'r_leg': '',
                           'n_frames': 0, 'note': why})
            continue

        trk = tracks[win['track_id']]
        keep = {fi: inst for fi, inst in trk['frames']}
        out_frames = [{'frame_id': fr.get('frame_id', fi),
                       'instances': ([keep[fi]] if fi in keep else [])}
                      for fi, fr in enumerate(frames)]

        pd_out = args.out_root / video / 'pred'
        pd_out.mkdir(parents=True, exist_ok=True)
        (pd_out / (video + '_infant.json')).write_text(json.dumps(out_frames))

        # convert_vitpose_json.py reads fps and frame size off the video sitting
        # next to pred/, so it has to be reachable from the new tree too.
        for f in sorted(glob.glob(os.path.join(vd, '*.*'))):
            if f.rsplit('.', 1)[-1].lower() in ('mp4', 'avi', 'mov', 'mkv', 'wmv'):
                link = args.out_root / video / os.path.basename(f)
                if not link.exists():
                    try:
                        link.symlink_to(os.path.abspath(f))
                    except OSError:
                        pass
                break

        # Three separate ways a video earns a human look. Coverage is not a
        # confidence problem but a data-loss one: if the kept track spans 8% of
        # the video, the other 92% of the infant's movement went in the bin
        # (usually scene cuts -- run scripts/32 first and select per scene).
        notes = []
        if cost > args.review_cost:
            notes.append('cost=%.1f' % cost)
        if margin < args.review_margin:
            notes.append('margin=%.2f' % margin)
        if win['presence'] < args.review_coverage:
            notes.append('coverage=%.0f%%' % (100 * win['presence']))
        flag = 'REVIEW' if notes else 'ok'

        print('%-24s track %-5d of %-4d cost=%-5.1f margin=%-6s cov=%3.0f%%  '
              'r_head=%.2f r_arm=%.2f r_leg=%.2f  %s %s'
              % (video, win['track_id'], len(tracks), cost,
                 ('only' if margin == float('inf') else '%.2f' % margin),
                 100 * win['presence'],
                 win['r_head'], win['r_arm'], win['r_leg'],
                 flag, ','.join(notes)))

        if not args.no_qa:
            qa_sheet(vd, frames, trk, args.out_root / 'qa' / (video + '.jpg'))

        report.append({
            'video': video, 'decision': flag, 'n_tracks': len(tracks),
            'cost': round(cost, 2),
            # 'only', not 'inf': one surviving candidate means NO competition,
            # which is the weakest evidence, not the strongest. Writing inf
            # here would sort it to the safe end of the review queue.
            'margin': ('only' if margin == float('inf') else round(margin, 3)),
            'coverage': round(win['presence'], 3),
            'r_head': round(win['r_head'], 3) if np.isfinite(win['r_head']) else '',
            'r_arm': round(win['r_arm'], 3) if np.isfinite(win['r_arm']) else '',
            'r_leg': round(win['r_leg'], 3) if np.isfinite(win['r_leg']) else '',
            'n_frames': win['n_frames'],
            'note': ','.join(notes),
        })

    # Riskiest first, so the top of the file IS the review queue: dropped
    # videos, then high-cost ones, then everything else by margin.
    def sort_key(r):
        if r['decision'] == 'dropped':
            return (0, 0.0)
        m = r['margin']
        return (1, -float(r['cost']) if m == 'only' else float(m))
    report.sort(key=sort_key)

    rp = args.out_root / 'report.csv'
    with rp.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)

    n_rev = sum(1 for r in report if r['decision'] == 'REVIEW')
    n_drop = sum(1 for r in report if r['decision'] == 'dropped')
    print('\n%d videos -> %s' % (len(report), rp))
    print('  %d flagged for review (high cost / low margin / low coverage), '
          '%d dropped' % (n_rev, n_drop))
    print('  review the top of report.csv against %s/qa/*.jpg BEFORE '
          'feeding this to convert_vitpose_json.py' % args.out_root)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    c = sub.add_parser('calibrate', help='learn infant proportions from '
                                         "Chambers' vetted infant-only pose")
    c.add_argument('--csv', type=Path,
                   default=Path('Chamber-etal-dataset/'
                                'pose_estimates_youtube_dataset.csv'))
    c.add_argument('--out', type=Path, default=DEFAULT_PRIOR)
    c.add_argument('--min-frames', type=int, default=30,
                   help='a video needs this many measurable frames to '
                        'contribute a ratio (default 30)')
    c.set_defaults(func=cmd_calibrate)

    s = sub.add_parser('select', help='keep the infant track per video')
    s.add_argument('--poses-root', default='outputs/poses')
    s.add_argument('--out-root', type=Path, default=Path('outputs/poses_infant'))
    s.add_argument('--prior', type=Path, default=DEFAULT_PRIOR)
    s.add_argument('--only', help='comma-separated video ids')
    s.add_argument('--kp-thr', type=float, default=0.3,
                   help='keypoint score below which a point is unusable '
                        '(matches convert_vitpose_json.py)')
    s.add_argument('--iou', type=float, default=0.3, help='tracker IoU gate')
    s.add_argument('--max-gap', type=int, default=15,
                   help='frames a track survives without a detection')
    s.add_argument('--min-presence', type=float, default=0.05,
                   help='drop tracks present in less than this fraction of '
                        'frames -- passers-by, not the subject')
    s.add_argument('--min-ratios', type=int, default=2,
                   help='a track needs this many measurable ratios to be '
                        'scored at all (default 2 of 3)')
    s.add_argument('--max-cost', type=float, default=6.0,
                   help='DROP the video if even the best track is this far '
                        'from infant proportions (mean squared z, so 6.0 is '
                        'about 2.5 sd on every ratio at once). This is the '
                        'gate that stops a lone adult winning by default')
    s.add_argument('--review-cost', type=float, default=3.0,
                   help='flag for review above this cost (default 3.0)')
    s.add_argument('--review-margin', type=float, default=0.5,
                   help='flag videos where the runner-up came within this '
                        'relative cost gap (default 0.5)')
    s.add_argument('--review-coverage', type=float, default=0.5,
                   help='flag when the kept track spans less than this '
                        'fraction of the video -- usually scene cuts '
                        'fragmenting one infant (default 0.5)')
    s.add_argument('--no-qa', action='store_true',
                   help='skip contact sheets (needs cv2 + the source video)')
    s.set_defaults(func=cmd_select)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
