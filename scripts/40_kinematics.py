#!/usr/bin/env python
"""
Kinematic features from the pose JSONs written by scripts/30_infant_pose.py.

    python scripts/40_kinematics.py --poses outputs/poses/0HkcGRBsPUM
    python scripts/40_kinematics.py --all                 # every video under outputs/poses
    python scripts/40_kinematics.py --poses outputs/poses/0HkcGRBsPUM --fps 25

CPU only -- runs on a login node in any env with numpy (scipy optional, for
Savitzky-Golay smoothing; falls back to a moving average without it).

PIPELINE, per video:
  1. Load pred/*.json (MMPose format: per frame, a list of person instances,
     each with 17 COCO keypoints [x, y] + per-keypoint scores).
  2. Pick ONE person and follow them: highest-scoring instance in the first
     usable frame, then greedy nearest-centroid tracking with a max-jump gate.
     (Script 30 poses EVERY person; the infant/child filter upstream is the
     26_*.jsonl annotations -- here we just keep one consistent body.)
  3. Clean: keypoints below --min-score become NaN, gaps up to --max-gap-sec
     are linearly interpolated, then the tracks are smoothed.
  4. Normalise to body scale: median trunk length (mid-shoulder to mid-hip)
     = 1 unit, so speeds are in trunk-lengths/second and comparable across
     zoom levels and videos.
  5. Differentiate: speed, acceleration, jerk per joint; joint angles at
     elbows/knees/hips/shoulders and their angular velocities.

OUTPUTS, under outputs/kinematics/<video_id>/:
  timeseries.csv   one row per frame: time_sec + per-joint x, y, score, speed
  features.json    all summary features, nested, with provenance (fps, scale)
  features.csv     the same features flattened to ONE ROW -- concatenate these
                   across videos for a feature matrix
With --all, also writes outputs/kinematics/all_features.csv (one row per video).

FEATURES (all speeds in trunk-lengths/sec, angles in degrees):
  per joint-group (wrists, ankles, knees, elbows, head, all17):
      speed mean/std/median/p95/max, acceleration mean/p95, jerk mean/p95,
      path length per second, fraction of time moving (> --move-thresh)
  per angle (elbow/knee/hip/shoulder, L and R):
      mean, std, range, angular velocity mean/p95
  whole body:
      centroid speed stats, centroid net displacement,
      left/right symmetry (Pearson r of wrist speeds, of ankle speeds),
      dominant movement frequency in 0.3-5 Hz + its power fraction
      (FFT of mean limb speed -- the band where rhythmic infant/child
      movement like kicking, clapping, stepping lives),
      tracking quality: frames tracked, mean keypoint score, longest gap.
"""
import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.signal import savgol_filter
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")

# COCO-17 keypoint order (what ViTPose/MMPose emit).
KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]
KP = {name: i for i, name in enumerate(KEYPOINTS)}

GROUPS = {
    "wrists": ["left_wrist", "right_wrist"],
    "ankles": ["left_ankle", "right_ankle"],
    "knees": ["left_knee", "right_knee"],
    "elbows": ["left_elbow", "right_elbow"],
    "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    "all17": KEYPOINTS,
}

# angle name -> (A, vertex, B): the angle at `vertex` between rays to A and B.
ANGLES = {
    "elbow_l": ("left_shoulder", "left_elbow", "left_wrist"),
    "elbow_r": ("right_shoulder", "right_elbow", "right_wrist"),
    "knee_l": ("left_hip", "left_knee", "left_ankle"),
    "knee_r": ("right_hip", "right_knee", "right_ankle"),
    "hip_l": ("left_shoulder", "left_hip", "left_knee"),
    "hip_r": ("right_shoulder", "right_hip", "right_knee"),
    "shoulder_l": ("left_elbow", "left_shoulder", "left_hip"),
    "shoulder_r": ("right_elbow", "right_shoulder", "right_hip"),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _instances(obj):
    """MMPose writes either a bare instance list or {'instances': [...]}."""
    if isinstance(obj, dict):
        return obj.get("instances", [])
    return obj if isinstance(obj, list) else []


def _frame_key(path: Path):
    """Natural sort: frame_10.json after frame_9.json, not after frame_1."""
    nums = re.findall(r"\d+", path.stem)
    return (int(nums[-1]) if nums else 0, path.stem)


def load_pred_frames(pred_dir: Path):
    """Return a list of frames; each frame is a list of instance dicts."""
    files = sorted(pred_dir.glob("*.json"), key=_frame_key)
    if not files:
        raise SystemExit(f"no .json under {pred_dir}")
    if len(files) == 1:
        # Whole-video export: one file, a list with one entry per frame
        # (each {'frame_id': i, 'instances': [...]} or a bare instance list).
        data = json.loads(files[0].read_text())
        if isinstance(data, list) and data and (
                isinstance(data[0], dict) and "instances" in data[0]
                or isinstance(data[0], list)):
            return [_instances(f) for f in data]
        return [_instances(data)]
    return [_instances(json.loads(f.read_text())) for f in files]


def instance_xys(inst):
    """(17,2) float array + (17,) scores from one MMPose instance dict."""
    kpts = np.asarray(inst.get("keypoints", []), dtype=float)
    if kpts.size == 0:
        return None, None
    if kpts.ndim == 1:                      # flat [x0, y0, (s0,) x1, ...]
        kpts = kpts.reshape(-1, 3 if kpts.size % 3 == 0 else 2)
    scores = inst.get("keypoint_scores")
    if scores is None and kpts.shape[1] >= 3:
        scores = kpts[:, 2]
    scores = (np.asarray(scores, dtype=float) if scores is not None
              else np.ones(len(kpts)))
    if len(kpts) < 17:
        return None, None
    return kpts[:17, :2], scores[:17]


def probe_fps(video: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate",
             "-of", "default=nw=1:nk=1", str(video)],
            capture_output=True, text=True, timeout=30).stdout.strip()
        num, _, den = out.partition("/")
        fps = float(num) / float(den or 1)
        return fps if fps > 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tracking: one consistent person through the clip
# ---------------------------------------------------------------------------
def track_person(frames, min_score: float, max_jump_frac: float = 0.5):
    """Greedy nearest-centroid track.

    Seed on the highest-mean-score instance in the first frame that has one;
    thereafter take the instance whose centroid is nearest the last known
    centroid, gated at max_jump_frac * image-diagonal-proxy per frame so an
    adult walking past cannot steal the track. Returns (T,17,2) xy and
    (T,17) scores with NaN/0 where the person was not found.
    """
    T = len(frames)
    xy = np.full((T, 17, 2), np.nan)
    sc = np.zeros((T, 17))
    last_c, span = None, None
    for t, insts in enumerate(frames):
        cands = []
        for inst in insts:
            k, s = instance_xys(inst)
            if k is None or np.nanmean(s) < min_score * 0.5:
                continue
            good = s >= min_score
            if good.sum() < 4:
                continue
            cands.append((k, s, np.nanmean(k[good], axis=0)))
        if not cands:
            continue
        if last_c is None:
            k, s, c = max(cands, key=lambda x: float(np.nanmean(x[1])))
        else:
            k, s, c = min(cands, key=lambda x: float(np.hypot(*(x[2] - last_c))))
            if span and float(np.hypot(*(c - last_c))) > max_jump_frac * span:
                continue                     # too far to be the same person
        xy[t], sc[t] = k, s
        last_c = c
        ext = np.nanmax(k, axis=0) - np.nanmin(k, axis=0)
        span = float(np.hypot(*ext)) or span
    return xy, sc


# ---------------------------------------------------------------------------
# Cleaning + normalisation
# ---------------------------------------------------------------------------
def interp_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN runs of length <= max_gap in a 1-D series."""
    x = x.copy()
    good = np.flatnonzero(~np.isnan(x))
    if good.size < 2:
        return x
    bad = np.isnan(x)
    filled = np.interp(np.arange(len(x)), good, x[good])
    # Only accept fills inside short-enough gaps (and never extrapolate).
    run_start = None
    for i in range(len(x) + 1):
        if i < len(x) and bad[i] and good[0] < i < good[-1]:
            run_start = i if run_start is None else run_start
        else:
            if run_start is not None and i - run_start <= max_gap:
                x[run_start:i] = filled[run_start:i]
            run_start = None
    return x


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """NaN-tolerant smoothing; savgol when scipy is around, else box filter."""
    if window < 3 or np.isnan(x).all():
        return x
    if HAVE_SCIPY:
        out = x.copy()
        ok = ~np.isnan(x)
        # savgol cannot see NaN: smooth contiguous valid runs independently.
        idx = np.flatnonzero(np.diff(np.concatenate(([0], ok.view(np.int8), [0]))))
        for a, b in zip(idx[::2], idx[1::2]):
            n = b - a
            if n >= window:
                w = window if window % 2 else window - 1
                out[a:b] = savgol_filter(x[a:b], w, polyorder=2)
        return out
    kernel = np.ones(window) / window
    pad = np.pad(x, window // 2, mode="edge")
    return np.convolve(np.nan_to_num(pad, nan=np.nanmedian(x)), kernel, "valid")[:len(x)]


def trunk_scale(xy: np.ndarray) -> float:
    """Median mid-shoulder-to-mid-hip distance in pixels (body-size unit)."""
    mid_sh = (xy[:, KP["left_shoulder"]] + xy[:, KP["right_shoulder"]]) / 2
    mid_hip = (xy[:, KP["left_hip"]] + xy[:, KP["right_hip"]]) / 2
    d = np.hypot(*(mid_sh - mid_hip).T)
    med = float(np.nanmedian(d))
    return med if med and med > 1e-6 else float("nan")


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def stats(x: np.ndarray, prefix: str) -> dict:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {f"{prefix}_{k}": None for k in
                ("mean", "std", "median", "p95", "max")}
    return {f"{prefix}_mean": float(np.mean(x)),
            f"{prefix}_std": float(np.std(x)),
            f"{prefix}_median": float(np.median(x)),
            f"{prefix}_p95": float(np.percentile(x, 95)),
            f"{prefix}_max": float(np.max(x))}


def joint_angle(xy: np.ndarray, a: str, v: str, b: str) -> np.ndarray:
    """Angle at vertex v (degrees, 0-180) over time."""
    u1 = xy[:, KP[a]] - xy[:, KP[v]]
    u2 = xy[:, KP[b]] - xy[:, KP[v]]
    dot = (u1 * u2).sum(axis=1)
    den = np.hypot(*u1.T) * np.hypot(*u2.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / den, -1, 1)
    return np.degrees(np.arccos(cos))


def dominant_freq(speed: np.ndarray, fps: float, lo=0.3, hi=5.0):
    """(freq_hz, power_fraction) of the strongest component in [lo, hi] Hz."""
    x = speed[~np.isnan(speed)]
    if x.size < int(2 * fps):                # need ~2s of signal
        return None, None
    x = x - x.mean()
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, d=1.0 / fps)
    band = (f >= lo) & (f <= hi)
    if not band.any() or p.sum() <= 0:
        return None, None
    i = np.argmax(p * band)
    return float(f[i]), float(p[i] / p.sum())


def compute_features(xy, sc, fps, move_thresh, tracked):
    T = xy.shape[0]
    dt = 1.0 / fps
    feats = {}

    speed = np.hypot(*np.gradient(xy, dt, axis=0).transpose(2, 0, 1))  # (T,17)
    acc = np.abs(np.gradient(speed, dt, axis=0))
    jerk = np.abs(np.gradient(acc, dt, axis=0))

    for gname, names in GROUPS.items():
        idx = [KP[n] for n in names]
        gs = np.nanmean(speed[:, idx], axis=1)
        ga = np.nanmean(acc[:, idx], axis=1)
        gj = np.nanmean(jerk[:, idx], axis=1)
        feats.update(stats(gs, f"{gname}_speed"))
        feats[f"{gname}_acc_mean"] = _nanmean(ga)
        feats[f"{gname}_acc_p95"] = _nanp95(ga)
        feats[f"{gname}_jerk_mean"] = _nanmean(gj)
        feats[f"{gname}_jerk_p95"] = _nanp95(gj)
        valid = gs[~np.isnan(gs)]
        feats[f"{gname}_path_per_sec"] = (float(np.sum(valid) * dt / (len(valid) * dt))
                                          if valid.size else None)
        feats[f"{gname}_frac_moving"] = (float(np.mean(valid > move_thresh))
                                         if valid.size else None)

    for aname, (a, v, b) in ANGLES.items():
        ang = joint_angle(xy, a, v, b)
        ok = ang[~np.isnan(ang)]
        if ok.size:
            feats[f"angle_{aname}_mean"] = float(np.mean(ok))
            feats[f"angle_{aname}_std"] = float(np.std(ok))
            feats[f"angle_{aname}_range"] = float(np.ptp(ok))
        else:
            feats[f"angle_{aname}_mean"] = feats[f"angle_{aname}_std"] = \
                feats[f"angle_{aname}_range"] = None
        av = np.abs(np.gradient(ang, dt))
        feats[f"angvel_{aname}_mean"] = _nanmean(av)
        feats[f"angvel_{aname}_p95"] = _nanp95(av)

    centroid = np.nanmean(xy, axis=1)                       # (T,2)
    cs = np.hypot(*np.gradient(centroid, dt, axis=0).T)
    feats.update(stats(cs, "centroid_speed"))
    ok = ~np.isnan(centroid).any(axis=1)
    feats["centroid_net_displacement"] = (
        float(np.hypot(*(centroid[ok][-1] - centroid[ok][0]))) if ok.sum() >= 2 else None)

    for pair, l, r in (("wrist", "left_wrist", "right_wrist"),
                       ("ankle", "left_ankle", "right_ankle")):
        a, b = speed[:, KP[l]], speed[:, KP[r]]
        ok = ~np.isnan(a) & ~np.isnan(b)
        feats[f"symmetry_{pair}_speed_corr"] = (
            float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > int(fps) else None)

    limb_idx = [KP[n] for n in GROUPS["wrists"] + GROUPS["ankles"]]
    f0, pf = dominant_freq(np.nanmean(speed[:, limb_idx], axis=1), fps)
    feats["dominant_freq_hz"] = f0
    feats["dominant_freq_power_frac"] = pf

    feats["n_frames"] = int(T)
    feats["n_frames_tracked"] = int(tracked.sum())
    feats["frac_tracked"] = float(tracked.mean()) if T else None
    feats["mean_kpt_score"] = _nanmean(np.where(sc > 0, sc, np.nan))
    gaps = np.diff(np.flatnonzero(np.concatenate(([1], tracked, [1]))))
    feats["longest_gap_frames"] = int(gaps.max() - 1) if gaps.size else 0
    return feats, speed


def _nanmean(x):
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if x.size else None


def _nanp95(x):
    x = x[~np.isnan(x)]
    return float(np.percentile(x, 95)) if x.size else None


# ---------------------------------------------------------------------------
# Per-video driver
# ---------------------------------------------------------------------------
def process_video(pose_dir: Path, out_root: Path, args) -> dict | None:
    pred_dir = pose_dir / "pred"
    if not pred_dir.is_dir():
        print(f"skip {pose_dir.name}: no pred/", file=sys.stderr)
        return None

    fps = args.fps
    if not fps:
        vids = sorted(pose_dir.glob("*.mp4")) + sorted(pose_dir.glob("*.mov"))
        fps = probe_fps(vids[0]) if vids else None
    if not fps:
        fps = 30.0
        print(f"WARN {pose_dir.name}: fps unknown, assuming 30 "
              f"(pass --fps to fix)", file=sys.stderr)

    frames = load_pred_frames(pred_dir)
    xy, sc = track_person(frames, args.min_score)
    xy[sc < args.min_score] = np.nan

    scale = trunk_scale(xy)
    if math.isnan(scale):
        print(f"skip {pose_dir.name}: trunk never visible, no scale",
              file=sys.stderr)
        return None

    # Tracking quality is judged on the RAW track, before gap-filling makes
    # every short dropout invisible.
    tracked = ~np.isnan(xy[:, :, 0]).all(axis=1)

    max_gap = max(1, int(round(args.max_gap_sec * fps)))
    window = max(3, int(round(args.smooth_sec * fps)) | 1)   # odd
    for j in range(17):
        for d in range(2):
            xy[:, j, d] = smooth(interp_gaps(xy[:, j, d], max_gap), window)
    xy_norm = xy / scale

    feats, speed = compute_features(xy_norm, sc, fps, args.move_thresh, tracked)
    feats.update({"video_id": pose_dir.name, "fps": float(fps),
                  "trunk_scale_px": scale, "min_score": args.min_score})

    out = out_root / pose_dir.name
    out.mkdir(parents=True, exist_ok=True)

    with (out / "timeseries.csv").open("w", newline="") as fh:
        cols = ["frame", "time_sec"] + [f"{n}_{d}" for n in KEYPOINTS
                                        for d in ("x", "y", "score", "speed")]
        w = csv.writer(fh)
        w.writerow(cols)
        for t in range(xy.shape[0]):
            row = [t, round(t / fps, 4)]
            for j in range(17):
                row += [_r(xy[t, j, 0]), _r(xy[t, j, 1]),
                        _r(sc[t, j]), _r(speed[t, j])]
            w.writerow(row)

    (out / "features.json").write_text(json.dumps(feats, indent=2))
    front = ["video_id", "fps", "n_frames", "n_frames_tracked", "frac_tracked",
             "trunk_scale_px"]
    ordered = {k: feats[k] for k in front} | {
        k: v for k, v in sorted(feats.items()) if k not in front}
    with (out / "features.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ordered))
        w.writeheader()
        w.writerow(ordered)

    print(f"{pose_dir.name}: {feats['n_frames_tracked']}/{feats['n_frames']} "
          f"frames tracked, fps={fps:.4g}, scale={scale:.1f}px -> {out}")
    return ordered


def _r(v):
    return "" if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--poses", type=Path,
                    help="one outputs/poses/<video_id> directory")
    ap.add_argument("--all", action="store_true",
                    help="every directory under --poses-root with a pred/")
    ap.add_argument("--poses-root", type=Path, default=ROOT / "outputs/poses")
    ap.add_argument("--out", type=Path, default=None,
                    help="output root (default: <poses-root>/../kinematics)")
    ap.add_argument("--fps", type=float, default=None,
                    help="frame rate; default: ffprobe the video next to pred/")
    ap.add_argument("--min-score", type=float, default=0.3,
                    help="keypoints below this confidence become NaN")
    ap.add_argument("--max-gap-sec", type=float, default=0.5,
                    help="interpolate tracking gaps up to this long")
    ap.add_argument("--smooth-sec", type=float, default=0.2,
                    help="smoothing window length")
    ap.add_argument("--move-thresh", type=float, default=0.05,
                    help="speed (trunk-lengths/s) above which a joint counts "
                         "as moving")
    args = ap.parse_args()

    if not args.poses and not args.all:
        ap.error("need --poses <dir> or --all")
    out_root = args.out or (args.poses_root.parent / "kinematics")

    dirs = ([args.poses] if args.poses else
            sorted(d for d in args.poses_root.iterdir()
                   if (d / "pred").is_dir()))
    rows = [r for d in dirs if (r := process_video(d, out_root, args))]

    if args.all and rows:
        combined = out_root / "all_features.csv"
        fields = sorted({k for r in rows for k in r},
                        key=lambda k: (k != "video_id", k))
        with combined.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\ncombined: {combined} ({len(rows)} videos)")


if __name__ == "__main__":
    main()
