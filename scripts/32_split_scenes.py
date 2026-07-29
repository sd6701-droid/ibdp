#!/usr/bin/env python
"""
Cut a compilation video back into its ORIGINAL source clips -- video AND the
audio that belongs to it -- using TransNetV2 shot-boundary detection.

    python scripts/32_split_scenes.py --only 8yDn1uFbs4s

    
    <outdir>/8yDn1uFbs4s/
        scenes.json          every boundary, in frames and seconds, with scores
        scenes.csv           the same thing you can eyeball in a terminal
        split_01/clip.mp4    scene 1, video + its audio, exact boundaries
        split_01/meta.json   start/end/duration/cut confidence for that clip
        split_02/...

WHY THIS EXISTS. 26_describe_segments_hf.py cuts on a FIXED 10s grid, which is
blind to the edit: a window routinely straddles two unrelated source clips, and
"how many infants are in this window" then has no single right answer. These
videos are compilations -- 20-60 unrelated clips glued together -- so the edit
points are the only segmentation the content actually has.

WHY TransNetV2 AND NOT PySceneDetect. PySceneDetect thresholds a frame-to-frame
colour/content difference. A compilation of infant videos breaks that in both
directions: a baby lunging at the camera, a pan across a bright window or a
cut-to-flash makes the difference spike WITHOUT an edit, and a cut between two
clips shot in the same beige living room barely moves it at all. TransNetV2 is
a 3D-CNN trained on shot transitions specifically, handles gradual dissolves,
and is the standard baseline on ClipShots/BBC Planet Earth. AdaptiveDetector is
kept behind --detector pyscenedetect as a no-weights fallback, not as the
default.

WEIGHTS. The architecture is vendored next to this file (transnetv2_pytorch.py,
MIT, upstream verbatim); the .pth is a bare state_dict fetched separately:

    scripts/13_fetch_models.sh --only transnetv2      # 30MB, login node

which lands at $ROOT/models/TransNetV2/transnetv2-pytorch-weights.pth. That is
the official Souček conversion, mirrored on the Hub (Sn4kehead/TransNetV2;
ByteDance/shot2story and MiaoshouAI ship the identical file).

CUTS ARE RE-ENCODED, NEVER -c copy. A stream copy snaps the start to the
previous keyframe -- on a YouTube 1080p mp4 that is up to 5s of the WRONG clip
welded onto the front, and the audio then leads the video by that much. libx264
at crf 18 costs seconds per clip and is frame-accurate. Same reasoning, and the
same -ss/-t form, as _cut() in 26_describe_segments_hf.py.

    -ss BEFORE -i     fast seek; still frame-accurate because ffmpeg decodes
                      from the preceding keyframe and discards.
    -t, NOT -to       -to as an INPUT option has changed meaning between ffmpeg
                      majors. A silently mis-cut clip looks like a detector
                      error, which is the most expensive kind of bug here.
    -map 0:a?         '?' so a silent video yields a clip instead of a failure.

WHAT THIS DOES NOT DO. Detecting cuts tells you "a new source clip starts
here". It does NOT tell you that split_03 and split_17 are both crawling --
that is a second, semantic pass over the clips this script writes:

    python scripts/26_describe_segments_hf.py \
        --model $ROOT/models/Qwen3-Omni-30B-A3B-Instruct \
        --videos $ROOT/outputs/scenes/8yDn1uFbs4s/clips

RUN IT TWICE, SAFELY. --resume keeps clips whose recorded boundaries still
match; stale split_* directories from a previous run with different parameters
are DELETED, because a 40-split run followed by a 30-split run would otherwise
leave split_31..40 behind, indistinguishable from real output.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

# transnetv2_pytorch.py sits next to this file; sbatch jobs cd elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")
DEFAULT_WEIGHTS = ROOT / "models/TransNetV2/transnetv2-pytorch-weights.pth"

# TransNetV2 is trained at exactly this resolution. It is not a knob.
NET_W, NET_H = 48, 27


# ---------------------------------------------------------------------------
# ffprobe / ffmpeg
# ---------------------------------------------------------------------------

def probe(clip: Path) -> dict:
    """duration, fps (exact Fraction), frame count, audio presence.

    fps is a Fraction and stays one all the way to the timestamps: 30000/1001
    is not 29.97, and rounding it turns into a drift of ~1.8 frames per minute
    -- half a second by the end of a ten-minute compilation, which is enough to
    put the last cuts inside the wrong clip.
    """
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(clip)],
        capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {clip.name}: {p.stderr.strip()}")
    d = json.loads(p.stdout or "{}")
    streams = d.get("streams") or []

    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise RuntimeError(f"{clip.name} has no video stream")

    # avg_frame_rate first (whole-file average), r_frame_rate as fallback --
    # the latter is the smallest rate that ticks every frame, which for a VFR
    # file can be an absurd 1000/1.
    fps = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = v.get(key) or ""
        try:
            f = Fraction(raw)
        except (ValueError, ZeroDivisionError):
            continue
        if f > 0:
            fps = f
            break
    if fps is None:
        raise RuntimeError(f"{clip.name}: ffprobe reports no usable frame rate")

    dur = None
    for src in (d.get("format") or {}, v):
        try:
            dur = float(src.get("duration"))
            break
        except (TypeError, ValueError):
            continue

    nb = v.get("nb_frames")
    try:
        n_frames = int(nb)
    except (TypeError, ValueError):
        n_frames = None

    return {
        "duration": dur,
        "fps": fps,
        "n_frames": n_frames,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "width": v.get("width"),
        "height": v.get("height"),
        "vcodec": v.get("codec_name"),
    }


def decode_small(clip: Path, fps: Fraction):
    """Whole video as uint8 [N, 27, 48, 3] RGB, one row per frame.

    Tiny on purpose: 48x27 is 3.9KB a frame, so a ten-minute video is ~60MB in
    RAM and decoding is I/O bound rather than pixel bound.

    THE fps FILTER IS LOAD-BEARING. Frame index is converted to a timestamp by
    dividing by fps, and that is only true if the stream is constant rate.
    yt-dlp output usually is, but not always; `fps=` resamples a VFR stream to
    a constant one by duplicating and dropping frames, so index/fps stays exact
    either way. Without it, one variable-rate video produces boundaries that
    drift further out of sync the longer the video runs -- and nothing in the
    output says so.
    """
    import numpy as np

    cmd = ["ffmpeg", "-v", "error", "-i", str(clip), "-map", "0:v:0",
           "-vf", f"fps={fps.numerator}/{fps.denominator},"
                  f"scale={NET_W}:{NET_H}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed on {clip.name}: "
                           f"{p.stderr.decode('utf8', 'replace').strip()}")

    stride = NET_W * NET_H * 3
    n = len(p.stdout) // stride
    if n == 0:
        raise RuntimeError(f"{clip.name}: decoded 0 frames -- unreadable video")
    if len(p.stdout) % stride:
        # A partial trailing frame means the decode was cut short; say so
        # rather than silently analysing a video that stops early.
        print(f"    WARNING: {len(p.stdout) % stride} trailing bytes -- "
              f"decode may have been truncated", flush=True)
    return np.frombuffer(p.stdout[:n * stride], np.uint8).reshape(
        [n, NET_H, NET_W, 3])


# ---------------------------------------------------------------------------
# Detectors
#
# Both return the same thing: a list of [first_frame, last_frame] pairs,
# inclusive, plus a per-frame transition score or None. Everything downstream
# -- merging, subdividing, cutting, the JSON -- is detector-agnostic.
# ---------------------------------------------------------------------------

class TransNetV2Detector:
    """The real one. ~250 frames/s on an A100, ~30 frames/s on CPU."""

    def __init__(self, weights: Path, device: str, batch: int):
        import torch
        from transnetv2_pytorch import TransNetV2

        if not Path(weights).is_file():
            raise SystemExit(
                f"TransNetV2 weights not found: {weights}\n"
                f"  Fetch them from a LOGIN node (30MB):\n"
                f"    scripts/13_fetch_models.sh --only transnetv2")

        self.torch = torch
        self.batch = batch
        self.device = device

        model = TransNetV2()
        sd = torch.load(str(weights), map_location="cpu", weights_only=True)
        # Some mirrors wrap the tensors; the official file does not.
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # strict=True, deliberately. A renamed or partially-loaded key would
        # leave randomly-initialised layers and produce cuts that look
        # plausible and are noise -- the one failure mode you cannot spot by
        # reading the output.
        model.load_state_dict(sd)
        self.model = model.eval().to(device)

    def _windows(self, frames):
        """Upstream's input_iterator, unchanged in behaviour.

        100-frame windows at stride 50, of which only the middle 50 are kept:
        the net needs context on both sides of a frame to call it a transition,
        so the first and last 25 of every window are context, not output. The
        video is padded with copies of its first and last frame so the true
        first and last frames get that context too.
        """
        import numpy as np

        pad_start = 25
        rem = len(frames) % 50
        pad_end = 25 + 50 - (rem if rem != 0 else 50)
        padded = np.concatenate(
            [np.repeat(frames[:1], pad_start, axis=0),
             frames,
             np.repeat(frames[-1:], pad_end, axis=0)], axis=0)

        ptr = 0
        while ptr + 100 <= len(padded):
            yield padded[ptr:ptr + 100]
            ptr += 50

    def predict(self, frames) -> list:
        """[0,1] transition probability per frame, len == len(frames)."""
        import numpy as np

        torch = self.torch
        preds, buf = [], []

        def flush():
            if not buf:
                return
            x = torch.from_numpy(np.ascontiguousarray(np.stack(buf)))
            with torch.inference_mode():
                logits, extra = self.model(x.to(self.device))
            # logits is the SINGLE-frame head -- "this frame is a transition".
            # extra["many_hot"] is the all-frames head, which fires across the
            # whole of a gradual dissolve; upstream cuts on the single-frame
            # one and so do we.
            single = torch.sigmoid(logits)[:, 25:75, 0].float().cpu().numpy()
            preds.extend(single.reshape(-1).tolist())
            buf.clear()

        for w in self._windows(frames):
            buf.append(w)
            if len(buf) == self.batch:
                flush()
        flush()
        return preds[:len(frames)]

    def scenes(self, frames):
        preds = self.predict(frames)
        return predictions_to_scenes(preds, self.threshold), preds

    # threshold is set by main() before scenes() is called
    threshold = 0.5


class PySceneDetectDetector:
    """No-weights fallback: AdaptiveDetector, which is content difference
    normalised by a rolling window and therefore tolerant of camera motion.

    Here as an escape hatch for a node with no weights and to sanity-check
    TransNetV2's boundaries against a completely different method -- not as the
    default. It has no per-frame score, so cut_confidence comes out null.
    """

    threshold = None

    def __init__(self, clip: Path, fps: Fraction, threshold: float):
        try:
            from scenedetect import AdaptiveDetector, SceneManager, open_video
        except ImportError:
            raise SystemExit(
                "--detector pyscenedetect needs the scenedetect package:\n"
                "  conda activate ibdp && pip install 'scenedetect[opencv]'")
        self._open_video = open_video
        self._SceneManager = SceneManager
        self._AdaptiveDetector = AdaptiveDetector
        self.clip = clip
        self.fps = fps
        self.adaptive_threshold = threshold

    def scenes(self, _frames_unused):
        video = self._open_video(str(self.clip))
        mgr = self._SceneManager()
        mgr.add_detector(
            self._AdaptiveDetector(adaptive_threshold=self.adaptive_threshold))
        mgr.detect_scenes(video, show_progress=False)
        out = []
        for start, end in mgr.get_scene_list():
            # get_scene_list() end is EXCLUSIVE; our contract is an inclusive
            # last frame, hence the -1.
            out.append([start.get_frames(), max(start.get_frames(),
                                                end.get_frames() - 1)])
        return (out or [[0, 0]]), None


def predictions_to_scenes(preds: list, threshold: float) -> list:
    """Per-frame transition probability -> [[first, last], ...] inclusive.

    Port of TransNetV2.predictions_to_scenes (upstream, TF inference file), kept
    line-for-line equivalent so our boundaries match the reference
    implementation's .scenes.txt exactly. Scenes are the RUNS BELOW threshold;
    the frames above it are the transitions between them.
    """
    if not preds:
        return []
    p = [1 if x > threshold else 0 for x in preds]

    scenes, t, t_prev, start = [], 0, 0, 0
    i = 0
    for i, t in enumerate(p):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    # Everything above threshold -- a video that is one long dissolve, or a
    # threshold set far too low. One scene covering the lot beats zero.
    if not scenes:
        return [[0, len(p) - 1]]
    return scenes


# ---------------------------------------------------------------------------
# Frames -> seconds -> the splits we actually cut
# ---------------------------------------------------------------------------

def frames_to_seconds(scenes: list, fps: Fraction, preds, duration) -> list:
    """Inclusive frame pairs -> {start, end} in seconds, plus cut confidence.

    end is (last_frame + 1)/fps, not last_frame/fps: the frame is a duration,
    not an instant, and dropping its length loses one frame at every boundary
    -- a whole frame of the clip, every time, silently.
    """
    out = []
    for k, (a, b) in enumerate(scenes):
        start = a / fps
        end = (b + 1) / fps
        if duration is not None:
            end = min(end, duration)
            start = min(start, end)

        # Confidence of the cut that ENDS this scene: the strongest transition
        # score between this scene's last frame and the next scene's first.
        # Low values here are where to look when a boundary is wrong -- and
        # they are exactly the ones --min-confidence can drop.
        conf = None
        if preds is not None and k + 1 < len(scenes):
            gap = preds[b:scenes[k + 1][0] + 1]
            conf = max(gap) if gap else None

        out.append({"start": float(start), "end": float(end),
                    "start_frame": int(a), "end_frame": int(b),
                    "cut_confidence": (float(conf) if conf is not None
                                       else None)})
    return out


def enforce_lengths(scenes: list, min_len: float, max_len: float) -> list:
    """Merge runt scenes into their neighbour; subdivide over-long ones.

    MERGE FIRST. A compilation is full of 3-frame flashes -- a title card, a
    white-flash transition, one frame of the next clip leaking through a
    dissolve. Each of those is a legitimate shot boundary and a useless clip:
    a 0.1s mp4 that no VLM can say anything about, occupying a split_NN slot
    that reads like real content.

    SUBDIVIDE SECOND, and only when asked. A 40s source clip is one scene and
    that is the truth about the edit; --max-len 10 says "I also want windows a
    video model can chew", which is a different requirement. Subdivided pieces
    are flagged in meta.json (subdivided/part/parts) so the two are never
    confused downstream.
    """
    merged = []
    for s in scenes:
        if merged and (s["end"] - s["start"]) < min_len:
            merged[-1]["end"] = s["end"]
            # .get, not []: a scenes.json written by hand or by an older
            # version carries seconds only, and frame indices are a nicety.
            merged[-1]["end_frame"] = s.get("end_frame")
            merged[-1]["cut_confidence"] = s.get("cut_confidence")
            merged[-1]["merged_runts"] = merged[-1].get("merged_runts", 0) + 1
        else:
            merged.append(dict(s))

    # The FIRST scene has no previous to merge into, so it survives the loop
    # above however short it is. Fold it forwards instead.
    while len(merged) > 1 and (merged[0]["end"] - merged[0]["start"]) < min_len:
        merged[1]["start"] = merged[0]["start"]
        merged[1]["start_frame"] = merged[0].get("start_frame")
        merged[1]["merged_runts"] = (merged[1].get("merged_runts", 0)
                                     + merged[0].get("merged_runts", 0) + 1)
        merged.pop(0)

    if not max_len:
        return merged

    out = []
    for s in merged:
        dur = s["end"] - s["start"]
        if dur <= max_len + 1e-6:
            out.append(s)
            continue
        # Equal parts, not max_len-sized parts with a remainder: 25s at
        # --max-len 10 becomes 3x8.3s rather than 10 + 10 + a 5s offcut.
        n = math.ceil(dur / max_len)
        step = dur / n
        for k in range(n):
            piece = dict(s)
            piece["start"] = s["start"] + k * step
            piece["end"] = s["start"] + (k + 1) * step if k < n - 1 else s["end"]
            piece["subdivided"] = True
            piece["part"] = k + 1
            piece["parts"] = n
            # Only the last piece ends on the real cut; the interior joins are
            # arbitrary and must not claim a detector's confidence.
            piece["cut_confidence"] = s["cut_confidence"] if k == n - 1 else None
            out.append(piece)
    return out


# ---------------------------------------------------------------------------
# Cutting
# ---------------------------------------------------------------------------

def cut_clip(src: Path, dst: Path, start: float, end: float, args,
             want_audio: bool):
    """[start, end] of src -> dst.mp4, video and audio together."""
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{end - start:.3f}",
           "-map", "0:v:0", "-map", "0:a?",
           "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
           "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", args.audio_bitrate,
           # +faststart moves the moov atom to the front. Costs a rewrite pass
           # on a file this small and makes every downstream reader able to
           # start decoding without seeking to the end.
           "-movflags", "+faststart",
           str(dst)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg could not cut [{start:.2f},{end:.2f}]: "
                           f"{p.stderr.strip()}")

    # VERIFY, DO NOT ASSUME. ffmpeg exits 0 on a window past the end of the
    # stream and writes a valid container with nothing in it. Downstream that
    # is a segfault in a native audio decoder (see QwenOmniAnnotator._cut in
    # 26_describe_segments_hf.py), i.e. a crash a long way from its cause.
    got = probe(dst)
    if not got["duration"] or got["duration"] < 0.05:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"cut [{start:.2f},{end:.2f}] came out empty "
                           f"(duration={got['duration']})")
    if want_audio and not got["has_audio"]:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"cut [{start:.2f},{end:.2f}] lost its audio track; "
                           f"the source has one")
    return got


def extract_wav(clip: Path, dst: Path):
    """16kHz mono PCM next to the clip -- what WhisperX and pyannote both want,
    and neither of them should be re-demuxing the mp4 to get it."""
    p = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(clip), "-vn",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"wav extraction failed: {p.stderr.strip()}")


def extract_thumb(clip: Path, dst: Path, at: float):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.3f}", "-i", str(clip),
         "-frames:v", "1", "-q:v", "3", str(dst)],
        capture_output=True, text=True)


# ---------------------------------------------------------------------------

def resolve_videos(args) -> list:
    """--only ids or paths -> [Path]. An entry that is an existing file is used
    as-is, so a one-off local mp4 needs no --videos gymnastics."""
    if args.only:
        out = []
        for tok in [t.strip() for t in args.only.split(",") if t.strip()]:
            p = Path(tok)
            if p.is_file():
                out.append(p)
                continue
            cand = args.videos / f"{tok}.mp4"
            if not cand.is_file():
                raise SystemExit(f"--only: no video for {tok!r} "
                                 f"(looked for {cand})")
            out.append(cand)
        return out

    clips = sorted(args.videos.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"no .mp4 under {args.videos}")
    return clips[:args.limit] if args.limit else clips


def split_dirname(index: int, total: int, pad: int) -> str:
    """split_07 for a 42-split video, split_007 for a 300-split one.

    Width from the TOTAL, as asked. Sorting stays lexicographic within a video,
    which is what `ls` and a glob both rely on; it is deliberately NOT constant
    across videos, so --pad 3 is there when you want to concatenate two videos'
    splits into one flat list.
    """
    width = pad if pad else max(2, len(str(total)))
    return f"split_{index:0{width}d}"


def process(clip: Path, args, detector) -> dict:
    vid = clip.stem
    detector_name = args.detector
    outdir = args.outdir / vid
    print(f"\n=== {vid} ===", flush=True)

    meta = probe(clip)
    fps, duration = meta["fps"], meta["duration"]
    print(f"    {meta['width']}x{meta['height']} {float(fps):.3f}fps "
          f"{duration:.1f}s audio={'yes' if meta['has_audio'] else 'NO'}",
          flush=True)

    # ---- boundaries -------------------------------------------------------
    reuse = None
    if args.scenes_json:
        reuse = (outdir / "scenes.json" if str(args.scenes_json) == "auto"
                 else Path(args.scenes_json))

    preds = None
    if reuse is not None:
        if not reuse.is_file():
            raise SystemExit(f"--scenes-json: {reuse} does not exist")
        saved = json.loads(reuse.read_text())
        scenes_sec = saved["scenes"]
        detector_name = saved.get("detector", detector_name) + "+reused"
        print(f"    reusing {len(scenes_sec)} scenes from {reuse}", flush=True)
    else:
        if detector_name == "transnetv2":
            frames = decode_small(clip, fps)
            print(f"    decoded {len(frames)} frames at {NET_W}x{NET_H}",
                  flush=True)
            scene_frames, preds = detector.scenes(frames)
        else:
            # Per video, unlike TransNetV2: PySceneDetect's detector object
            # carries per-video state (its rolling window), so reusing one
            # across videos would carry the tail of video A into video B.
            scene_frames, preds = PySceneDetectDetector(
                clip, fps, args.adaptive_threshold).scenes(None)
        scenes_sec = frames_to_seconds(scene_frames, fps, preds, duration)
        print(f"    {len(scenes_sec)} raw scenes", flush=True)

    # ---- shape them into the splits we want -------------------------------
    splits = enforce_lengths(scenes_sec, args.min_len, args.max_len)

    # Weak cuts are DROPPED BY MERGING, never by deleting the clip: deleting
    # would put a hole in the coverage of the video, and coverage is the one
    # property that makes these splits usable as a dataset.
    # Gated on the flag alone, NOT on `preds`: the confidence lives in the
    # scene dicts, so this works just as well re-cutting from a scenes.json
    # written by an earlier run. A detector with no scores (pyscenedetect)
    # leaves every cut_confidence null and merges nothing, which the null
    # checks below already say.
    if args.min_confidence:
        fused = []
        for s in splits:
            c = s.get("cut_confidence")
            # The merge decision depends ONLY on the cut BETWEEN fused[-1] and
            # s -- which is fused[-1]'s cut_confidence. s's own ending
            # confidence (c) must not gate it: the LAST scene of every video
            # has c=None (there is no cut after it), and requiring c would
            # make a weak final cut the one weak cut that can never be merged.
            if (fused and not s.get("subdivided")
                    and fused[-1].get("cut_confidence") is not None
                    and fused[-1]["cut_confidence"] < args.min_confidence):
                fused[-1]["end"] = s["end"]
                fused[-1]["end_frame"] = s.get("end_frame")
                fused[-1]["cut_confidence"] = c
                fused[-1]["low_conf_merges"] = \
                    fused[-1].get("low_conf_merges", 0) + 1
            else:
                fused.append(s)
        if len(fused) != len(splits):
            print(f"    {len(splits) - len(fused)} cuts below "
                  f"{args.min_confidence} merged away", flush=True)
        splits = fused

    total = len(splits)
    lengths = [s["end"] - s["start"] for s in splits]
    print(f"    {total} splits, {min(lengths):.1f}s..{max(lengths):.1f}s "
          f"(median {sorted(lengths)[total // 2]:.1f}s)", flush=True)

    if args.dry_run:
        for i, s in enumerate(splits, 1):
            print(f"      {split_dirname(i, total, args.pad)}  "
                  f"{s['start']:8.3f} -> {s['end']:8.3f}  "
                  f"{s['end'] - s['start']:6.2f}s  "
                  f"conf={s.get('cut_confidence')}", flush=True)
        return {"video_id": vid, "n_splits": total, "dry_run": True}

    # ---- write ------------------------------------------------------------
    outdir.mkdir(parents=True, exist_ok=True)
    record = {
        "video_id": vid,
        "source": str(clip),
        "detector": detector_name,
        "weights": str(args.weights) if detector_name.startswith("transnetv2")
                   else None,
        "threshold": args.threshold,
        "fps": f"{fps.numerator}/{fps.denominator}",
        "duration": duration,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "min_confidence": args.min_confidence,
        "crf": args.crf,
        "preset": args.preset,
        "n_splits": total,
        "scenes": splits,
    }
    (outdir / "scenes.json").write_text(json.dumps(record, indent=2))
    with (outdir / "scenes.csv").open("w") as f:
        f.write("split,start,end,duration,cut_confidence\n")
        for i, s in enumerate(splits, 1):
            f.write(f"{split_dirname(i, total, args.pad)},{s['start']:.3f},"
                    f"{s['end']:.3f},{s['end'] - s['start']:.3f},"
                    f"{'' if s.get('cut_confidence') is None else format(s['cut_confidence'], '.4f')}\n")

    if args.save_predictions and preds is not None:
        import numpy as np
        np.save(outdir / "predictions.npy", np.asarray(preds, dtype="float32"))

    wanted, done, failed = set(), 0, []
    for i, s in enumerate(splits, 1):
        name = split_dirname(i, total, args.pad)
        wanted.add(name)
        d = outdir / name
        clip_path = d / "clip.mp4"
        meta_path = d / "meta.json"

        if args.resume and clip_path.is_file() and meta_path.is_file():
            try:
                old = json.loads(meta_path.read_text())
                same = (abs(old["start"] - s["start"]) < 1e-3
                        and abs(old["end"] - s["end"]) < 1e-3
                        and old.get("crf") == args.crf)
            except (json.JSONDecodeError, KeyError, TypeError):
                same = False
            if same:
                done += 1
                continue

        d.mkdir(parents=True, exist_ok=True)
        # STALE SIDECARS. This split is being (re)cut, so any clip.wav or
        # thumb.jpg already in the directory belongs to the PREVIOUS
        # boundaries. Left alone -- --wav not passed this time, say -- the
        # directory ends up holding a 2s wav next to a 4s mp4, and nothing
        # downstream has any way to notice they disagree.
        for sidecar in ("clip.wav", "thumb.jpg"):
            (d / sidecar).unlink(missing_ok=True)
        try:
            got = cut_clip(clip, clip_path, s["start"], s["end"], args,
                           want_audio=meta["has_audio"])
        except RuntimeError as e:
            failed.append((name, str(e)))
            print(f"    FAIL {name}: {e}", flush=True)
            continue

        if args.wav and meta["has_audio"]:
            extract_wav(clip_path, d / "clip.wav")
        if args.thumbs:
            extract_thumb(clip_path, d / "thumb.jpg",
                          (s["end"] - s["start"]) / 2.0)

        meta_path.write_text(json.dumps({
            "video_id": vid,
            "split": name,
            "index": i,
            "n_splits": total,
            "source": str(clip),
            "start": s["start"],
            "end": s["end"],
            "duration": s["end"] - s["start"],
            "start_frame": s.get("start_frame"),
            "end_frame": s.get("end_frame"),
            "cut_confidence": s.get("cut_confidence"),
            "merged_runts": s.get("merged_runts", 0),
            "low_conf_merges": s.get("low_conf_merges", 0),
            "subdivided": bool(s.get("subdivided")),
            "part": s.get("part"),
            "parts": s.get("parts"),
            "detector": detector_name,
            "has_audio": bool(got["has_audio"]),
            "encoded_duration": got["duration"],
            "crf": args.crf,
            "preset": args.preset,
        }, indent=2))
        done += 1
        print(f"    {name}  {s['start']:8.3f} -> {s['end']:8.3f}  "
              f"{s['end'] - s['start']:6.2f}s"
              f"{'  [no audio]' if not got['has_audio'] else ''}", flush=True)

    # STALE SPLITS FROM A PREVIOUS RUN. Different --min-len, --max-len or
    # threshold means a different number of splits; the leftovers of the longer
    # run are indistinguishable from this run's output once you are looking at
    # the directory rather than the log.
    stale = sorted(p for p in outdir.glob("split_*")
                   if p.is_dir() and p.name not in wanted)
    if stale:
        if args.keep_stale:
            print(f"    NOTE: {len(stale)} stale split dirs left in place "
                  f"({stale[0].name}...) -- they are NOT from this run",
                  flush=True)
        else:
            for p in stale:
                shutil.rmtree(p, ignore_errors=True)
            print(f"    removed {len(stale)} stale split dirs from an earlier "
                  f"run ({stale[0].name}...)", flush=True)

    return {"video_id": vid, "n_splits": total, "written": done,
            "failed": failed, "outdir": str(outdir)}


def main():
    ap = argparse.ArgumentParser(
        description="Split a compilation video into its source clips "
                    "(video+audio) with TransNetV2 shot-boundary detection.")
    ap.add_argument("--videos", type=Path,
                    default=ROOT / "youtube_dataset/videos")
    ap.add_argument("--only", default=None,
                    help="comma-separated video id(s) or file path(s)")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = every video under --videos")
    ap.add_argument("--outdir", type=Path, default=ROOT / "outputs/scenes",
                    help="clips land in <outdir>/<video_id>/split_NN/")
    ap.add_argument("--detector", choices=["transnetv2", "pyscenedetect"],
                    default="transnetv2")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--device", default=None,
                    help="cuda / cpu (default: cuda when one is visible)")
    ap.add_argument("--batch", type=int, default=8,
                    help="100-frame windows per forward pass. Windows are "
                         "independent, so this is pure throughput.")
    # 0.5 is upstream's own default and what the published numbers use. Lower
    # it to catch soft dissolves at the cost of false cuts inside one clip.
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="TransNetV2: transition probability above which a "
                         "frame counts as a cut")
    # Separate flag, not a reused --threshold: 0.5 and 3.0 are not the same
    # quantity, and one knob meaning two things is how you end up running
    # AdaptiveDetector at a threshold tuned for a sigmoid.
    ap.add_argument("--adaptive-threshold", type=float, default=3.0,
                    help="pyscenedetect only: AdaptiveDetector threshold")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="merge across cuts whose peak score is below this. "
                         "0 = keep every cut the threshold found.")
    ap.add_argument("--min-len", type=float, default=1.0,
                    help="scenes shorter than this merge into their neighbour")
    ap.add_argument("--max-len", type=float, default=0.0,
                    help="0 = keep source clips whole. 10 = also chop anything "
                         "longer into equal pieces of at most 10s.")
    ap.add_argument("--pad", type=int, default=0,
                    help="digits in split_NN (0 = from the video's split count)")
    ap.add_argument("--crf", type=int, default=18,
                    help="x264 quality, lower is better. 18 is near-visually-"
                         "lossless; these clips are model input, not a proxy.")
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--audio-bitrate", default="192k")
    ap.add_argument("--wav", action="store_true",
                    help="also write 16kHz mono clip.wav (WhisperX/pyannote)")
    ap.add_argument("--thumbs", action="store_true",
                    help="also write a mid-clip thumb.jpg per split")
    ap.add_argument("--save-predictions", action="store_true",
                    help="dump the per-frame transition curve as .npy")
    ap.add_argument("--scenes-json", default=None,
                    help="re-cut from an existing scenes.json instead of "
                         "detecting again ('auto' = the one in --outdir)")
    ap.add_argument("--resume", action="store_true",
                    help="keep clips whose recorded boundaries still match")
    ap.add_argument("--keep-stale", action="store_true",
                    help="do NOT delete split dirs left by an earlier run")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the split plan; write nothing")
    args = ap.parse_args()

    clips = resolve_videos(args)
    print(f"videos:   {len(clips)}")
    print(f"detector: {args.detector}")
    print(f"outdir:   {args.outdir}")

    # Loaded ONCE for the whole run, not per video: 195 videos would otherwise
    # pay the weight load and the CUDA context 195 times. Skipped entirely when
    # --scenes-json means no detection happens at all.
    detector = None
    if args.detector == "transnetv2" and not args.scenes_json:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("note:     no GPU -- TransNetV2 on CPU is ~30fps, so a "
                  "10-minute video takes ~8 minutes", flush=True)
        detector = TransNetV2Detector(args.weights, device, args.batch)
        detector.threshold = args.threshold

    results = []
    for clip in clips:
        try:
            results.append(process(clip, args, detector))
        except (RuntimeError, SystemExit) as e:
            print(f"    FAIL {clip.stem}: {e}", flush=True)
            results.append({"video_id": clip.stem, "error": str(e)})

    print("\n" + "-" * 60)
    for r in results:
        if r.get("error"):
            print(f"{r['video_id']:16s} FAILED: {r['error']}")
        elif r.get("dry_run"):
            print(f"{r['video_id']:16s} {r['n_splits']} splits (dry run)")
        else:
            bad = len(r.get("failed") or [])
            print(f"{r['video_id']:16s} {r['written']}/{r['n_splits']} splits"
                  f"{f', {bad} failed' if bad else ''} -> {r['outdir']}")


if __name__ == "__main__":
    main()
