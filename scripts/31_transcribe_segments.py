#!/usr/bin/env python
"""Transcribe each video's audio and attribute it to the SAME segments
26_describe_segments_hf.py annotates.

Output joins to the annotations on (video_id, segment_index), so a segment's
sound and its visual annotation sit side by side with no realignment.

WHY THE WHOLE TRACK IS TRANSCRIBED IN ONE PASS, not window by window:
a word straddling a segment boundary would be cut in half by per-window
decoding, and Whisper -- which pads everything to 30s anyway -- would burn 40
forward passes per video to produce that worse result. One pass with word-level
timestamps gives clean text AND exact attribution: each word lands in the
segment whose [start, end) contains it. Boundary words go to the segment
holding their MIDPOINT, so a word is never counted twice.

WHY WORDS ARE ALSO MAPPED TO FRAMES:
audio_windows gives per-frame bins, so each word carries the index of the frame
it was spoken over. That is what makes a claim like "the caregiver speaks while
the infant is prone" checkable rather than assumed.

The per-frame RMS is recorded too. It is cheap and it answers the question you
need answered before trusting any of this: a channel that lays a uniform music
bed over everything has flat RMS and a transcript full of song lyrics, and no
amount of downstream modelling will rescue that.

Usage:
    python scripts/31_transcribe_segments.py --only VIDEOID --limit 1   # measure
    python scripts/31_transcribe_segments.py                            # the rest

The Whisper checkpoint must be local -- the batch env sets HF_HUB_OFFLINE=1.
Fetch it on a login node first:
    scripts/13_fetch_models.sh --only whisper-large-v3
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from audio_windows import AudioWindows, NoAudioTrack, window_geometry

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")
DEFAULT_MODEL = ROOT / "models/whisper-large-v3"


def video_stream_limit(clip: Path) -> float | None:
    """Decodable length of the VIDEO stream, in seconds.

    MUST come from the video stream, not the audio one. 26's InternVL backend
    clamps its window against the video stream (end_stream_seconds); an mp4's
    audio and video streams routinely end a few ms apart, and clamping the
    audio against its own length yields a different `step`. The bins then walk
    away from the frames across the window -- silently, and only on the last
    segment of each video, which is the hardest place to notice it.

    None if ffprobe cannot say, in which case no clamp is applied and the
    window geometry matches the Qwen backend (which does not clamp either --
    qwen_vl_utils handles it internally).
    """
    for entries in ("stream=duration", "format=duration"):
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", entries, "-of", "csv=p=0", str(clip)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        val = p.stdout.strip().split("\n")[0].strip()
        if p.returncode == 0 and val and val != "N/A":
            try:
                return float(val)
            except ValueError:
                pass
    return None


def hhmmss(t: float) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"


def load_manifest(path: Path) -> dict:
    """Same 4-column TSV 10_fetch_youtube.sh writes."""
    rows = {}
    with path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            vid, title, dur, url = parts[0], parts[1], parts[2], parts[3]
            try:
                duration = float(dur)
            except ValueError:
                duration = None
            rows[vid] = {"title": title, "duration": duration, "url": url}
    return rows


def segments(duration: float, window: float, min_tail: float):
    """VERBATIM from 26_describe_segments_hf.py. Duplicated deliberately: this
    script must produce the same boundaries even if someone edits 26 without
    thinking about the join, and a silent boundary drift between the two files
    would corrupt every downstream comparison with no error to announce it.
    If you change one, change both -- and rerun both."""
    out, start = [], 0.0
    while start < duration:
        end = min(start + window, duration)
        if duration - end < min_tail and end < duration:
            end = duration
        out.append((start, end))
        start = end
    return out


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------
class Transcriber:
    def __init__(self, model_dir: Path, args):
        from transformers import (AutoModelForSpeechSeq2Seq, AutoProcessor,
                                  pipeline)

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(model_dir), torch_dtype=dtype, low_cpu_mem_usage=True)
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        proc = AutoProcessor.from_pretrained(str(model_dir))

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=proc.tokenizer,
            feature_extractor=proc.feature_extractor,
            torch_dtype=dtype,
            device=0 if torch.cuda.is_available() else -1,
            chunk_length_s=30,
            # Word timestamps are the whole point -- segment-level ones are too
            # coarse to attribute to a 10s window, let alone a frame.
            return_timestamps="word",
        )
        # Pinned, not auto-detected. Language ID on a few seconds of nursery
        # noise is unreliable, and a misdetect makes Whisper TRANSLATE -- which
        # produces fluent English that never appeared in the audio.
        self.gen = {"language": args.language, "task": "transcribe"}

    def words(self, wav: np.ndarray) -> list[dict]:
        with torch.inference_mode():
            out = self.pipe(wav, generate_kwargs=self.gen)

        words = []
        for ch in out.get("chunks") or []:
            ts = ch.get("timestamp") or (None, None)
            a, b = ts[0], ts[1]
            if a is None:
                continue            # nothing to attribute it to; drop it
            if b is None:
                b = a               # Whisper leaves the final word open-ended
            text = str(ch.get("text", "")).strip()
            if text:
                words.append({"w": text, "start": float(a), "end": float(b)})
        return words


# ---------------------------------------------------------------------------
def attribute(words, start, end, n, step):
    """Words whose MIDPOINT falls in [start, end), tagged with their frame.

    Midpoint, not overlap: a word overlapping two segments would otherwise be
    recorded in both, and the transcripts would not sum back to the original.
    """
    out = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        if not (start <= mid < end):
            continue
        frame = int((mid - start) / step)
        out.append({**w, "frame": max(0, min(n - 1, frame))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--videos", type=Path, default=ROOT / "youtube_dataset/videos")
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "youtube_dataset/manifest.tsv")
    ap.add_argument("--outdir", type=Path, default=ROOT / "outputs/transcripts")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated video id(s); process only these.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all videos")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--language", default="en")

    # These four MUST match the annotation run you intend to join against.
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--min-tail", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=64)

    args = ap.parse_args()

    if not args.model.is_dir():
        raise SystemExit(
            f"no checkpoint at {args.model}\n"
            f"  fetch it on a LOGIN node: scripts/13_fetch_models.sh "
            f"--only whisper-large-v3")

    model_tag = args.model_tag or args.model.name
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        # Max-plus-one, same reasoning as 26: file COUNT plus one collides with
        # an existing file the moment anyone deletes a middle one.
        pat = re.compile(rf"transcripts_{re.escape(model_tag)}_(\d{{3}})\.jsonl$")
        nums = [int(m.group(1))
                for f in args.outdir.glob(f"transcripts_{model_tag}_*.jsonl")
                if (m := pat.match(f.name))]
        run_no = (max(nums) + 1) if nums else 1
        args.out = args.outdir / f"transcripts_{model_tag}_{run_no:03d}.jsonl"

    meta = load_manifest(args.manifest)
    clips = sorted(args.videos.glob("*.mp4"))
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        clips = [c for c in clips if c.stem in wanted]
        if missing := wanted - {c.stem for c in clips}:
            raise SystemExit(f"--only: no .mp4 for {sorted(missing)}")
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"no .mp4 under {args.videos}")

    # Resume is per VIDEO, not per segment: the decode is one pass over the
    # whole track, so a half-done video costs nothing extra to redo and
    # tracking it per segment would just invite a partial-track join.
    done = set()
    if args.resume:
        for prev in sorted(args.outdir.glob("transcripts_*.jsonl")):
            with prev.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("model") == model_tag and r.get("complete"):
                        done.add(r["video_id"])

    todo = [c for c in clips if c.stem not in done]
    print(f"manifest: {len(meta)} videos", flush=True)
    print(f"model:    {model_tag}", flush=True)
    print(f"writing:  {args.out}", flush=True)
    print(f"{len(todo)} videos to transcribe ({len(done)} already done)", flush=True)
    if not todo:
        return

    tr = Transcriber(args.model, args)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"ready on {dev}\n", flush=True)

    t_all = time.time()
    with args.out.open("a") as fout:
        for k, clip in enumerate(todo, 1):
            vid = clip.stem
            info = meta.get(vid)
            if info is None or not info["duration"]:
                print(f"[{k}/{len(todo)}] SKIP {vid} (not in manifest / no duration)",
                      flush=True)
                continue

            t0 = time.time()
            try:
                aw = AudioWindows(clip)
                words = tr.words(aw.wav)
            except NoAudioTrack as e:
                # Not fatal and not silent: a silent video is a real finding
                # about the corpus, and the record says so explicitly rather
                # than looking like a video nobody ever spoke during.
                print(f"[{k}/{len(todo)}] NO AUDIO {vid}: {e}", flush=True)
                fout.write(json.dumps({
                    "model": model_tag, "video_id": vid, "audio_ok": False,
                    "complete": True, "segment_index": None}) + "\n")
                fout.flush()
                continue
            except Exception as e:
                print(f"[{k}/{len(todo)}] FAIL {vid}: {e}", flush=True)
                continue

            segs = segments(info["duration"], args.seconds, args.min_tail)
            vlimit = video_stream_limit(clip)
            for i, (s0, e0) in enumerate(segs):
                try:
                    start, end, n, step = window_geometry(
                        s0, e0, args.fps, args.max_frames, limit=vlimit)
                except ValueError:
                    continue        # window empty after clamping; 26 skips it too

                bins = aw.bins(start, end, n)
                rms = np.sqrt((bins ** 2).mean(axis=1))
                seg_words = attribute(words, start, end, n, step)
                text = " ".join(w["w"] for w in seg_words).strip()

                fout.write(json.dumps({
                    "model": model_tag,
                    "audio_ok": True,
                    "complete": i == len(segs) - 1,
                    "video_id": vid,
                    "video_name": info["title"],
                    "url": info["url"],
                    "url_at": f"{info['url']}&t={int(start)}s",
                    "segment_index": i,
                    "start_sec": round(start, 2),
                    "end_sec": round(end, 2),
                    "timestamp": f"{hhmmss(start)}-{hhmmss(end)}",
                    "transcript": text,
                    "n_words": len(seg_words),
                    "words": seg_words,
                    "n_frames": n,
                    # Per-frame, so it lines up with the VLM's frames one-to-one.
                    "frame_rms": [round(float(v), 5) for v in rms],
                    "rms_mean": round(float(rms.mean()), 5),
                    # Flat RMS across a whole video is the music-bed signature.
                    "rms_std": round(float(rms.std()), 5),
                }) + "\n")
            fout.flush()

            dt = time.time() - t0
            print(f"[{k}/{len(todo)}] {vid}  {len(segs)} segs  {len(words)} words  "
                  f"{dt:.1f}s  ({info['duration']:.0f}s audio)", flush=True)

    print(f"\ndone in {(time.time() - t_all) / 60:.1f}m -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
