"""Frame-aligned audio for the windows 26_describe_segments_hf.py annotates.

The video side samples a window at bin CENTRES: a 10s window at 2fps yields
frames at 0.25s, 0.75s, ... Each of those frames is the representative of a bin
[start + i*step, start + (i+1)*step). This module returns the audio of exactly
those bins, so bins(...)[i] is the sound during frames[i] -- same count, same
order, same boundaries.

WHY THE WHOLE TRACK IS DECODED AT ONCE, not per window:
Per-window `ffmpeg -ss` re-seeks for every segment, and input seeking lands on
the nearest decodable point rather than the exact sample. Over a 40-segment
video those roundings drift, and drift is precisely what "frame aligned" rules
out. Decoding once and slicing by sample INDEX makes alignment arithmetic
instead of an ffmpeg behaviour. It is also cheaper: one subprocess per video,
not one per segment. A 10-minute track at 16k mono float32 is ~38MB.

USAGE, mirroring InternVLAnnotator._frames:

    from audio_windows import AudioWindows, window_geometry

    aw = AudioWindows(clip)                      # decode once, reuse per video
    start, end, n, step = window_geometry(start, end, fps, max_frames, limit)
    bins = aw.bins(start, end, n)                # (n, samples_per_bin) float32

CAVEAT -- these bins are too short for speech recognition. At 2fps a bin is
0.5s; Whisper wants seconds of context and will hallucinate on half-second
fragments. Use bins() for per-frame audio FEATURES (event tagging, energy,
CLAP embeddings) and transcribe with window(start, end) or the full track,
attributing the resulting word timestamps back to frames.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000        # what Whisper, CLAP and the omni models all expect


def window_geometry(start: float, end: float, fps: float, max_frames: int,
                    limit: float | None = None):
    """The window arithmetic from InternVLAnnotator._frames, factored out so the
    audio cannot disagree with the video about where the bins are.

    Returns (start, end, n, step). `limit` is the decodable stream length; pass
    it to reproduce the clamp at 26_describe_segments_hf.py:394. Alignment is
    only exact if BOTH sides use the clamped bounds -- clamping the frames and
    not the audio is the one way to silently desync them.
    """
    if limit is not None:
        end = min(end, limit - 1e-3)
        start = min(start, end - 1e-3)
    if end <= start:
        raise ValueError(
            f"window [{start:.2f},{end:.2f}] is empty after clamping to the "
            f"stream ({limit}).")

    n = int(round((end - start) * fps))
    n = max(1, min(max_frames, n))
    step = (end - start) / n
    return start, end, n, step


def frame_stamps(start: float, n: int, step: float) -> list[float]:
    """Bin centres -- the timestamps _frames() hands to the video decoder."""
    return [start + (i + 0.5) * step for i in range(n)]


class AudioWindows:
    """One decoded audio track, sliced into frame-aligned bins on demand.

    Construct one per video and reuse it across that video's segments; the
    decode happens once, lazily, on first use.
    """

    def __init__(self, clip: Path | str, sr: int = SAMPLE_RATE):
        self.clip = Path(clip)
        self.sr = sr
        self._wav: np.ndarray | None = None

    # -- decode ------------------------------------------------------------
    @property
    def wav(self) -> np.ndarray:
        if self._wav is None:
            self._wav = self._decode()
        return self._wav

    def _decode(self) -> np.ndarray:
        """Full track -> mono float32 at self.sr.

        s16le to a pipe rather than a temp wav: no file to clean up, and no
        44-byte header to skip. -vn is the mirror of the -an in 30_infant_pose.
        """
        cmd = ["ffmpeg", "-loglevel", "error", "-i", str(self.clip),
               "-vn", "-ac", "1", "-ar", str(self.sr),
               "-f", "s16le", "pipe:1"]
        p = subprocess.run(cmd, capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(
                f"ffmpeg could not decode audio from {self.clip.name}: "
                f"{p.stderr.decode(errors='replace').strip()}")
        if not p.stdout:
            # A silent or audio-less mp4 is a data problem, not a crash: the
            # caller may legitimately want to annotate it on video alone.
            raise NoAudioTrack(f"{self.clip.name} has no decodable audio stream")
        return np.frombuffer(p.stdout, dtype="<i2").astype(np.float32) / 32768.0

    @property
    def duration(self) -> float:
        return len(self.wav) / self.sr

    # -- slicing -----------------------------------------------------------
    def bins(self, start: float, end: float, n: int) -> np.ndarray:
        """(n, samples_per_bin) float32 -- row i is the audio under frame i.

        Bin i is anchored at its exact start time, round(start_i * sr), and all
        bins share one width so the result is rectangular. Anchoring per bin
        rather than striding from the first keeps the error at the bin edge
        below one sample (62.5us at 16k) instead of accumulating across n.

        Past the end of the track the tail is zero-padded, which is what the
        final window of a video hits when the container duration overshoots the
        real stream -- the same overshoot the video side clamps for.
        """
        step = (end - start) / n
        width = int(round(step * self.sr))
        if width < 1:
            raise ValueError(
                f"bin of {step * 1000:.1f}ms is under one sample at {self.sr}Hz")

        wav = self.wav
        out = np.zeros((n, width), dtype=np.float32)
        for i in range(n):
            a = int(round((start + i * step) * self.sr))
            if a >= len(wav):
                break                       # rest stays zero-padded
            chunk = wav[a:a + width]
            out[i, :len(chunk)] = chunk
        return out

    def window(self, start: float, end: float) -> np.ndarray:
        """The whole window as one waveform -- for transcription, where the
        per-frame bins are too short to be meaningful."""
        a = int(round(start * self.sr))
        b = int(round(end * self.sr))
        return self.wav[max(0, a):max(0, b)]


class NoAudioTrack(RuntimeError):
    """Raised when a clip carries no decodable audio."""


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Verify audio bins line up with the video frames.")
    ap.add_argument("clip", type=Path)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=64)
    a = ap.parse_args()

    aw = AudioWindows(a.clip)
    start, end, n, step = window_geometry(a.start, a.end, a.fps, a.max_frames)
    b = aw.bins(start, end, n)

    print(f"track      : {aw.duration:.2f}s @ {aw.sr}Hz")
    print(f"window     : [{start:.3f}, {end:.3f}]  n={n}  step={step:.3f}s")
    print(f"bins       : {b.shape}  ({b.shape[1] / aw.sr:.3f}s each)")
    print(f"frame stamp / bin span / rms")
    for i, t in enumerate(frame_stamps(start, n, step)):
        lo = start + i * step
        print(f"  frame{i + 1:>3}  t={t:7.3f}  "
              f"[{lo:7.3f},{lo + step:7.3f}]  rms={np.sqrt((b[i] ** 2).mean()):.4f}")
