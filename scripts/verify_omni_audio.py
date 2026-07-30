#!/usr/bin/env python
"""Why has an Omni run no audio fields? Separates the three possible causes.

  1. THE CLIPS ARE SILENT     -- ffprobe finds no audio stream, or a stream
                                 whose samples are all zero. A data problem;
                                 nothing downstream can fix it.
  2. THE MODEL WAS NEVER ASKED -- records have no "audio_events" KEY at all.
                                 native_audio was False, so AUDIO_PROMPT was
                                 never appended: backend misdetected, or the
                                 cluster is running older code.
  3. THE MODEL ANSWERED EMPTY  -- the key is present but [] / "". It was asked,
                                 it heard, and it reported nothing.

Absent key and empty value look identical in a quick eyeball and mean opposite
things, which is the whole reason this script exists.

    python scripts/verify_omni_audio.py --video 8A3wVWWfiMY
    python scripts/verify_omni_audio.py --video 8A3wVWWfiMY --scenes
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")

AUDIO_KEYS = ["audio_events", "infant_vocalising", "speech_present",
              "audio_description"]


def probe_audio(clip: Path) -> dict:
    """Audio stream present? And is it actually non-silent?"""
    q = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,sample_rate,duration",
         "-of", "default=nw=1", str(clip)],
        capture_output=True, text=True)
    has = bool(q.stdout.strip())
    out = {"clip": clip, "has_audio": has, "codec": None, "peak": None}
    if not has:
        return out
    for line in q.stdout.strip().splitlines():
        if line.startswith("codec_name="):
            out["codec"] = line.split("=", 1)[1]

    # A track of pure zeroes passes every "has audio" check and is still
    # silence. volumedetect reports the real peak.
    v = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(clip), "-af", "volumedetect",
         "-f", "null", "-"],
        capture_output=True, text=True)
    for line in v.stderr.splitlines():
        if "max_volume" in line:
            out["peak"] = line.split("max_volume:")[1].strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", type=Path, default=ROOT / "outputs")
    ap.add_argument("--scenes", action="store_true",
                    help="check scene splits instead of the source mp4")
    ap.add_argument("--videos", type=Path,
                    default=ROOT / "youtube_dataset/videos")
    ap.add_argument("--scenes-dir", type=Path, default=ROOT / "outputs/scenes")
    ap.add_argument("--n", type=int, default=5, help="clips to probe")
    a = ap.parse_args()

    # ---- 1. do the clips carry sound? -------------------------------------
    print("=" * 72)
    print("1. AUDIO ON DISK")
    print("=" * 72)
    if a.scenes:
        clips = sorted((a.scenes_dir / a.video).glob("split_*/clip.mp4"))[:a.n]
        if not clips:
            print(f"  no splits under {a.scenes_dir / a.video}")
    else:
        clips = [a.videos / f"{a.video}.mp4"]

    for c in clips:
        if not c.exists():
            print(f"  MISSING {c}")
            continue
        r = probe_audio(c)
        tag = "OK " if r["has_audio"] else "NO AUDIO"
        print(f"  {tag} {c.parent.name}/{c.name}  codec={r['codec']}  "
              f"peak={r['peak']}")
        if r["peak"] and "-91" in str(r["peak"]):
            print("       ^ peak is the float32 noise floor: this track is "
                  "digital silence.")

    # ---- 2/3. what did the model record? ----------------------------------
    print()
    print("=" * 72)
    print("2/3. WHAT THE RECORDS SAY")
    print("=" * 72)
    files = sorted(a.outdir.glob("annotations_*Omni*.jsonl"))
    if not files:
        print(f"  no annotations_*Omni*.jsonl under {a.outdir}")
        return 1

    for f in files:
        rows = []
        for line in f.open():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("video_id") == a.video:
                rows.append(r)
        if not rows:
            continue

        ok = [r for r in rows if r.get("parse_ok")]
        print(f"\n{f.name}")
        print(f"  {len(ok)}/{len(rows)} parsed   "
              f"prompt_sha={rows[0].get('prompt_sha')}   "
              f"audio_mode={rows[0].get('audio_mode')}   "
              f"segmentation={rows[0].get('segmentation')}")

        # THE DECISIVE DISTINCTION: key missing vs key empty.
        for k in AUDIO_KEYS:
            present = sum(1 for r in ok if k in r)
            truthy = sum(1 for r in ok if r.get(k) not in (None, "", [], False))
            if present == 0:
                verdict = "KEY ABSENT -> the model was never asked (cause 2)"
            elif truthy == 0:
                verdict = "key present, all empty -> asked, answered nothing (cause 3)"
            else:
                verdict = f"{truthy}/{present} populated -> working"
            print(f"    {k:20s} present={present:3d}  {verdict}")

        for r in ok[:3]:
            print(f"\n    [{r.get('timestamp')}] events={r.get('audio_events', '<ABSENT>')}")
            print(f"      audio: {r.get('audio_description', '<ABSENT>')}")
            print(f"      video: {(r.get('description') or '')[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
