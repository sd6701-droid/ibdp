#!/usr/bin/env python3
"""
Build the Chambers et al. labelled YouTube dataset from URL_labelled_dataset.csv.

    python scripts/39_fetch_chambers_youtube.py                 # list + download
    python scripts/39_fetch_chambers_youtube.py --list-only     # manifest only
    python scripts/39_fetch_chambers_youtube.py --clips         # also cut the
                                                                # labelled windows

MUST run on a BigPurple LOGIN node. Compute nodes have no outbound internet, so
yt-dlp stalls there with no useful error -- same reason 10_fetch_youtube.sh is
not an sbatch job. A 10s preflight below fails fast instead of hanging.

Resumable: yt-dlp records every completed download in downloaded.txt, so a
re-run picks up where it left off. Safe to kill and restart.

Layout, under datasets/chambers_infant_youtube/:
    URL_labelled_dataset.csv   the source of truth (tracked in git)
    videos/<id>.mp4            one file per unique video, NAMED BY YOUTUBE ID
    metadata/<id>.info.json    yt-dlp sidecars (+ subtitles when present)
    clips/<id>_<s>-<e>s.mp4    only with --clips: the labelled window, cut out
    manifest.tsv               one row per CSV label window, joined to the file
    downloaded.txt             yt-dlp's resume archive
    logs/                      per-run yt-dlp output

WHY NAMED BY ID: titles carry slashes, quotes and emoji, and the id is the only
stable key that joins the video back to its row in the CSV. The url in the
manifest is `https://www.youtube.com/watch?v=<id>`, so id -> url is a lookup,
not a parse.

TWO THINGS TO KNOW:
1. The CSV has 103 rows but only 101 unique videos -- two videos are labelled
   twice with different windows. Downloads are deduped by id; the manifest
   keeps every window.
2. YouTube videos go private or get deleted. Failures are expected and are
   NOT fatal: every id that did not land on disk is written to missing.txt
   with the row it came from, and the run exits 0. Re-run later to retry.

BEFORE YOU BELIEVE missing.txt: yt-dlp needs a JavaScript runtime to solve
YouTube's player challenge. Without one it falls back to the `android_vr`
client, which reports `Video unavailable` for videos that play fine in a
browser -- so a missing runtime looks identical to a dead dataset. Install
deno once (login node, it needs internet) and the false failures go away:

    curl -fsSL https://deno.land/install.sh | sh
    export PATH="$HOME/.deno/bin:$PATH"     # add to ~/.bashrc to keep it
    pip install -U "yt-dlp[default]"        # + curl_cffi, for impersonation

The script warns loudly when no runtime is on PATH. If videos still fail
after that, try --player-client (e.g. 'default,web_safari') and then
--cookies; only what survives all three is genuinely gone.
"""
import argparse
import csv
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")
DEST = "datasets/chambers_infant_youtube"

# The 11-char YouTube id, wherever it sits in a watch/shorts/youtu.be url.
YT_ID = re.compile(r"(?:v=|/shorts/|youtu\.be/|/embed/)([0-9A-Za-z_-]{11})")


def resolve_id(url: str) -> str | None:
    """Pull the 11-char id out of a YouTube url. None if there isn't one."""
    m = YT_ID.search(url.strip())
    return m.group(1) if m else None


def to_seconds(minutes, seconds) -> float:
    """The CSV splits every timestamp into a float minute + a float second
    column (`0.0, 44.0` == 44s; `2.0, 58.0` == 178s)."""
    return float(minutes or 0) * 60.0 + float(seconds or 0)


def read_labels(csv_path: Path) -> list[dict]:
    """CSV rows -> label windows. One dict per row, id resolved, times in
    seconds. Rows whose url has no parseable id are reported and dropped."""
    labels, bad = [], []
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            url = (row.get("url") or "").strip()
            vid = resolve_id(url)
            if not vid:
                bad.append(url)
                continue
            labels.append({
                "video_number": (row.get("video_number") or "").strip(),
                "video_id": vid,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "start_sec": to_seconds(row.get("start_min"), row.get("start_sec")),
                "end_sec": to_seconds(row.get("end_min"), row.get("end_sec")),
            })
    if bad:
        print(f"WARNING: {len(bad)} row(s) had no YouTube id and were skipped:",
              file=sys.stderr)
        for url in bad:
            print(f"  {url!r}", file=sys.stderr)
    return labels


def have_network() -> bool:
    """Refuse to run somewhere with no route to YouTube rather than hang."""
    try:
        urllib.request.urlopen("https://www.youtube.com", timeout=10).close()
        return True
    except Exception:
        return False


def js_runtime() -> str | None:
    """The JS runtime yt-dlp will pick up off PATH, if any.

    Modern yt-dlp needs one to solve YouTube's player challenge. WITHOUT IT it
    quietly falls back to the `android_vr` client, which reports
    `ERROR: <id>: Video unavailable` for videos that play fine in a browser --
    so a missing runtime looks exactly like a dead dataset. Hence the loud
    warning at the call site rather than a silent degrade."""
    for runtime in ("deno", "node", "bun"):
        if shutil.which(runtime):
            return runtime
    return None


def download(ids: list[str], dest: Path, log: Path, args) -> int:
    """Hand every id to yt-dlp in one batch. Returns its exit code; non-zero is
    normal here because --ignore-errors keeps going past dead videos."""
    urls = dest / "urls.txt"
    urls.write_text("".join(f"https://www.youtube.com/watch?v={i}\n" for i in ids))
    cmd = [
        "yt-dlp",
        "--batch-file", str(urls),
        "--download-archive", str(dest / "downloaded.txt"),
        "--paths", f"home:{dest / 'videos'}",
        "--paths", f"infojson:{dest / 'metadata'}",
        "--paths", f"subtitle:{dest / 'metadata'}",
        "--output", "%(id)s.%(ext)s",
        # Cap at 1080p and prefer mp4/m4a so every file muxes to one container.
        "--format", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
                    "best[height<=1080]/best",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-subs", "--write-auto-subs", "--sub-langs", "en.*",
        "--sub-format", "vtt",
        "--concurrent-fragments", "4",
        "--retries", "10", "--fragment-retries", "10",
        # Politeness: 101 videos back to back off one IP invites throttling.
        "--sleep-requests", "1", "--sleep-interval", "2", "--max-sleep-interval", "6",
        "--ignore-errors",
        # --force re-fetches on purpose, so it must also lift yt-dlp's own
        # refusal to clobber -- otherwise the download is a silent no-op.
        "--force-overwrites" if args.force else "--no-overwrites",
        "--progress",
    ]
    if args.rate_limit:
        cmd += ["--limit-rate", args.rate_limit]
    if args.cookies:
        # Exported browser cookies. The single most effective answer to both
        # "Sign in to confirm you're not a bot" and age/region gates from a
        # datacenter IP -- which is what a BigPurple login node looks like.
        cmd += ["--cookies", str(args.cookies)]
    if args.player_client:
        cmd += ["--extractor-args", f"youtube:player_client={args.player_client}"]
    if args.js_runtime:
        cmd += ["--js-runtimes", args.js_runtime]
    if args.slow:
        # Answer to `HTTP Error 403: Forbidden` on the media fetch: that is
        # YouTube throttling a datacenter IP that asked for too much too fast,
        # not a broken video. One fragment at a time and longer gaps trade
        # wall-clock for a much higher completion rate. Appended last so these
        # win over the defaults set above.
        cmd += ["--concurrent-fragments", "1",
                "--sleep-requests", "3",
                "--sleep-interval", "5", "--max-sleep-interval", "15",
                "--retry-sleep", "http:exp=2:60"]
    cmd += args.ytdlp_arg
    print(f"downloading {len(ids)} video(s) -> {dest / 'videos'}", flush=True)
    with log.open("a") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        return proc.wait()


def video_path(dest: Path, vid: str) -> Path | None:
    """The downloaded file for an id, whatever extension it landed with."""
    for path in sorted((dest / "videos").glob(f"{vid}.*")):
        if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
            return path
    return None


def cut_clip(src: Path, start: float, end: float, dst: Path) -> bool:
    """The labelled window of src -> dst. Re-encodes rather than -c copy so the
    cut is frame-accurate instead of snapping to the nearest keyframe -- the
    windows here are as short as 10s, where a keyframe snap is a large fraction
    of the clip."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-i", str(src)]
    if end > start:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", str(dst)]
    return subprocess.run(cmd).returncode == 0


def write_manifest(path: Path, labels: list[dict], dest: Path):
    """One row per label window, joined to whatever is on disk."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["video_number", "video_id", "url", "start_sec",
                         "end_sec", "downloaded", "path"])
        for lab in labels:
            found = video_path(dest, lab["video_id"])
            writer.writerow([
                lab["video_number"], lab["video_id"], lab["url"],
                f"{lab['start_sec']:.1f}", f"{lab['end_sec']:.1f}",
                "yes" if found else "no",
                str(found.relative_to(dest)) if found else "",
            ])


def main():
    ap = argparse.ArgumentParser(
        description="Download the Chambers et al. labelled YouTube videos.")
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="project root (default: the gpfs checkout)")
    ap.add_argument("--dest", type=Path, default=None,
                    help=f"dataset dir (default: <root>/{DEST})")
    ap.add_argument("--csv", type=Path, default=None,
                    help="URL_labelled_dataset.csv (default: <dest>/ then <root>/)")
    ap.add_argument("--list-only", action="store_true",
                    help="write the manifest, download nothing")
    ap.add_argument("--clips", action="store_true",
                    help="also cut each labelled window into clips/")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N unique videos (for a smoke test)")
    ap.add_argument("--rate-limit", default=None,
                    help="yt-dlp --limit-rate, e.g. 5M")
    ap.add_argument("--cookies", type=Path, default=None,
                    help="exported cookies.txt -- the fix for 'Sign in to "
                         "confirm you're not a bot' and age-gated videos")
    ap.add_argument("--player-client", default=None,
                    help="youtube:player_client extractor arg, e.g. "
                         "'default,web_safari' or 'tv'. Try this when videos "
                         "report 'Video unavailable' but play in a browser.")
    ap.add_argument("--js-runtime", default=None,
                    help="yt-dlp --js-runtimes value, e.g. 'deno:/path/to/deno'. "
                         "Only needed when the runtime is NOT on PATH.")
    ap.add_argument("--slow", action="store_true",
                    help="one fragment at a time + longer sleeps. Use when the "
                         "log shows 'HTTP Error 403: Forbidden' -- that is "
                         "throttling, not a dead video.")
    ap.add_argument("--ytdlp-arg", action="append", default=[], metavar="ARG",
                    help="pass an extra flag straight through to yt-dlp; "
                         "repeatable. MUST use = form, since the value itself "
                         "starts with a dash: --ytdlp-arg=--no-check-certificate")
    ap.add_argument("--force", action="store_true",
                    help="re-download everything, OVERWRITING videos already "
                         "on disk. Not needed to retry failures -- only "
                         "successes are archived, so a plain re-run already "
                         "retries whatever is in missing.txt.")
    args = ap.parse_args()

    dest = args.dest or (args.root / DEST)
    csv_path = args.csv
    if csv_path is None:
        for cand in (dest / "URL_labelled_dataset.csv",
                     args.root / "URL_labelled_dataset.csv"):
            if cand.is_file():
                csv_path = cand
                break
    if csv_path is None or not csv_path.is_file():
        raise SystemExit(
            f"no URL_labelled_dataset.csv found under {dest} or {args.root}\n"
            "pass --csv, or copy it from the Chambers et al. release.")

    for sub in ("videos", "metadata", "logs"):
        (dest / sub).mkdir(parents=True, exist_ok=True)

    labels = read_labels(csv_path)
    if not labels:
        raise SystemExit(f"no usable rows in {csv_path}")

    # Dedup for downloading; the manifest still carries every window.
    ids = list(dict.fromkeys(lab["video_id"] for lab in labels))
    if args.limit:
        ids = ids[:args.limit]
        labels = [lab for lab in labels if lab["video_id"] in set(ids)]
    print(f"{csv_path.name}: {len(labels)} label window(s) over "
          f"{len(ids)} unique video(s)", flush=True)

    manifest = dest / "manifest.tsv"
    if args.list_only:
        write_manifest(manifest, labels, dest)
        print(f"manifest -> {manifest}")
        return

    if not shutil.which("yt-dlp"):
        raise SystemExit("yt-dlp not found. pip install -U yt-dlp")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found. Above 360p YouTube serves video and "
                         "audio separately and yt-dlp shells out to ffmpeg to "
                         "mux them. conda install -c conda-forge ffmpeg")
    if not have_network():
        raise SystemExit("no route to youtube.com -- you are probably on a "
                         "compute node. Log into a login node and run there "
                         "(tmux is your friend: this takes a while).")

    runtime = js_runtime()
    if runtime and not args.js_runtime and runtime != "deno":
        # yt-dlp auto-detects ONLY deno. node/bun exist but stay disabled until
        # named explicitly, so finding one on PATH is not enough -- without
        # this yt-dlp would degrade to android_vr while we report a runtime.
        args.js_runtime = runtime
    if runtime:
        print(f"js runtime: {runtime}"
              f"{' (passed via --js-runtimes)' if args.js_runtime else ''}",
              flush=True)
    elif not args.js_runtime:
        # Not fatal -- some videos still come through -- but it is by far the
        # most likely reason for a run that reports most of the dataset as
        # "Video unavailable", so say so before burning an hour on it.
        print("WARNING: no JavaScript runtime (deno/node/bun) on PATH.\n"
              "         yt-dlp falls back to the android_vr client, which\n"
              "         reports 'Video unavailable' for videos that are fine.\n"
              "         Install one, then re-run:\n"
              "           curl -fsSL https://deno.land/install.sh | sh\n"
              "           export PATH=\"$HOME/.deno/bin:$PATH\"\n"
              "         Also worth having, for the impersonation warning:\n"
              "           pip install -U 'yt-dlp[default]'\n", file=sys.stderr)

    if args.force:
        # Re-fetch EVERYTHING, including videos already on disk. Both skip
        # layers have to come off together: dropping the archive alone leaves
        # the `already` filter below to short-circuit the same files, which
        # makes the flag look like it worked while changing nothing.
        (dest / "downloaded.txt").unlink(missing_ok=True)

    already = set() if args.force else {i for i in ids if video_path(dest, i)}
    todo = [i for i in ids if i not in already]
    if args.force:
        print(f"--force: re-downloading all {len(todo)}, existing files "
              f"will be overwritten", flush=True)
    elif already:
        print(f"{len(already)} already on disk, {len(todo)} to fetch", flush=True)
    if todo:
        download(todo, dest, dest / "logs" / "download.log", args)

    write_manifest(manifest, labels, dest)

    got = {i for i in ids if video_path(dest, i)}
    missing = [i for i in ids if i not in got]
    missing_path = dest / "missing.txt"
    if missing:
        # Private / deleted / region-locked videos. Expected, not fatal.
        missing_path.write_text(
            "".join(f"{i}\thttps://www.youtube.com/watch?v={i}\n" for i in missing))
    elif missing_path.exists():
        missing_path.unlink()

    n_clips = 0
    if args.clips:
        print("cutting labelled windows -> clips/", flush=True)
        for lab in labels:
            src = video_path(dest, lab["video_id"])
            if not src:
                continue
            dst = (dest / "clips" /
                   f"{lab['video_id']}_{lab['start_sec']:.0f}-{lab['end_sec']:.0f}s.mp4")
            if dst.exists():
                n_clips += 1
                continue
            if cut_clip(src, lab["start_sec"], lab["end_sec"], dst):
                n_clips += 1
            else:
                print(f"  ffmpeg failed on {lab['video_id']}", file=sys.stderr)

    print(f"\ndone: {len(got)}/{len(ids)} videos in {dest / 'videos'}")
    print(f"  manifest : {manifest}")
    if args.clips:
        print(f"  clips    : {dest / 'clips'} ({n_clips}/{len(labels)})")
    if missing:
        print(f"  MISSING  : {len(missing)} video(s) -> {missing_path}")
        print("             (private/deleted/region-locked; re-run to retry)")


if __name__ == "__main__":
    main()
