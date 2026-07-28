#!/usr/bin/env python
"""
Segment-wise STRUCTURED annotation of the YouTube dataset via transformers.

Each video is cut into fixed-length windows (default 10s) and each window is
annotated independently. Windows are NOT extracted to disk: qwen_vl_utils takes
video_start/video_end and decodes only the requested range out of the full mp4.

The model is asked for JSON, not prose. Fields per segment:

    has_infant          bool
    num_infants         int
    has_adult           bool
    num_adults          int
    num_humans_total    int     (infants + adults; the model's own total)
    infant_visibility   "full_body" | "partial_body" | "not_visible"
    visible_infant_parts  list of head/face/torso/arms/hands/legs/feet
    description         one or two sentences

WHY JSON AND NOT PROSE: a free-text "describe this video" prompt produced
100-200 word narrations for a 10-second clip, most of the ~12s/segment being
generation rather than decode -- and it degenerated into repetition loops under
greedy decoding (one segment repeated "the baby is still splashing" until it hit
the token cap). Structured output is shorter, faster, parseable, and cannot
ramble.

Resume is per (MODEL, SEGMENT), so a walltime kill costs nothing. Rerun to
continue.

    python scripts/26_describe_segments_hf.py --limit 2    # measure first
    python scripts/26_describe_segments_hf.py              # then the rest

MULTI-MODEL: --model points at any local checkpoint. Two backends are supported
and picked automatically from the checkpoint:

    Qwen*-VL     transformers + qwen_vl_utils, which decodes [start,end] out of
                 the full mp4 for us.
    InternVL3*   trust_remote_code + model.chat(). qwen_vl_utils does NOT drive
                 it, so we sample the window's frames ourselves with torchcodec
                 and hand InternVL one 448px tile per frame.

Every record carries "model", the output filename carries the model tag, and
--resume only reuses records from the SAME model. Running three models over one
video therefore gives you three independent files:

    for M in Qwen2.5-VL-72B-Instruct InternVL3-78B InternVL3-38B; do
      python scripts/26_describe_segments_hf.py --model $ROOT/models/$M --only VIDEOID
    done

NOTE: torchcodec needs the CUDA 12 NPP libs on LD_LIBRARY_PATH. Per shell:
    SITE=$(python -c "import site; print(site.getsitepackages()[0])")
    export LD_LIBRARY_PATH="$SITE/nvidia/npp/lib:$SITE/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
"""
import argparse, hashlib, json, os, re, time
from pathlib import Path

# Must precede the qwen_vl_utils import. decord hangs on decode and is
# unmaintained; Qwen recommends torchcodec.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _preload_cuda_libs():
    """torchcodec's .so needs libnppicc.so.12 / libnvrtc, which live in the
    pip nvidia-*-cu12 packages but are NOT on the default loader path. Rather
    than require `export LD_LIBRARY_PATH=...` in every shell (which broke this
    run three times), dlopen them RTLD_GLOBAL here so their symbols are resolved
    when torchcodec loads. Silent if a lib is absent -- torchcodec then reports
    its own clear error."""
    import ctypes
    import glob
    import site
    roots = list(site.getsitepackages())
    if hasattr(site, "getusersitepackages"):
        roots.append(site.getusersitepackages())
    for sp in roots:
        for so in glob.glob(os.path.join(sp, "nvidia", "*", "lib", "*.so*")):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_cuda_libs()

import torch

ROOT = Path("/gpfs/scratch/sd6701/personal/ibdp")
DEFAULT_MODEL = ROOT / "models/Qwen3-VL-30B-A3B-Instruct"

# Records written before this script grew a --model flag have no "model" field.
# They all came from the checkpoint below, so that is what they are credited to
# on resume -- otherwise every one of them would look like a foreign model's
# work and be regenerated.
LEGACY_MODEL_TAG = "Qwen3-VL-30B-A3B-Instruct"

# "infant" is defined explicitly. Left to itself the model drifts between
# "baby", "toddler" and "young child" across segments, which makes the counts
# useless to aggregate.
PROMPT = """Analyse this clip. Reply with JSON only.

Sort EVERY visible person into exactly one bucket; omit nobody:
- INFANT: cannot walk unaided (crawls, sits, lies, is carried).
- CHILD: walks unaided, not yet an adult.
- ADULT: grown person.

{
  "has_infant": bool,
  "num_infants": int,
  "num_children": int,
  "has_adult": bool,
  "num_adults": int,
  "num_humans_total": int,
  "infant_visibility": "full_body" | "partial_body" | "not_visible",
  "visible_infant_parts": ["head","face","torso","arms","hands","legs","feet"],
  "description": "at most 2 sentences"
}

- Count distinct people across the whole clip, not per frame.
- num_humans_total = num_infants + num_children + num_adults.
- full_body = whole infant head-to-feet seen at some point; partial_body = only
  part of them; not_visible = no infant, and then parts is [].
- description: max 2 sentences, concrete and specific -- who, where, what they
  physically do, what they touch. Name the infant's posture. Only what is
  visible; no mood, no padding, no repetition."""

VALID_VISIBILITY = {"full_body", "partial_body", "not_visible"}
VALID_PARTS = {"head", "face", "torso", "arms", "hands", "legs", "feet"}


# ---------------------------------------------------------------------------
# Backends
#
# Both expose the same one-method contract -- annotate(clip, start, end) -> raw
# model text -- so main()'s loop, the JSON parsing and the resume logic are
# identical no matter which checkpoint is loaded.
# ---------------------------------------------------------------------------

def detect_backend(model_dir: Path) -> str:
    """Checkpoint -> backend name. Reads config.json's architectures first,
    because that is authoritative; the directory name is only a fallback for a
    checkpoint that was renamed on download."""
    cfg = model_dir / "config.json"
    if cfg.is_file():
        try:
            arch = " ".join(json.loads(cfg.read_text()).get("architectures") or [])
        except (json.JSONDecodeError, OSError):
            arch = ""
        if "InternVL" in arch:
            return "internvl"
        if "Qwen" in arch and "VL" in arch:
            return "qwen"
    name = model_dir.name.lower()
    if "internvl" in name:
        return "internvl"
    if "qwen" in name and "vl" in name:
        return "qwen"
    raise SystemExit(
        f"cannot tell which backend {model_dir.name} needs. Pass --backend "
        f"qwen|internvl explicitly.")


def _device_map():
    """'auto' shards a 72B/78B across every visible card. Single-GPU stays on
    cuda:0 -- 'auto' there can still offload to CPU under memory pressure and
    turn a 12s segment into minutes with no error to explain why."""
    return "auto" if torch.cuda.device_count() > 1 else "cuda:0"


class QwenAnnotator:
    """Qwen*-VL. qwen_vl_utils decodes only [video_start, video_end] out of the
    full mp4, so nothing is ever extracted to disk."""

    def __init__(self, model_dir: Path, args):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._process_vision_info = process_vision_info
        self.args = args
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        print(f"loading weights from {model_dir.name} (off GPFS, minutes for a "
              f"72B)...", flush=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, device_map=_device_map())
        self.model.eval()

    def annotate(self, clip: Path, start: float, end: float) -> str:
        args = self.args
        msgs = [{
            "role": "user",
            "content": [
                # total_pixels is the real memory knob, not fps. If a clip
                # OOMs, lower total_pixels first -- you keep temporal
                # coverage and give up spatial detail, the better trade.
                {"type": "video", "video": str(clip), "fps": args.fps,
                 "video_start": start, "video_end": end,
                 "total_pixels": args.total_pixels_factor * 32 * 32},
                {"type": "text", "text": PROMPT},
            ],
        }]
        images, videos, video_kwargs = self._process_vision_info(
            msgs, return_video_kwargs=True)

        # fps comes back as a LIST (one per video in the batch) -- the
        # processor validates it as a scalar. It feeds MRoPE's temporal
        # positions, so use the ACTUAL sampled rate, not args.fps.
        fps = video_kwargs.get("fps")
        if isinstance(fps, (list, tuple)):
            fps = fps[0] if fps else args.fps

        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            videos=videos,
            fps=fps,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,          # deterministic: this is extraction
                repetition_penalty=1.05,  # greedy decoding looped without it
            )
        return self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True)[0].strip()


class InternVLAnnotator:
    """InternVL3. No qwen_vl_utils: we sample the window ourselves and feed one
    448px tile per frame, which is what InternVL's own video example does.

    Single tile per frame ON PURPOSE. InternVL's dynamic tiling can emit up to
    12 tiles for ONE image; at 20 frames per 10s window that is 240 tiles and a
    guaranteed OOM. Tiling buys spatial detail on a still image, which is not
    what a 10-second infant-motion clip is asking for."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    @staticmethod
    def _check_deps(model_dir: Path):
        """Fail here, legibly, rather than 40 frames into a traceback from
        inside transformers' dynamic module loader.

        InternVL's architecture is not part of transformers -- it is vendored in
        the checkpoint and loaded with trust_remote_code. That vendored code
        hard-imports einops and timm. If the .py files are missing the download
        was incomplete, which `ls` will not tell you."""
        needed = ["modeling_internvl_chat.py", "modeling_intern_vit.py",
                  "configuration_internvl_chat.py", "conversation.py"]
        missing = [f for f in needed if not (model_dir / f).is_file()]
        if missing:
            raise SystemExit(
                f"{model_dir.name} is missing its vendored architecture code: "
                f"{', '.join(missing)}\n"
                f"  Re-download: scripts/13_fetch_models.sh --only "
                f"internvl3-{'78b' if '78' in model_dir.name else '38b'}\n"
                f"  (rm {model_dir}/.complete first to force it)")

        # importlib.util, not importlib: the submodule is not bound by a bare
        # `import importlib`, so find_spec would raise AttributeError here.
        import importlib.util
        for mod, pip in (("einops", "einops"), ("timm", "timm>=0.9")):
            if importlib.util.find_spec(mod) is None:
                raise SystemExit(
                    f"InternVL's vendored code imports `{mod}`, which is not "
                    f"installed.\n  conda activate ibdp && pip install '{pip}'")

    def __init__(self, model_dir: Path, args):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoModel, AutoTokenizer

        self._check_deps(model_dir)
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True, use_fast=False)
        print(f"loading weights from {model_dir.name} (off GPFS, minutes for a "
              f"78B)...", flush=True)
        # trust_remote_code: InternVL ships its own modelling code, including
        # the .chat() helper used below. AutoModelForImageTextToText does not
        # cover it.
        self.model = AutoModel.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, trust_remote_code=True,
            low_cpu_mem_usage=True, device_map=_device_map()).eval()

        size = args.internvl_tile
        self.transform = T.Compose([
            T.Lambda(lambda im: im.convert("RGB")),
            T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def _frames(self, clip: Path, start: float, end: float):
        """Frames from [start, end], sampled at args.fps and hard-capped.
        Timestamps are bin CENTRES, so a 10s window at 2fps samples 0.25s,
        0.75s ... rather than clipping the boundary frame at exactly `end`,
        which is one frame past the window."""
        from torchcodec.decoders import VideoDecoder

        n = int(round((end - start) * self.args.fps))
        n = max(1, min(self.args.max_frames, n))
        step = (end - start) / n
        stamps = [start + (i + 0.5) * step for i in range(n)]

        dec = VideoDecoder(str(clip))
        batch = dec.get_frames_played_at(stamps)
        return batch.data          # (N, C, H, W) uint8

    def annotate(self, clip: Path, start: float, end: float) -> str:
        from PIL import Image

        frames = self._frames(clip, start, end)
        imgs = [Image.fromarray(f.permute(1, 2, 0).cpu().numpy()) for f in frames]

        pixel_values = torch.stack([self.transform(im) for im in imgs])
        pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)

        # InternVL wants one <image> placeholder per frame, numbered. Without
        # the Frame-N prefixes it treats the batch as unordered stills and
        # loses the temporal ordering the whole task depends on.
        prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(imgs)))
        question = prefix + PROMPT

        with torch.inference_mode():
            out = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                dict(max_new_tokens=self.args.max_new_tokens,
                     do_sample=False,
                     repetition_penalty=1.05),
                num_patches_list=[1] * len(imgs),
            )
        return str(out).strip()


BACKENDS = {"qwen": QwenAnnotator, "internvl": InternVLAnnotator}


def hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def load_manifest(path: Path) -> dict:
    """id -> {title, duration, url}. yt-dlp prints NA for missing durations."""
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
    """[(start, end)] covering [0, duration). A runt tail merges into the previous
    window rather than becoming a 0.4s clip the model can say nothing about."""
    out, start = [], 0.0
    while start < duration:
        end = min(start + window, duration)
        if duration - end < min_tail and end < duration:
            end = duration
        out.append((start, end))
        start = end
    return out


def parse_annotation(raw: str) -> dict:
    """Model text -> validated dict. Never raises: a segment that returns junk
    records parse_ok=false and keeps its raw text, rather than killing the run or
    silently writing zeros that look like real observations."""
    out = {"parse_ok": False, "raw": raw}

    # It is told not to fence the JSON, but it sometimes does anyway.
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return out
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out

    def as_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1")
        return bool(v)

    def as_int(v):
        try:
            return max(0, int(v))
        except (TypeError, ValueError):
            return 0

    vis = str(d.get("infant_visibility", "")).strip().lower()
    if vis not in VALID_VISIBILITY:
        vis = None

    parts = d.get("visible_infant_parts") or []
    if not isinstance(parts, list):
        parts = []
    parts = [p for p in (str(x).strip().lower() for x in parts) if p in VALID_PARTS]

    ann = {
        "has_infant": as_bool(d.get("has_infant")),
        "num_infants": as_int(d.get("num_infants")),
        "num_children": as_int(d.get("num_children")),
        "has_adult": as_bool(d.get("has_adult")),
        "num_adults": as_int(d.get("num_adults")),
        "num_humans_total": as_int(d.get("num_humans_total")),
        "infant_visibility": vis,
        "visible_infant_parts": parts,
        "description": str(d.get("description", "")).strip(),
    }

    # Self-consistency. The model sometimes says has_infant=true with
    # num_infants=0, or a total that does not add up. Flag the disagreement --
    # do NOT quietly pick a winner, because a silently-corrected count is
    # indistinguishable from a real observation when you aggregate later.
    parts_sum = ann["num_infants"] + ann["num_children"] + ann["num_adults"]
    ann["inconsistent"] = bool(
        ann["has_infant"] != (ann["num_infants"] > 0)
        or ann["has_adult"] != (ann["num_adults"] > 0)
        or (not ann["has_infant"] and ann["visible_infant_parts"])
        or ann["num_humans_total"] != parts_sum
    )
    ann["parse_ok"] = True
    return ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                    help="local checkpoint dir under $ROOT/models")
    ap.add_argument("--backend", choices=sorted(BACKENDS),
                    help="override backend auto-detection")
    ap.add_argument("--model-tag", default=None,
                    help="short name used in the output filename and in every "
                         "record (default: the checkpoint directory name)")
    ap.add_argument("--videos", type=Path, default=ROOT / "youtube_dataset/videos")
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "youtube_dataset/manifest.tsv")
    # Default: a NEW timestamped file per run, so a run never mutates an older
    # one. Pass --out explicitly to override.
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=ROOT / "outputs")
    # Resume scans every annotations_*.jsonl in --outdir, but only reuses records
    # whose prompt hash matches the CURRENT prompt. Change the prompt and stale
    # records are ignored and regenerated, rather than silently kept.
    ap.add_argument("--resume", action="store_true",
                    help="skip segments already done under the same prompt")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all VIDEOS (not segments). Use 2 to measure speed.")
    ap.add_argument("--only", default=None,
                    help="comma-separated video id(s); process only these.")
    ap.add_argument("--seconds", type=float, default=10.0, help="window length")
    ap.add_argument("--min-tail", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=2.0)
    # 256: the structured fields are tiny and `description` is capped at two
    # sentences, so this is ample. It must stay comfortably ABOVE the real
    # output length -- a description truncated mid-sentence yields JSON with no
    # closing brace, parse_ok goes false, and the whole segment is lost, counts
    # included. Generation dominates per-segment cost, so this is also the
    # cheapest lever on total runtime.
    ap.add_argument("--max-new-tokens", type=int, default=256)
    # Qwen only: whole-clip vision-token budget, ~= total_pixels / (32*32).
    ap.add_argument("--total-pixels-factor", type=int, default=20480)
    # InternVL only. 32 frames x 448px is already ~8k vision tokens; the cap
    # stops a long tail window from quietly becoming an OOM.
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--internvl-tile", type=int, default=448)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU -- needs an A100 allocation, not a login node.")

    if not args.model.is_dir():
        raise SystemExit(f"no checkpoint at {args.model}")
    model_tag = args.model_tag or args.model.name
    backend = args.backend or detect_backend(args.model)

    # Stamped into every record. Two runs with different prompts are then
    # distinguishable after the fact, and --resume can tell fresh from stale.
    prompt_sha = hashlib.sha256(PROMPT.encode()).hexdigest()[:8]

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        # Sequential counter: highest existing annotations_NNN_*.jsonl + 1.
        # NOT "file count + 1" -- deleting a middle file would then make the
        # next run collide with an existing higher number and append into it.
        # Max-plus-one is always fresh.
        # Strict pattern: exactly 3 digits then the 8-hex prompt hash. A loose
        # \d+ also matches the DATE in old timestamp-named files
        # (annotations_20260714_...), which made the counter jump to 20260715.
        #
        # The model tag sits in the name so three models over one video give
        # you three obviously-distinct files instead of three files whose only
        # difference is a counter. The counter is scoped PER TAG, so each
        # model numbers its own runs from 001.
        pat = re.compile(rf"annotations_{re.escape(model_tag)}_(\d{{3}})_"
                         rf"[0-9a-f]{{8}}\.jsonl$")
        nums = []
        for f in args.outdir.glob(f"annotations_{model_tag}_*.jsonl"):
            m = pat.match(f.name)
            if m:
                nums.append(int(m.group(1)))
        run_no = (max(nums) + 1) if nums else 1
        args.out = (args.outdir /
                    f"annotations_{model_tag}_{run_no:03d}_{prompt_sha}.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    meta = load_manifest(args.manifest)
    print(f"manifest: {len(meta)} videos", flush=True)
    print(f"model:    {model_tag}  [backend={backend}]", flush=True)
    print(f"prompt:   {prompt_sha}", flush=True)
    print(f"writing:  {args.out}", flush=True)

    clips = sorted(args.videos.glob("*.mp4"))
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        clips = [c for c in clips if c.stem in wanted]
        missing = wanted - {c.stem for c in clips}
        if missing:
            raise SystemExit(f"--only: no .mp4 for {sorted(missing)} under {args.videos}")
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"no .mp4 under {args.videos}")

    # Resume across ALL prior runs, not just one file -- but only honour records
    # written under the SAME prompt. A changed prompt makes old records stale,
    # and silently skipping them would leave a corpus that is half one prompt
    # and half another with nothing in the data to say which.
    done = set()
    if args.resume:
        stale = other_model = 0
        for prev in sorted(args.outdir.glob("annotations_*.jsonl")):
            with prev.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # A segment annotated by ANOTHER model is not this model's
                    # work. Without this check, running three models over one
                    # video gives you one full file and two nearly empty ones --
                    # and no error to tell you the comparison is worthless.
                    if r.get("model", LEGACY_MODEL_TAG) != model_tag:
                        other_model += 1
                        continue
                    if r.get("prompt_sha") != prompt_sha:
                        stale += 1
                        continue
                    if r.get("parse_ok"):   # retry anything that failed to parse
                        done.add((r["video_id"], r["segment_index"]))
        print(f"resume:   {len(done)} done, {stale} stale (different prompt), "
              f"{other_model} from other models (ignored)", flush=True)

    work, skipped = [], []
    for clip in clips:
        vid = clip.stem
        info = meta.get(vid)
        if info is None:
            skipped.append(f"{vid} (not in manifest)")
            continue
        if not info["duration"]:
            skipped.append(f"{vid} (no duration)")
            continue
        for i, (start, end) in enumerate(
                segments(info["duration"], args.seconds, args.min_tail)):
            if (vid, i) not in done:
                work.append((clip, vid, info["title"], info["url"], i, start, end))

    if skipped:
        print(f"SKIPPED {len(skipped)}: {', '.join(skipped[:5])}"
              f"{' ...' if len(skipped) > 5 else ''}", flush=True)
    print(f"{len(work)} segments to process ({len(done)} already done)", flush=True)
    if not work:
        return

    annotator = BACKENDS[backend](args.model, args)
    ngpu = torch.cuda.device_count()
    print(f"ready on {ngpu} x {torch.cuda.get_device_name(0)}\n", flush=True)

    n = len(work)
    t_start = time.time()
    ok = bad = 0

    with args.out.open("a") as fout:
        for k, (clip, vid, title, url, i, start, end) in enumerate(work, 1):
            t0 = time.time()
            try:
                raw = annotator.annotate(clip, start, end)
            except Exception as e:
                print(f"[{k}/{n}] FAIL {vid} seg {i}: {e}", flush=True)
                continue

            ann = parse_annotation(raw)
            if ann["parse_ok"]:
                ok += 1
            else:
                bad += 1

            rec = {
                "model": model_tag,
                "backend": backend,
                "prompt_sha": prompt_sha,
                "video_id": vid,
                "video_name": title,
                "url": url,
                # Deep link straight to this segment -- makes spot-checking an
                # annotation a click instead of a scrub.
                "url_at": f"{url}&t={int(start)}s",
                "video": str(clip),
                "segment_index": i,
                "start_sec": round(start, 2),
                "end_sec": round(end, 2),
                "timestamp": f"{hhmmss(start)}-{hhmmss(end)}",
                **ann,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()   # survive a walltime kill

            dt = time.time() - t0
            rate = (time.time() - t_start) / max(ok + bad, 1)
            if ann["parse_ok"]:
                flag = " INCONSISTENT" if ann["inconsistent"] else ""
                summary = (f"infant={ann['num_infants']} adult={ann['num_adults']} "
                           f"vis={ann['infant_visibility']}{flag}")
            else:
                summary = "PARSE FAILED"
            print(f"[{k}/{n}] {vid} seg {i} [{hhmmss(start)}-{hhmmss(end)}] "
                  f"{dt:.1f}s (avg {rate:.1f}s/seg, "
                  f"eta {(n - k) * rate / 3600:.1f}h): {summary}", flush=True)

    total = time.time() - t_start
    print(f"\n{ok} parsed, {bad} unparseable, of {n} segments "
          f"in {total/60:.1f} min ({total/max(ok+bad,1):.1f}s each)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
