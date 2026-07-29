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
import argparse, atexit, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

# audio_windows.py sits next to this file; the sbatch jobs cd elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
# Audio addendum -- ONLY for a model that actually hears the clip (qwen-omni).
#
# Kept OUT of PROMPT deliberately. Asking a video-only model what it hears gets
# you a confident answer invented from the pixels, which is worse than no
# answer because nothing in the output marks it as a guess. It would also
# change prompt_sha for all four existing models and strand the current corpus.
#
# The event vocabulary is CLOSED for the same reason "infant" is defined in
# PROMPT: left open, the model drifts across "baby noise", "cooing", "babble"
# and "vocalisation" between segments, and the labels stop aggregating.
AUDIO_EVENTS = ["infant_vocalisation", "infant_crying", "infant_laughing",
                "adult_speech", "child_speech", "singing", "music",
                "tv_or_device", "household_noise", "outdoor_noise", "silence"]

VALID_AUDIO_EVENTS = set(AUDIO_EVENTS)

AUDIO_PROMPT = """

You can HEAR this clip as well as see it. Add these keys to the same JSON
object:

  "audio_events": [%s],
  "infant_vocalising": bool,
  "speech_present": bool,
  "audio_description": "at most 2 sentences"

THE TWO DESCRIPTIONS ARE SEPARATE AND MUST NOT OVERLAP:
- "description" is what you SEE. Write it as if the clip were silent. Never
  mention a sound, a voice, or anything you only know by hearing it.
- "audio_description" is what you HEAR. Write it as if you could not see the
  clip. Never mention a colour, a place, an object or an action you only know
  by looking at it.
Someone reading the two together should be able to tell which sense each came
from. If a fact appears in both, it belongs in only one of them.

- audio_events: every category actually AUDIBLE. [] if there is no sound at
  all; use "silence" only when the clip has an audio track that is silent.
- Judge sound by EAR, not by what the picture implies. A visibly crying infant
  with no audible cry is not "infant_crying".
- infant_vocalising covers any infant sound -- cry, babble, laugh, grunt.
- speech_present: any intelligible human speech, including a narrator or
  someone off-camera.
- audio_description: what is heard, and who or what makes it. Say if a voice
  comes from someone not visible. Do not speculate about a cause you cannot
  hear.
- Do NOT let the audio change your visual counts. Someone you only hear is not
  a visible person.""" % ", ".join(f'"{e}"' for e in AUDIO_EVENTS)


# ---------------------------------------------------------------------------
# Backends
#
# Both expose the same one-method contract -- annotate(clip, start, end) -> raw
# model text -- so main()'s loop, the JSON parsing and the resume logic are
# identical no matter which checkpoint is loaded.
# ---------------------------------------------------------------------------

def verify_checkpoint(model_dir: Path):
    """Every shard the index names must actually exist.

    A DIRECTORY IS NOT A CHECKPOINT. An interrupted download leaves one that
    passes every cheap test -- it exists, it is 54GB, `ls` looks right -- and
    then dies deep inside from_pretrained with

        FileNotFoundError: .../model-00009-of-00014.safetensors

    after minutes of loading. On a multi-model run that is minutes wasted per
    model behind it, too. The index file lists exactly which shards are
    required, so checking is cheap and exact; do it before touching a GPU."""
    idx = model_dir / "model.safetensors.index.json"
    if idx.is_file():
        try:
            want = set(json.loads(idx.read_text()).get("weight_map", {}).values())
        except (json.JSONDecodeError, OSError) as e:
            raise SystemExit(f"{model_dir.name}: unreadable weight index ({e})")
        missing = sorted(f for f in want if not (model_dir / f).is_file())
        if missing:
            raise SystemExit(
                f"{model_dir.name} is INCOMPLETE: {len(missing)} of {len(want)} "
                f"shards missing, starting with {missing[0]}\n"
                f"  The download was interrupted. Finish it from a login node:\n"
                f"    sbatch scripts/13_fetch_models.sbatch --only <key>\n"
                f"  (a killed transfer keeps finished shards, so it resumes)")
    elif not any(model_dir.glob("*.safetensors")) and not any(model_dir.glob("*.bin")):
        raise SystemExit(f"{model_dir.name}: no weight files at all -- "
                         f"the download never got past metadata.")


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
        # Omni BEFORE the VL check: Qwen3OmniMoeForConditionalGeneration
        # contains neither "VL" nor "Omni"-plus-"VL", but a future variant
        # might -- and routing an Omni checkpoint to the qwen backend would
        # load it as video-only and silently discard the audio, which is the
        # one failure this whole comparison exists to avoid.
        if "Omni" in arch:
            return "qwen-omni"
        if "Qwen" in arch and "VL" in arch:
            return "qwen"
    name = model_dir.name.lower()
    if "internvl" in name:
        return "internvl"
    if "omni" in name:
        return "qwen-omni"
    if "qwen" in name and "vl" in name:
        return "qwen"
    raise SystemExit(
        f"cannot tell which backend {model_dir.name} needs. Pass --backend "
        f"qwen|internvl|qwen-omni explicitly.")


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

    def annotate(self, clip: Path, start: float, end: float,
                 prompt: str = PROMPT) -> str:
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
                {"type": "text", "text": prompt},
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
    def _split_model(model_dir: Path):
        """Explicit layer -> GPU map, mirroring InternVL's own split_model().

        NOT device_map="auto", for a hard reason: "auto" routes through
        transformers' infer_auto_device_map, which reads
        model.all_tied_weights_keys. That attribute is populated by
        PreTrainedModel.post_init(), and InternVL's vendored __init__ never
        calls post_init -- so on transformers 4.5x it is an AttributeError
        before a single weight loads. An explicit dict skips that path entirely.

        The placement also matters on its own terms: the vision tower, the
        embeddings, the final norm and lm_head all sit on GPU 0, and GPU 0 takes
        roughly half a share of decoder layers to pay for carrying the ViT.
        """
        import math
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        llm_cfg = getattr(cfg, "llm_config", None) or getattr(cfg, "text_config", None)
        if llm_cfg is None:
            raise SystemExit(
                f"{model_dir.name}: config has neither llm_config nor "
                f"text_config; cannot place layers. Try --backend qwen if this "
                f"is not actually an InternVL checkpoint.")
        num_layers = llm_cfg.num_hidden_layers
        world = torch.cuda.device_count()

        per = math.ceil(num_layers / (world - 0.5))
        counts = [per] * world
        counts[0] = math.ceil(per * 0.5)

        device_map, layer = {}, 0
        for gpu, n in enumerate(counts):
            for _ in range(n):
                if layer >= num_layers:
                    break
                device_map[f"language_model.model.layers.{layer}"] = gpu
                layer += 1

        # Everything non-decoder on GPU 0, including the LAST layer: it feeds
        # the norm and lm_head, and splitting them costs a device hop per token.
        for k in ("vision_model", "mlp1",
                  "language_model.model.tok_embeddings",
                  "language_model.model.embed_tokens",
                  "language_model.model.norm",
                  "language_model.model.rotary_emb",
                  "language_model.output",
                  "language_model.lm_head"):
            device_map[k] = 0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
        return device_map

    @staticmethod
    def _patch_tied_weights():
        """transformers 4.5x reads all_tied_weights_keys in several load paths,
        not only the device-map one. post_init() sets it; InternVL's vendored
        __init__ skips post_init, so give the base class a default.

        A plain CLASS ATTRIBUTE, deliberately not a property: any model that
        does set its own instance value must still shadow this, and a read-only
        property would break that assignment for every other model in the
        process."""
        from transformers import PreTrainedModel
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = {}

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
        self._patch_tied_weights()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True, use_fast=False)
        print(f"loading weights from {model_dir.name} (off GPFS, minutes for a "
              f"78B)...", flush=True)
        # trust_remote_code: InternVL ships its own modelling code, including
        # the .chat() helper used below. AutoModelForImageTextToText does not
        # cover it.
        #
        # device_map is an explicit dict on multi-GPU (see _split_model) and a
        # plain {"": 0} on one card. Neither is the string "auto", which is what
        # triggers infer_auto_device_map and the all_tied_weights_keys crash.
        device_map = (self._split_model(model_dir)
                      if torch.cuda.device_count() > 1 else {"": 0})
        self.model = AutoModel.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, trust_remote_code=True,
            low_cpu_mem_usage=True, device_map=device_map).eval()

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

        dec = VideoDecoder(str(clip))

        # CLAMP TO THE REAL STREAM, NOT THE MANIFEST.
        #
        # Segment bounds come from the manifest duration, which is yt-dlp's
        # container metadata and routinely overshoots the decodable stream by a
        # fraction of a second. On the FINAL window that makes the last sample
        # land past the last frame and torchcodec refuses it outright:
        #   "frame pts is 493.750000; must be less than 493.400000"
        # which cost one segment per video per InternVL model. qwen_vl_utils
        # clamps internally, which is why the Qwen backend never showed this.
        md = getattr(dec, "metadata", None)
        limit = None
        for attr in ("end_stream_seconds", "duration_seconds"):
            v = getattr(md, attr, None) if md is not None else None
            if v:
                limit = float(v)
                break
        if limit is not None:
            # Strictly BELOW the limit: the check is exclusive.
            end = min(end, limit - 1e-3)
            start = min(start, end - 1e-3)

        if end <= start:
            raise ValueError(
                f"window [{start:.2f},{end:.2f}] is empty after clamping to the "
                f"stream ({limit}) -- the manifest duration overshoots the file.")

        n = int(round((end - start) * self.args.fps))
        n = max(1, min(self.args.max_frames, n))
        step = (end - start) / n
        stamps = [start + (i + 0.5) * step for i in range(n)]

        batch = dec.get_frames_played_at(stamps)
        return batch.data          # (N, C, H, W) uint8

    def annotate(self, clip: Path, start: float, end: float,
                 prompt: str = PROMPT) -> str:
        from PIL import Image

        frames = self._frames(clip, start, end)
        imgs = [Image.fromarray(f.permute(1, 2, 0).cpu().numpy()) for f in frames]

        pixel_values = torch.stack([self.transform(im) for im in imgs])
        pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)

        # InternVL wants one <image> placeholder per frame, numbered. Without
        # the Frame-N prefixes it treats the batch as unordered stills and
        # loses the temporal ordering the whole task depends on.
        prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(imgs)))
        question = prefix + prompt

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


class QwenOmniAnnotator:
    """Qwen3-Omni. The only backend here that receives the AUDIO ITSELF.

    The other three get a Whisper transcript in their prompt (--transcribe);
    this one gets the waveform, so it can use crying, tone and room noise --
    none of which survives transcription. That is the whole point of having it
    in the comparison.

    TWO THINGS DIFFER FROM QwenAnnotator, both forced:

    1. qwen_omni_utils, not qwen_vl_utils. Different package, different entry
       point (process_mm_info), and it takes use_audio_in_video to pull the
       clip's own soundtrack rather than a separate audio file.

    2. NO video_start/video_end. qwen_vl_utils decodes a sub-range out of the
       full mp4, which is what lets every other backend segment without
       touching the disk. process_mm_info takes a whole file, so a segment has
       to BE a file -- hence _cut(), which writes a real 10s mp4 to a temp dir.
       That is the cost of native audio here, and it is why this backend is
       slower per segment than the frame-sampling ones.
    """

    def __init__(self, model_dir: Path, args):
        from transformers import Qwen3OmniMoeForConditionalGeneration, AutoProcessor
        from qwen_omni_utils import process_mm_info

        self.args = args
        self._process_mm_info = process_mm_info
        self.processor = AutoProcessor.from_pretrained(str(model_dir))

        # enable_audio_output=False at LOAD time, not disable_talker() after.
        #
        # The talker synthesises SPEECH; we want JSON, so it is pure cost. But
        # the order matters for more than memory: built first and deleted
        # after, it is still present while accelerate computes the device map,
        # so "auto" spreads the thinker across every card to make room for
        # weights that are about to be thrown away. The thinker's towers then
        # sit on different GPUs than its text model and every segment dies
        # with:
        #   Expected all tensors to be on the same device, but found at least
        #   two devices, cuda:0 and cuda:1!
        # Never building it keeps the thinker whole.
        #
        # Talker-less, this is ~62GB in bf16, which fits one 80GB A100 with
        # room for activations -- so pin it to ONE card and the entire class of
        # cross-device bug cannot occur. Fall back to sharding only if the
        # cards are too small to hold it.
        free, total = torch.cuda.mem_get_info(0)
        single_card = total / 1e9 > 75

        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            str(model_dir),
            dtype="auto",
            device_map={"": 0} if single_card else "auto",
            enable_audio_output=False,
            attn_implementation="sdpa")
        self.model.eval()
        if getattr(self.model, "has_talker", False):
            # Belt and braces: if a future config ignores the kwarg, drop it
            # rather than silently paying for it.
            self.model.disable_talker()

        self._tmp = Path(tempfile.mkdtemp(prefix="omni_seg_"))
        self._limits: dict[str, float] = {}   # clip -> decodable length
        # Segments are unlinked as they are used; this clears the directory
        # itself, including anything left by a segment that raised mid-cut.
        atexit.register(shutil.rmtree, self._tmp, True)

    def _cut(self, clip: Path, start: float, end: float) -> Path:
        """[start, end] of clip -> a real mp4, video AND audio.

        Re-encoded, not -c copy: a stream copy snaps to the nearest keyframe,
        which would put this backend on different frames than the others and
        make the comparison meaningless. No -an here -- the audio is the point.

        -t (duration), NOT -to (absolute stop): as an INPUT option the meaning
        of -to has shifted between ffmpeg majors, and a silently mis-cut window
        would look like a model disagreement rather than a bug.

        crf 18 / veryfast, not ultrafast: this backend is the only one seeing
        re-encoded pixels, so compression artefacts here are a confound the
        other four do not carry. Near-visually-lossless is worth the seconds.
        """
        dst = self._tmp / f"{clip.stem}_{start:.3f}_{end:.3f}.mp4"
        if dst.exists():
            return dst
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-ss", str(start), "-t", str(end - start), "-i", str(clip),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-c:a", "aac", str(dst)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0 or not dst.exists():
            raise RuntimeError(f"ffmpeg could not cut {clip.name} "
                               f"[{start:.2f},{end:.2f}]: {p.stderr.strip()}")

        # VERIFY BEFORE HANDING IT TO THE DECODER.
        #
        # ffmpeg exits 0 on a window past the end of the stream and writes a
        # valid container with no frames in it. qwen_omni_utils then feeds that
        # to a native audio decoder which does not check, and the process
        # SEGFAULTS -- taking every remaining segment with it. Raising here
        # turns a run-ending crash into one logged FAIL line.
        q = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=codec_type", "-of", "csv=p=0", str(dst)],
            capture_output=True, text=True)
        fields = [x.strip() for x in q.stdout.split() if x.strip()]
        dur = None
        for f in fields:
            try:
                dur = float(f)
            except ValueError:
                continue
        if dur is None or dur < 0.05:
            dst.unlink(missing_ok=True)
            raise RuntimeError(
                f"cut of {clip.name} [{start:.2f},{end:.2f}] is empty "
                f"(duration={dur}) -- window is past the end of the stream.")
        if "audio" not in q.stdout:
            dst.unlink(missing_ok=True)
            raise RuntimeError(
                f"cut of {clip.name} [{start:.2f},{end:.2f}] has no audio "
                f"stream; this backend exists to hear it.")
        return dst

    def _limit(self, clip: Path) -> float | None:
        """Decodable length of clip, seconds. Cached: one ffprobe per video,
        not per segment."""
        key = str(clip)
        if key not in self._limits:
            p = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(clip)], capture_output=True, text=True)
            val = p.stdout.strip()
            try:
                self._limits[key] = float(val) if val and val != "N/A" else None
            except ValueError:
                self._limits[key] = None
        return self._limits[key]

    def annotate(self, clip: Path, start: float, end: float,
                 prompt: str = PROMPT) -> str:
        # CLAMP TO THE REAL STREAM -- the same correction _frames() makes for
        # InternVL, and for the same reason: segment bounds come from the
        # manifest, which is yt-dlp container metadata and routinely overshoots
        # the decodable file.
        #
        # It matters MORE here. InternVL got a clean torchcodec error. This
        # backend hands the out-of-range window to ffmpeg, which cheerfully
        # writes a zero-length clip, and the native audio decoder then
        # SEGFAULTS on it -- killing the whole process and every segment still
        # queued behind it. A run that should lose one segment loses the rest
        # of the video, and the traceback names neither audio nor the segment.
        limit = self._limit(clip)
        if limit is not None:
            end = min(end, limit - 1e-3)
            start = min(start, end - 1e-3)
        if end <= start:
            raise ValueError(
                f"window [{start:.2f},{end:.2f}] is empty after clamping to "
                f"the stream ({limit}) -- the manifest duration overshoots "
                f"the file.")

        seg = self._cut(clip, start, end)
        try:
            msgs = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": str(seg),
                     "fps": self.args.fps,
                     "total_pixels": self.args.total_pixels_factor * 32 * 32},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            # use_audio_in_video MUST match between process_mm_info and the
            # processor call. Mismatched, the audio and video token counts
            # disagree and it fails deep inside the model with a shape error.
            audios, images, videos = self._process_mm_info(
                msgs, use_audio_in_video=True)
            # dtype=, not just device=. The audio feature extractor emits
            # float32 mel features, but the audio tower's weights are bf16, and
            # a bare .to(device) moves them without converting:
            #   Input type (float) and bias type (c10::BFloat16) should be the
            #   same
            # -- raised on the tower's first Conv1d, so EVERY segment fails and
            # none of them fail for a reason that mentions audio.
            #
            # BatchFeature.to() applies a dtype only to floating-point tensors
            # and merely moves the rest, so input_ids stays Long. That is why
            # this can be a blanket cast rather than a per-key one.
            inputs = self.processor(
                text=[text], audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True,
                use_audio_in_video=True,
            ).to(device=self.model.device, dtype=self.model.dtype)

            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    # thinker_max_new_tokens, NOT max_new_tokens. Omni's
                    # generate() seeds thinker_kwargs with its OWN default of
                    # 1024 and then merges extra kwargs only `if key not in
                    # thinker_kwargs` -- so a plain max_new_tokens is silently
                    # DISCARDED. That is a 4x longer generation than every
                    # other backend, and the rambling this prompt was designed
                    # to stop, with nothing in the output to say why.
                    thinker_max_new_tokens=self.args.max_new_tokens,
                    # These two have no thinker_ default, so they fall through
                    # shared_kwargs into the thinker as-is.
                    do_sample=False,
                    repetition_penalty=1.05,
                    use_audio_in_video=True,
                    return_audio=False,     # text out; no speech synthesis
                )
            # generate() ALWAYS returns (thinker_result, None) when it is not
            # synthesising audio -- a tuple even with the talker deleted.
            if isinstance(out, (tuple, list)):
                out = out[0]
            return self.processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)[0].strip()
        finally:
            # Per segment, not per run: a 500s video at 10s windows would
            # otherwise leave 50 re-encoded mp4s in $TMPDIR.
            seg.unlink(missing_ok=True)


BACKENDS = {"qwen": QwenAnnotator, "internvl": InternVLAnnotator,
            "qwen-omni": QwenOmniAnnotator}


# ---------------------------------------------------------------------------
# Audio: transcript injected into the prompt (--transcribe)
# ---------------------------------------------------------------------------
# None of the VLMs here has an audio encoder, so "give the model the audio"
# means giving it TEXT. This is the cheap half of that; the other half is a
# real omni model, which sees the waveform itself.
#
# Appended, never interpolated into the middle of PROMPT: the JSON contract
# above stays byte-identical, so a run with audio and a run without differ by
# a suffix rather than by a reworded task.
TRANSCRIPT_TEMPLATE = (
    "\n\nAudio transcript of this clip (speech only; it may be empty, "
    "inaccurate, or come from off-camera narration rather than the people you "
    "can see):\n\"{transcript}\"\n"
    "Use it only where it agrees with what is visible. Do NOT count a person "
    "you cannot see just because you can hear them."
)


class CachedTranscriber:
    """Whisper, run at most ONCE per video across every run and every model.

    WHY CACHED RATHER THAN JUST TRANSCRIBED INLINE:
    28_run_all_models.sh puts four models over the same video. Transcribing per
    run costs four ASR passes for one result, and -- worse -- if any two passes
    differ by a word, 27_compare_models.py is silently comparing models AND
    transcripts at once. The cache makes the text an input to the comparison
    instead of a variable in it.

    The cache key is the video id plus the ASR model tag, so swapping Whisper
    for something else does not quietly reuse the old text.
    """

    def __init__(self, model_dir: Path, cache_dir: Path, args):
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.args = args
        self.tag = self.model_dir.name
        self._pipe = None
        self._mem: dict[str, list] = {}

    def _load(self):
        """Deferred: a cache hit on every video means Whisper never loads, and
        that is the common case on a rerun."""
        if self._pipe is not None:
            return
        from transformers import (AutoModelForSpeechSeq2Seq, AutoProcessor,
                                  pipeline)
        print(f"loading ASR: {self.tag}", flush=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(self.model_dir), torch_dtype=dtype, low_cpu_mem_usage=True)
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        proc = AutoProcessor.from_pretrained(str(self.model_dir))
        self._pipe = pipeline(
            "automatic-speech-recognition", model=model,
            tokenizer=proc.tokenizer, feature_extractor=proc.feature_extractor,
            torch_dtype=dtype, device=0 if torch.cuda.is_available() else -1,
            chunk_length_s=30, return_timestamps="word")

    def words(self, clip: Path) -> list:
        """[{w, start, end}] for the whole track. Cached on disk and in memory.

        Returns [] for a clip with no audio -- a silent video is a fact about
        the corpus, not an error, and the annotation still has the frames.
        """
        vid = clip.stem
        if vid in self._mem:
            return self._mem[vid]

        cache = self.cache_dir / f"{vid}.{self.tag}.json"
        if cache.exists():
            try:
                self._mem[vid] = json.loads(cache.read_text())["words"]
                return self._mem[vid]
            except (json.JSONDecodeError, KeyError):
                # A run killed mid-write leaves a truncated file. Redo it
                # rather than propagating half a transcript.
                cache.unlink(missing_ok=True)

        from audio_windows import AudioWindows, NoAudioTrack
        try:
            wav = AudioWindows(clip).wav
        except NoAudioTrack:
            words = []
        else:
            self._load()
            with torch.inference_mode():
                out = self._pipe(
                    wav, generate_kwargs={"language": self.args.language,
                                          "task": "transcribe"})
            words = []
            for ch in out.get("chunks") or []:
                ts = ch.get("timestamp") or (None, None)
                a, b = ts[0], ts[1]
                if a is None:
                    continue
                text = str(ch.get("text", "")).strip()
                if text:
                    words.append({"w": text, "start": float(a),
                                  "end": float(b if b is not None else a)})

        # Written atomically: two array-job tasks on the same video otherwise
        # interleave into one unparseable file.
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps({"model": self.tag, "video_id": vid,
                                   "words": words}))
        tmp.replace(cache)
        self._mem[vid] = words
        return words

    def segment_text(self, clip: Path, start: float, end: float) -> str:
        """Words whose MIDPOINT lands in [start, end).

        Midpoint, not overlap: a word straddling the boundary would otherwise
        be handed to both segments, and the per-segment transcripts would no
        longer sum back to what was actually said.
        """
        out = [w["w"] for w in self.words(clip)
               if start <= (w["start"] + w["end"]) / 2.0 < end]
        return " ".join(out).strip()


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------

def init_wandb(args, model_tag, backend, prompt_sha, videos, n_segments):
    """Start a W&B run, or return None when --wandb was not passed.

    OFFLINE BY DEFAULT, and that default is the whole point: compute nodes here
    have no outbound internet, and wandb.init() in online mode does not fail
    fast on that -- it blocks, retries, and then drops your metrics while the
    job appears to run normally. Offline writes to disk; you sync from a login
    node afterwards (the command is printed at the end of the run).

    Runs are GROUPED BY VIDEO and named by model, so the three models over one
    video land in one comparable group in the W&B UI instead of three unrelated
    runs you have to eyeball side by side.

    The run NAME carries the audio mode too: the same model over the same video
    with and without audio is otherwise two runs with one name, which is the
    one pair you most need to tell apart."""
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        raise SystemExit(
            "--wandb needs the wandb package, which is not installed.\n"
            "  Compute nodes have no internet, so install from a LOGIN node:\n"
            "    ssh bigpurple-ln3 && conda activate ibdp && pip install wandb")

    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    # Run dirs on gpfs, never $HOME -- system metrics every few seconds for
    # hours is not something to point at a small quota.
    os.environ.setdefault("WANDB_DIR", str(args.outdir))

    # "native" for a model that hears the waveform, "transcript" for one fed
    # Whisper text, "none" for video only. Three modes, not a bool, because
    # Omni-with-audio and Qwen-VL-with-a-transcript are not the same treatment.
    audio_mode = ("native" if backend == "qwen-omni"
                  else "transcript" if args.transcribe
                  else "none")

    group = videos[0] if len(videos) == 1 else f"{len(videos)}-videos"
    suffix = "" if audio_mode == "none" else f"--{audio_mode}"

    # SHARED-RUN MODE. When WANDB_RUN_ID is set -- 28_run_all_models.sh sets one
    # per video -- every model in the sweep writes into ONE run instead of one
    # run each. wandb reads the id straight from the environment, and
    # WANDB_RESUME=allow lets each successive process attach to it.
    #
    # Two things MUST change in that mode or the shared run is unreadable:
    #   1. metric keys get a per-model prefix (see the log call), otherwise six
    #      models overwrite each other on identical key names;
    #   2. the explicit step= is dropped. Steps must increase monotonically
    #      within a run, and every process restarts its own k at 1 -- passing it
    #      makes wandb discard everything after the first model.
    shared = bool(os.environ.get("WANDB_RUN_ID"))

    run = wandb.init(
        project=args.wandb_project,
        name=f"all-models--{group}" if shared else f"{model_tag}--{group}{suffix}",
        group=group,
        job_type=f"{model_tag}[{audio_mode}]",
        config={
            "model": model_tag,
            "backend": backend,
            "prompt_sha": prompt_sha,
            "videos": videos,
            "n_segments": n_segments,
            "window_sec": args.seconds,
            "min_tail_sec": args.min_tail,     # also moves segment boundaries
            "fps": args.fps,
            "max_new_tokens": args.max_new_tokens,
            "total_pixels_factor": args.total_pixels_factor,
            "max_frames": args.max_frames,
            # THE AUDIO AXIS. Without these, an audio run and a video-only run
            # are distinguishable in the UI only by a changed prompt_sha -- a
            # hex string -- which is useless for grouping or filtering the one
            # comparison this whole thing exists to make.
            "audio_mode": audio_mode,
            "asr_model": (args.asr_model.name if args.transcribe else None),
            "language": (args.language if args.transcribe else None),
            "n_gpus": torch.cuda.device_count(),
            "gpu": torch.cuda.get_device_name(0),
        },
    )
    # GPU utilisation, VRAM, power and temperature are captured automatically
    # from here on -- that is the "track the GPU usage" part, no code needed.
    print(f"wandb:    {os.environ.get('WANDB_MODE')} mode, run {run.name}",
          flush=True)
    return run


def table_columns(audio: bool) -> list:
    """Columns for the per-segment results table. Audio columns only when the
    model heard the clip -- an empty audio column on a video-only run reads as
    "heard nothing", the same trap as a null field in the JSONL."""
    cols = ["segment", "timestamp", "url_at", "parse_ok",
            "num_infants", "num_children", "num_adults", "num_humans_total",
            "infant_visibility", "visible_parts", "inconsistent",
            "description"]
    if audio:
        cols += ["audio_events", "infant_vocalising", "speech_present",
                 "audio_inconsistent", "audio_description"]
    cols += ["elapsed_sec"]
    return cols


def table_row(rec: dict, ann: dict, audio: bool) -> list:
    """One row, ordered to match table_columns(). Lists are joined to strings:
    W&B renders a Python list as its repr, which is unreadable in a cell and
    cannot be filtered on."""
    row = [rec["segment_index"], rec["timestamp"], rec["url_at"],
           bool(ann.get("parse_ok")),
           ann.get("num_infants"), ann.get("num_children"),
           ann.get("num_adults"), ann.get("num_humans_total"),
           ann.get("infant_visibility"),
           ", ".join(ann.get("visible_infant_parts") or []),
           bool(ann.get("inconsistent")),
           # Unparseable segments keep their raw text -- that is the only place
           # you can see WHY they failed, and truncation is invisible in a
           # parse_ok=false flag alone.
           ann.get("description") or (ann.get("raw") or "")[:500]]
    if audio:
        row += [", ".join(ann.get("audio_events") or []),
                bool(ann.get("infant_vocalising")),
                bool(ann.get("speech_present")),
                bool(ann.get("audio_inconsistent")),
                ann.get("audio_description") or ""]
    row += [rec["elapsed_sec"]]
    return row


def log_table(run, rows: list, audio: bool):
    """Log the accumulated rows as a fresh wandb.Table.

    REBUILT each time rather than mutating one long-lived Table: W&B treats a
    logged Table as immutable and adding rows to one that has already been
    logged either warns or silently drops the additions.

    Called periodically, NOT only at the end -- this job dies to walltime kills
    and, as of the last run, a segfault on the final segment. A table logged
    only on clean exit is a table you never get for the runs you most want to
    inspect.
    """
    if run is None:
        return
    import wandb
    run.log({"results": wandb.Table(columns=table_columns(audio), data=rows)})


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


def parse_annotation(raw: str, audio: bool = False) -> dict:
    """Model text -> validated dict. Never raises: a segment that returns junk
    records parse_ok=false and keeps its raw text, rather than killing the run or
    silently writing zeros that look like real observations.

    audio=True adds the AUDIO_PROMPT keys. The fields are absent rather than
    null for a video-only run: a null "infant_vocalising" on a model that
    cannot hear reads like a heard-nothing observation, and would be counted as
    one by anything aggregating later.
    """
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

    if audio:
        events = d.get("audio_events") or []
        if not isinstance(events, list):
            events = []
        # Unknown labels are DROPPED, not kept: a free-text event that appears
        # in one segment and never again is noise in every aggregate, and
        # keeping it would defeat the point of a closed vocabulary.
        events = [e for e in (str(x).strip().lower() for x in events)
                  if e in VALID_AUDIO_EVENTS]
        ann["audio_events"] = events
        ann["infant_vocalising"] = as_bool(d.get("infant_vocalising"))
        ann["speech_present"] = as_bool(d.get("speech_present"))
        ann["audio_description"] = str(d.get("audio_description", "")).strip()

        # Same treatment as the visual counts: flag the disagreement, never
        # silently repair it. "silence" alongside real events, or a claim of
        # infant vocalisation with no infant sound in the event list, means the
        # segment should be looked at rather than quietly trusted.
        infant_sounds = {"infant_vocalisation", "infant_crying",
                         "infant_laughing"}
        speech_sounds = {"adult_speech", "child_speech"}
        ann["audio_inconsistent"] = bool(
            ("silence" in events and len(events) > 1)
            or (ann["infant_vocalising"] and not (infant_sounds & set(events)))
            or (ann["speech_present"] and not (speech_sounds & set(events)))
        )

    ann["parse_ok"] = True
    return ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                    help="local checkpoint dir under $ROOT/models")
    ap.add_argument("--backend", choices=sorted(BACKENDS),
                    help="override backend auto-detection")
    ap.add_argument("--transcribe", action="store_true",
                    help="transcribe the audio and put it in the prompt. "
                         "Cached per video, so all models see the same text.")
    ap.add_argument("--asr-model", type=Path,
                    default=ROOT / "models/whisper-large-v3")
    ap.add_argument("--asr-cache", type=Path,
                    default=ROOT / "outputs/transcripts/cache")
    ap.add_argument("--language", default="en",
                    help="pinned, not auto-detected: a misdetect makes Whisper "
                         "TRANSLATE, inventing fluent English nobody said.")
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
    ap.add_argument("--wandb", action="store_true",
                    help="log GPU usage + per-segment metrics to Weights & Biases")
    ap.add_argument("--wandb-project", default="ibdp")
    ap.add_argument("--wandb-table-every", type=int, default=10,
                    help="re-log the results table every N segments, so a "
                         "killed run still has its results in the UI.")
    ap.add_argument("--wandb-mode", default="offline",
                    choices=["offline", "online", "disabled"],
                    help="offline is the default: compute nodes have no "
                         "internet and online mode hangs there. Sync afterwards.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU -- needs an A100 allocation, not a login node.")

    if not args.model.is_dir():
        raise SystemExit(f"no checkpoint at {args.model}")
    verify_checkpoint(args.model)
    model_tag = args.model_tag or args.model.name
    backend = args.backend or detect_backend(args.model)

    # Stamped into every record. Two runs with different prompts are then
    # distinguishable after the fact, and --resume can tell fresh from stale.
    #
    # --transcribe is folded into the hash. Without that, an audio run and a
    # video-only run produce records that are indistinguishable, and --resume
    # would treat the older ones as already done -- leaving a corpus that is
    # half one modality and half the other with nothing in the data to say so.
    # AUDIO_PROMPT likewise: an Omni run that reports what it heard asks a
    # strictly larger question than one that does not, so the two must not
    # resume into each other either.
    native_audio = backend == "qwen-omni"
    prompt_sha = hashlib.sha256(
        (PROMPT
         + (TRANSCRIPT_TEMPLATE if args.transcribe else "")
         + (AUDIO_PROMPT if native_audio else "")).encode()
    ).hexdigest()[:8]

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

    # Before the weight load, so the load itself shows up in the GPU-memory
    # trace -- that spike is exactly what you want to see when sizing a model
    # against the cards you have.
    run = init_wandb(args, model_tag, backend, prompt_sha,
                     sorted({w[1] for w in work}), len(work))

    asr = None
    if args.transcribe:
        if not args.asr_model.is_dir():
            raise SystemExit(
                f"--transcribe needs an ASR checkpoint at {args.asr_model}\n"
                f"  fetch it on a LOGIN node: "
                f"scripts/13_fetch_models.sh --only whisper-large-v3")
        asr = CachedTranscriber(args.asr_model, args.asr_cache, args)
        print(f"asr:      {asr.tag}  (cache: {args.asr_cache})", flush=True)

    annotator = BACKENDS[backend](args.model, args)
    ngpu = torch.cuda.device_count()
    print(f"ready on {ngpu} x {torch.cuda.get_device_name(0)}\n", flush=True)

    n = len(work)
    t_start = time.time()
    ok = bad = 0
    table_rows = []

    with args.out.open("a") as fout:
        for k, (clip, vid, title, url, i, start, end) in enumerate(work, 1):
            t0 = time.time()

            # Transcription is timed INSIDE the segment, but the cache means
            # only the first segment of each video actually pays for it.
            transcript = None
            # A model that hears the clip is asked to report what it heard.
            prompt = PROMPT + (AUDIO_PROMPT if native_audio else "")
            if asr is not None:
                try:
                    transcript = asr.segment_text(clip, start, end)
                    prompt = prompt + TRANSCRIPT_TEMPLATE.format(
                        transcript=transcript)
                except Exception as e:
                    # Falling back to video-only is correct, but it must be
                    # visible: an annotation silently made without the audio
                    # it was supposed to have is worse than a loud failure.
                    print(f"[{k}/{n}] ASR FAIL {vid} seg {i}: {e}", flush=True)
                    transcript = None

            try:
                raw = annotator.annotate(clip, start, end, prompt)
            except Exception as e:
                print(f"[{k}/{n}] FAIL {vid} seg {i}: {e}", flush=True)
                continue

            ann = parse_annotation(raw, audio=native_audio)
            if ann["parse_ok"]:
                ok += 1
            else:
                bad += 1

            # Measured here, before the record is written, so every annotation
            # carries what it cost. Comparing models is not only about whether
            # they agree -- a 78B that is 4x slower for the same counts is not
            # worth the four GPUs, and without this the comparison cannot say so.
            dt = time.time() - t0

            rec = {
                "model": model_tag,
                "backend": backend,
                "elapsed_sec": round(dt, 2),
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
                # Stored so a description can be audited against what the model
                # was actually told. "audio_mode" distinguishes a genuinely
                # silent segment (transcript "") from a video-only run (null).
                "audio_mode": ("transcript" if asr is not None else None),
                "transcript": transcript,
                **ann,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()   # survive a walltime kill

            rate = (time.time() - t_start) / max(ok + bad, 1)

            if run is not None:
                # max_memory_allocated is cumulative-max across the process, so
                # on a sharded model it answers "did any card come close to the
                # edge", which is the question that matters before scaling up.
                peak = sum(torch.cuda.max_memory_allocated(d)
                           for d in range(torch.cuda.device_count()))
                metrics = {
                    "segment/elapsed_sec": dt,
                    "segment/parse_ok": int(ann["parse_ok"]),
                    "run/parsed": ok,
                    "run/unparseable": bad,
                    "run/parse_rate": ok / max(ok + bad, 1),
                    "run/avg_sec_per_segment": rate,
                    "run/eta_hours": (n - k) * rate / 3600,
                    "gpu/peak_alloc_gb": peak / 1e9,
                }
                if ann["parse_ok"]:
                    metrics.update({
                        "ann/num_infants": ann["num_infants"],
                        "ann/num_adults": ann["num_adults"],
                        "ann/num_humans_total": ann["num_humans_total"],
                        "ann/inconsistent": int(ann["inconsistent"]),
                    })
                # Shared run: namespace by model and let wandb assign the step.
                # Six models write into one run, so identical key names would
                # collide, and each process restarting k at 1 would make an
                # explicit step= non-monotonic -- wandb drops those silently.
                if os.environ.get("WANDB_RUN_ID"):
                    run.log({f"{model_tag}/{key}": v for key, v in metrics.items()})
                else:
                    run.log(metrics, step=k)
                table_rows.append(table_row(rec, ann, native_audio))
                # Periodic, so a killed run still has most of its results in
                # the UI. Every row is already safe in the JSONL; this is about
                # being able to READ them without sshing in.
                if k % args.wandb_table_every == 0:
                    log_table(run, table_rows, native_audio)
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

    if run is not None:
        summary = {
            "segments_attempted": n,
            "parsed": ok,
            "unparseable": bad,
            "parse_rate": ok / max(ok + bad, 1),
            "total_minutes": total / 60,
            "sec_per_segment": total / max(ok + bad, 1),
            "output_file": args.out.name,
        }
        # Same collision problem as the metrics: in a shared run the last model
        # to finish would otherwise be the only one whose summary survives.
        if os.environ.get("WANDB_RUN_ID"):
            summary = {f"{model_tag}/{k2}": v for k2, v in summary.items()}
        run.summary.update(summary)
        # Final flush: the last partial block since the periodic log.
        log_table(run, table_rows, native_audio)
        run.finish()
        # Offline runs are inert until synced, and a run nobody syncs is a run
        # nobody sees. Print the exact command rather than leaving it to be
        # rediscovered later.
        if os.environ.get("WANDB_MODE") == "offline":
            print(f"\nwandb: offline. From a LOGIN node, sync with:\n"
                  f"  wandb sync {args.outdir}/wandb/offline-run-*")


if __name__ == "__main__":
    main()
