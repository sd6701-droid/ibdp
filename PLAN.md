# IBDP — Qwen3 + Qwen3-VL on BigPurple

Local, PHI-safe multimodal inference for the IBD Program.
Text reasoning over clinical notes, extraction from PDFs/images, and **video clip → text** (endoscopy).

---

## 0. Target environment (confirmed)

| | |
|---|---|
| Cluster | NYU Langone **BigPurple** (SLURM, Lmod, Singularity, no root, no Docker) |
| GPU | **NVIDIA A100 (40 / 80 GB)** — compute capability 8.0 (Ampere) |
| Data | **PHI.** Everything stays on-cluster. No external API calls, ever. |
| Compute nodes | Assume **no internet**. Models are pre-staged from a login node. |
| Modes | **Both** — a vLLM server for interactive work, *and* sbatch scripts for batch runs |
| Dev machine | Apple M4 Pro (authoring only — models never run here) |

### Two hardware facts that drive every decision below

1. **Use bf16 checkpoints, NOT FP8.** A100 is CC 8.0; native FP8 tensor-core math requires CC ≥ 8.9 (Ada/Hopper). vLLM *will* still load a `-FP8` checkpoint on A100 by silently falling back to **FP8-Marlin weight-only (W8A16)** — you get the VRAM saving but **no speedup**, and Marlin's FP8 MoE coverage has been incomplete (vLLM issue #17579), which hits exactly the `30B-A3B-FP8` model we'd otherwise want. If we ever need to shrink memory, **AWQ / GPTQ-Int4 is the better-trodden path on Ampere than FP8.**
2. **Never let this land on a V100.** V100 (CC 7.0) cannot do bf16 at all, and FlashAttention-2 does not build for Volta. Every SLURM script pins `--constraint=a100`.

---

## 1. Model selection

**Vision (the main workhorse — images, PDF pages, and video):**

| Model | VRAM (bf16) | Fits | Use |
|---|---|---|---|
| `Qwen/Qwen3-VL-8B-Instruct` | ~18 GB | 1×A100-40, TP1 | **Default.** Dev, iteration, most batch work. |
| `Qwen/Qwen3-VL-32B-Instruct` | ~66 GB | 1×A100-80, or TP2 on 40 GB | Scale-up when 8B quality isn't enough. |

Qwen3-VL has **native video** support (interleaved-MRoPE + DeepStack + text-timestamp alignment) with 256K context — this is what makes the endoscopy-clip use case viable at all.

**Text (notes, reasoning, RAG generation):**

| Model | VRAM (bf16) | Fits | Use |
|---|---|---|---|
| `Qwen/Qwen3-8B` | ~16 GB | 1×A100-40 | Default. Hybrid-thinking; disable with `enable_thinking=False`. |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | ~61 GB | 1×A100-80, TP2 on 40 | Best quality per A100. MoE = 3.3B active. |

> ⚠️ There is **no** `Qwen3-8B-Instruct-2507` or `Qwen3-32B-Instruct-2507`. The 2507 refresh only covers 4B / 30B-A3B / 235B-A22B. For 8B and 32B you use the original hybrid checkpoints and turn thinking off in the chat template.
>
> ⚠️ **MoE saves compute, not memory.** All 30B of experts stay resident in VRAM. `30B-A3B` is *not* a 3B model.

**Deliberately deferred:** Qwen3.5 (Feb 2026, natively multimodal, no separate `-VL` line) is newer but shares the video bug below and has thinner tooling. We start on Qwen3-VL and revisit.

---

## 2. Repository layout

```
ibdp/
├── PLAN.md                     ← this file
├── README.md                   ← quickstart
├── pyproject.toml              ← package metadata; src/ layout
├── requirements/
│   ├── serve.txt               ← vLLM env (the server)
│   └── batch.txt               ← transformers env (offline batch)
├── configs/
│   ├── models.yaml             ← model ids, TP size, VRAM, context
│   ├── cluster.yaml            ← BigPurple paths, partition, account
│   └── prompts/                ← task prompts, version-controlled
├── scripts/                    ← everything you run ON BigPurple
│   ├── 00_probe_cluster.sh     ← discover partitions/GPUs/modules
│   ├── 01_setup_env.sh         ← build the two conda envs
│   ├── 02_fetch_models.sh      ← pre-stage weights to /gpfs (login node)
│   ├── 03_serve.sbatch         ← launch vLLM (text or VL)
│   ├── 04_tunnel.sh            ← SSH tunnel from laptop → compute node
│   └── 05_batch.sbatch         ← array job for dataset-scale runs
├── src/ibdp/
│   ├── config.py               ← loads YAML, resolves paths
│   ├── client.py               ← OpenAI-compatible client (points at vLLM)
│   ├── media.py                ← PDF→PNG, video frame sampling
│   ├── schemas.py              ← Pydantic models for structured extraction
│   ├── tasks/{video,document,text}.py
│   ├── rag/{index,query}.py
│   └── batch/run.py            ← the batch driver
└── examples/                   ← synthetic-data smoke tests only
```

---

## 3. The plan, step by step

### Phase 1 — Ground truth on the cluster

> **In plain English:** Before writing anything that assumes how BigPurple works, log in and *ask it*. Which partitions have A100s, which CUDA modules exist, where your lab's storage is, and how much quota you have. Everything after this depends on those four answers.

1. Run `scripts/00_probe_cluster.sh` on a login node.
2. It reports: GPU partitions (`sinfo`), available CUDA/conda modules (`module avail`), your SLURM account (`sacctmgr`), and free space on `/gpfs`.
3. **Paste the output back.** We fill `configs/cluster.yaml` with real values — partition name, account, and the `/gpfs/data/<lab>` path. Until then that file holds placeholders that will fail loudly rather than silently.

**Exit criteria:** `configs/cluster.yaml` contains no `CHANGEME`.

---

### Phase 2 — Two environments, built once

> **In plain English:** vLLM pins its own exact torch version and fights with anything else. So we build **two separate conda environments** instead of one that half-works: one that only serves models, one that only does batch/dev work. This is the single most common way these setups break, and keeping them apart avoids it entirely.

1. `scripts/01_setup_env.sh serve` → env `ibdp-serve`: `vllm==0.19.1` (pulls its own `torch==2.11.0`). Nothing else.
2. `scripts/01_setup_env.sh batch` → env `ibdp-batch`: `transformers>=4.57`, `qwen-vl-utils==0.0.14`, `torchcodec`, `pymupdf`, `pydantic`.
3. Python **3.11 or 3.12**. (vLLM requires `>=3.10,<3.14`; your Mac's 3.14 is too new — irrelevant on the cluster, but don't mirror it.)
4. Video decoding uses **torchcodec**, not decord: `FORCE_QWENVL_VIDEO_READER=torchcodec`. Qwen now recommends against decord — it hangs on decode and is unmaintained.

**Exit criteria:** both envs import cleanly; `python -c "import vllm"` and `import transformers, qwen_vl_utils` succeed.

---

### Phase 3 — Stage the weights (PHI-critical)

> **In plain English:** Compute nodes have no internet, so the models must be downloaded *first*, from a login node, onto shared storage. After that we flip HuggingFace into hard offline mode — which is also the safety property we want: a model that cannot reach the network cannot leak a patient note to one.

1. On a **login node** (has internet), run `scripts/02_fetch_models.sh`.
2. It `hf download`s each model into `$IBDP_MODELS` on `/gpfs` (~18 GB for VL-8B, ~16 GB for Qwen3-8B; budget ~100 GB if you also take the 32B).
3. Every downstream script exports **`HF_HUB_OFFLINE=1`** and loads models **by local path**, not by repo id.
4. Storage lives on `/gpfs`, never in `$HOME` (small quota) and never in git.

**Exit criteria:** model dirs exist on `/gpfs`; a compute node loads one with networking unavailable.

---

### Phase 4 — The vLLM server

> **In plain English:** One sbatch script starts vLLM on a GPU node and serves an OpenAI-compatible API. You then open an SSH tunnel so that, from your laptop, the cluster's GPU looks like `localhost:8000`. Your notebooks talk to it with the ordinary OpenAI client library — but nothing ever leaves the cluster.

1. `sbatch scripts/03_serve.sbatch vl` (or `text`) — reads `configs/models.yaml`, pins `--constraint=a100`.
2. Server flags that matter:
   - `--allowed-local-media-path $IBDP_MEDIA` — **required** to pass `file://` videos/images. This is the PHI-safe path: media is read off local disk, never uploaded. Scope this directory tightly; everything under it becomes server-readable.
   - `--limit-mm-per-prompt.video 1 --limit-mm-per-prompt.image 8` — per-prompt caps. Setting `.video 0` on a text/image-only server measurably cuts profiling memory.
   - `--max-model-len 128000` — the default 262K reserves an enormous KV cache. 128K is the recipe's recommendation.
   - `--media-io-kwargs '{"video": {"num_frames": -1}}'` — decode all frames, let the processor sample.
3. The job writes its node + port to `logs/server.json`; `scripts/04_tunnel.sh` reads that and opens the tunnel for you.

**Exit criteria:** `curl localhost:8000/v1/models` through the tunnel returns the model.

---

### Phase 5 — Client library + the three tasks

> **In plain English:** A thin Python layer so day-to-day code says `describe_video(path, prompt)` instead of hand-building JSON. Three entry points, one per thing you said you need.

**`tasks/video.py` — video clip → text.** The headline feature.
- Sends `{"type": "video_url", "video_url": {"url": "file:///..."}}`.
- 🐛 **Always pass `fps` alongside `num_frames`.** A live vLLM bug (issue #35909) throws `AssertionError: The timestamps length should be equal video length` when `num_frames` is passed alone — the processor assumes fps=2 and the reconstructed timestamps desync. Our client enforces this invariant so you can't hit it.
- ⚠️ Frame-sampling params are **launch-time**, not fully per-request, in online serving. For reproducibility we support a client-side pre-sampling mode that sends pre-extracted JPEG frames instead.

**`tasks/document.py` — PDF / image → structured fields.**
- Render pages with **PyMuPDF** at ~200 DPI (*not* pdf2image — that needs a Poppler system binary you cannot `apt install` on an HPC node).
- **Text-native PDFs skip the VLM entirely:** pull the text layer directly, and only route scanned/image pages through Qwen3-VL. Cheaper *and* more accurate.
- Output validated against a Pydantic schema.

**`tasks/text.py` — reasoning over notes.** Plain chat against the text server, `enable_thinking` toggle exposed.

**Memory knob to remember:** vision tokens ≈ `total_pixels / (32*32)`. When a long video OOMs, **lower `total_pixels` before you lower `fps`** — you keep temporal coverage and drop spatial detail, which is usually the right trade.

**Exit criteria:** each task runs end-to-end against synthetic data in `examples/`.

---

### Phase 6 — Batch + RAG

> **In plain English:** Once single calls work, make it scale. A SLURM array job fans a folder of clips or PDFs across GPUs, checkpoints as it goes (so a preempted job resumes instead of restarting), and writes one JSON per input.

1. `scripts/05_batch.sbatch` — array job; each task takes a shard of the manifest.
2. `batch/run.py` — resumable (skips outputs that already exist), writes JSONL, logs failures rather than dying.
3. `rag/` — chunk + embed notes locally, retrieve, generate with the text server. Local embedding model, no external calls.

**Exit criteria:** a 50-item batch completes with a failure report and no PHI in any log.

---

## 4. PHI safety rules (non-negotiable)

1. **`.gitignore` blocks data by default** — `data/`, `outputs/`, and every media/document extension. Only `examples/assets/` is opted back in. Verify with `git status` before every push.
2. **`HF_HUB_OFFLINE=1`** everywhere. No model call can reach the network.
3. **No external APIs.** The OpenAI client library is used, but it points at *your* vLLM on *your* node. No key ever leaves the cluster because there is no key.
4. **PHI never enters a prompt that gets logged.** Log IDs and metrics, not content.
5. **`--allowed-local-media-path` is scoped to one directory**, not `/`.
6. Third-party PDF/OCR repos referenced during research were **not audited** — do not adopt them for PHI without review.

---

## 5. Pinned versions

```
# serve env
vllm==0.19.1              # min 0.11.0 for Qwen3-VL; brings torch==2.11.0

# batch env
transformers>=4.57.0      # first release with Qwen3-VL
qwen-vl-utils==0.0.14
torchcodec                # video backend; NOT decord
pymupdf                   # PDF render; NOT pdf2image (no Poppler on HPC)
pydantic>=2
```

**Environment:** `FORCE_QWENVL_VIDEO_READER=torchcodec`, `HF_HUB_OFFLINE=1`.

---

## 6. Known issues we are designing around

| Issue | Impact | Our mitigation |
|---|---|---|
| vLLM #35909 — `num_frames` without `fps` → timestamp assertion | Video calls crash | Client always sends both |
| FP8 on Ampere → Marlin W8A16 fallback, no speedup; MoE coverage incomplete | Silent perf loss | Use bf16; AWQ if memory-bound |
| Per-request frame control limited in online serving | Can't tune fps per call | Optional client-side frame pre-sampling |
| vLLM's frame extraction ≠ `qwen_vl_utils`'s | Server and batch paths disagree | Pre-sample client-side when reproducibility matters |
| Speculative decoding incompatible with Qwen3-VL | n/a | Don't enable it |

---

## 7. Open questions for you

1. **Partition / account / lab storage path** on BigPurple → Phase 1 answers this.
2. **What does the video task actually output?** Free-text description, or a structured score (Mayo endoscopic subscore, UCEIS)? A structured target changes the prompts and the Pydantic schema, and makes the results gradeable.
3. **Clip length and volume** — 30-second clips or 20-minute procedures? Ten of them or ten thousand? This sets `total_pixels`, `max-model-len`, and whether we need TP2.
4. **Is there ground truth** to evaluate against? If yes, we add an eval harness early rather than bolting one on later.
