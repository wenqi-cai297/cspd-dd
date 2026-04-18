# CSPD-DD

Minimal executable scaffold for **Prep** plus **Stage 1** in the CSPD-DD pipeline.

- **Prep** = `classes.json` generation + `class -> archetype` mapping
- **Stage 1** = attribute extraction + normalization + canonical semantic rendering

## Current repo scope

### Prep

- `classes.json` generation from Python or JSON class maps
- Fixed or VLM-assisted `class -> archetype` mapping
- Manually curated archetype taxonomy config under `configs/stage1/`
- Server helpers that materialize prep artifacts under `runs/prep/...`

### Stage 1

- Unified attribute schema
- Direct input from an **ImageFolder-style dataset root**
- Pluggable VLM client interface
- `mock` backend for local pipeline plumbing tests
- `qwen_local` backend for real local GPU inference with Qwen2.5-VL
- Prompting uses an explicit JSON template, plus a narrow fallback parser for bullet-style pseudo-JSON outputs
- Optional class-name mapping for synset-style datasets such as ImageNette / ImageNet subsets
- Class-adaptive slot schemas chosen from the class semantic archetype
- Retry + validation + failure logging
- CLI progress bar with success / failure counters and current sample summary
- Incremental JSONL flushing so partial results are visible on disk during long runs
- Conservative normalization helper for `attributes.jsonl` outputs
- Deterministic canonical semantic rendering from normalized Stage 1 records

### Main CLI entrypoints

- `cspd-stage1 run --dataset-root ... --output-dir ...`
- `cspd-stage1 render --input ... --output-dir ...`
- `cspd-stage2 train --dataset-root ... --render-input ... [--output-dir ...]`
- Canonical Stage 1 render implementation now lives under `src/cspd_stage1/`
- Stage 2 now means generative-backbone adaptation / canonical-semantic-space familiarization; it no longer refers to render

### Main server helper scripts

- `bash scripts/stage2/check_stage2_sdxl_env.sh [optional_explicit_sdxl_script_path]`
- `bash scripts/stage2/run_sdxl_stage2_official.sh ...`
- `bash scripts/prep/prepare_stage1_metadata.sh ...`
- `bash scripts/stage1/run_stage1_pipeline.sh ...`
- `bash scripts/stage1/run_stage1_qwen_local.sh ...`
- `bash scripts/stage1/run_stage1_normalization.sh ...`
- `bash scripts/stage1/run_stage1_render.sh ...`
- `bash scripts/stage2/run_stage2_train.sh ...`

### Default server-side output roots

- Prep metadata: `runs/prep/...`
- Stage 1 attributes: `runs/stage1/attributes/<dataset_name>/<backend>/<timestamp>`
- Stage 1 render: `runs/stage1/render/<dataset_name>/<backend>/<timestamp>`
- Stage 2 train scaffold: `runs/stage2/train/<dataset_label>/<backbone>/<timestamp>`
- Stage 2 SDXL official wrapper materializes a diffusers imagefolder dataset under each run at `sdxl_materialized_dataset/` with `metadata.jsonl` and copied training images
  - default dataset label is the dataset-root basename, except split-only roots like `.../train` become `<parent>_train` (same for `val`/`valid`/`validation`/`test`/`testing`)
  - optional override: set `STAGE2_DATASET_LABEL=...` before `bash scripts/stage2/run_stage2_train.sh ...`

## Expected dataset layout

Stage 1 assumes a simple ImageFolder layout:

```text
dataset_root/
  class_a/
    img1.jpg
    img2.jpg
  class_b/
    img3.jpg
    img4.png
```

Notes:
- each immediate subdirectory under `dataset_root` is treated as one class
- class ids are assigned by sorting class directory names alphabetically
- images are discovered recursively inside each class folder
- supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

## Environment setup

### Recommended: conda on Linux GPU servers

```bash
conda env create -f environment.yml
conda activate cspd-dd
bash scripts/stage1/check_stage1_env.sh
```

This is the intended one-command environment bootstrap for new servers.

### Alternative: local venv / manual Python environment

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .
```

Run with mock backend:

```bash
cspd-stage1 run --dataset-root path/to/dataset --output-dir runs/stage1_mock --backend mock
```

## Real local VLM backend

The repository includes a real local backend:
- backend name: `qwen_local`
- default model: `Qwen/Qwen2.5-VL-7B-Instruct`

Example attribute extraction run on the Linux GPU server:

```bash
cspd-stage1 run \
  --dataset-root /path/to/imagefolder_dataset \
  --output-dir runs/stage1/attributes/my_dataset/qwen_local/2026-03-25_170000 \
  --backend qwen_local \
  --model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --torch-dtype float16 \
  --device-map auto \
  --max-new-tokens 256
```

Then normalize + render:

```bash
python scripts/stage1/normalize_stage1_attributes.py \
  --input runs/stage1/attributes/my_dataset/qwen_local/2026-03-25_170000/attributes.jsonl \
  --output-dir runs/stage1/attributes/my_dataset/qwen_local/2026-03-25_170000/normalization/2026-03-25_180000

cspd-stage1 render \
  --input runs/stage1/attributes/my_dataset/qwen_local/2026-03-25_170000/normalization/2026-03-25_180000/attributes_normalized.jsonl \
  --output-dir runs/stage1/render/my_dataset/qwen_local/2026-03-25_181500
```

Then build the Stage 2 canonical-caption-conditioned training run from real images + Stage 1 canonical captions.
For routine server usage, prefer the helper so the output directory stays under the structured convention `runs/stage2/train/<dataset_label>/<backbone_slug>/<timestamp>`:

```bash
STAGE2_NUM_PROCESSES=2 bash scripts/stage2/run_stage2_train.sh \
  /path/to/imagefolder_dataset \
  runs/stage1/render/my_dataset/qwen_local/2026-03-25_181500/records.jsonl \
  black-forest-labs/FLUX.1-Kontext-dev \
  4 \
  1 \
  --gradient-accumulation-steps 1
```

If you intentionally need direct CLI control, `--output-dir` is now optional there too. When omitted, the CLI derives the same structured path as the helper: `runs/stage2/train/<dataset_label>/<backbone_slug>/<timestamp>`.

Recommended direct CLI form:

```bash
accelerate launch --num_processes 2 \
  -m cspd_stage2.cli train \
  --dataset-root /path/to/imagefolder_dataset \
  --render-input runs/stage1/render/my_dataset/qwen_local/2026-03-25_181500/records.jsonl \
  --backbone-name black-forest-labs/FLUX.1-Kontext-dev \
  --trainable-component-group full_transformer \
  --batch-size 4 \
  --epochs 1 \
  --gradient-accumulation-steps 1
```

If you still want to pin a custom run directory manually, `--output-dir ...` continues to override the derived default.

If you hit memory pressure, you now have two honest fallback levels while keeping the same top-level/fine-grained selector interface. First, you can stay in real-parameter mode but restrict training to conditioning-focused transformer internals:

```bash
accelerate launch --num_processes 2 \
  -m cspd_stage2.cli train \
  --dataset-root /path/to/imagefolder_dataset \
  --render-input runs/stage1/render/my_dataset/qwen_local/2026-03-25_181500/records.jsonl \
  --backbone-name black-forest-labs/FLUX.1-Kontext-dev \
  --trainable-component-group conditioning_transformer \
  --batch-size 4 \
  --epochs 1 \
  --gradient-accumulation-steps 1
```

Or switch to the new real LoRA path, which injects trainable adapters into the selected conditioning-related transformer linear layers while freezing the base weights:

```bash
accelerate launch --num_processes 2 \
  -m cspd_stage2.cli train \
  --dataset-root /path/to/imagefolder_dataset \
  --render-input runs/stage1/render/my_dataset/qwen_local/2026-03-25_181500/records.jsonl \
  --backbone-name black-forest-labs/FLUX.1-Kontext-dev \
  --training-parameterization lora \
  --trainable-component-group conditioning_transformer \
  --adapter-rank 16 \
  --adapter-alpha 16 \
  --batch-size 4 \
  --epochs 1 \
  --gradient-accumulation-steps 1
```

PixArt-Σ is now also wired as a real Stage 2 path for the same `(image, canonical_caption)` objective. This stays in the text-to-image generative framing: real images are VAE-encoded to latents, canonical captions are encoded with the PixArt text stack, and the PixArt transformer is adapted to model those caption-conditioned latents. It is not an image-editing path.

Recommended current practical baseline: PixArt-Σ dual-GPU **full-transformer LoRA** at 512 resolution. The PixArt LoRA path now keeps the frozen base transformer/text/VAE runtime unchanged, but promotes the injected LoRA adapter weights to FP32 by default before optimizer setup so AdamW state and the first update no longer run in float16. The adapter forward path still casts inputs/outputs conservatively at the module boundary, and Stage 2 keeps the explicit post-step trainable-parameter finiteness check so the run fails immediately if an optimizer update corrupts trainable weights.

Proven stable baseline command:

```bash
accelerate launch --num_processes 2 \
  -m cspd_stage2.cli train \
  --dataset-root /path/to/imagefolder_dataset \
  --render-input runs/stage1/render/my_dataset/qwen_local/2026-03-25_181500/records.jsonl \
  --backbone-name PixArt-alpha/PixArt-Sigma-XL-2-512-MS \
  --resolution 512 \
  --backbone-torch-dtype float16 \
  --training-parameterization lora \
  --trainable-component-group full_transformer \
  --adapter-rank 64 \
  --adapter-alpha 64 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5 \
  --lr-scheduler constant_with_warmup \
  --lr-warmup-steps 1000 \
  --max-grad-norm 0.01 \
  --adam-weight-decay 0.0 \
  --pixart-sigma-prompt-dropout-prob 0.1 \
  --epochs 1
```

If GPU memory headroom remains comfortable on the target server, the first reasonable throughput/stability tuning step is to try the same command with `--batch-size 2`.

Stage 2 also now supports optional Weights & Biases logging without making `wandb` a hard dependency. Enable it only when you actually want remote experiment tracking and the package is installed in the runtime environment. Example PixArt run with scalar logging plus periodic sample uploads:

```bash
accelerate launch --num_processes 2 \
  -m cspd_stage2.cli train \
  --dataset-root /path/to/imagefolder_dataset \
  --render-input runs/stage1/render/my_dataset/qwen_local/2026-03-25_181500/records.jsonl \
  --backbone-name PixArt-alpha/PixArt-Sigma-XL-2-512-MS \
  --resolution 512 \
  --backbone-torch-dtype float16 \
  --training-parameterization lora \
  --trainable-component-group full_transformer \
  --adapter-rank 64 \
  --adapter-alpha 64 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5 \
  --lr-scheduler constant_with_warmup \
  --lr-warmup-steps 1000 \
  --max-grad-norm 0.01 \
  --adam-weight-decay 0.0 \
  --pixart-sigma-prompt-dropout-prob 0.1 \
  --wandb \
  --wandb-project cspd-stage2 \
  --wandb-run-name imagenette_pixart_stage2 \
  --wandb-tag pixart \
  --wandb-tag stage2 \
  --sample-every 100 \
  --sample-prompt-file configs/stage2/sample_prompts_imagenette.txt \
  --sample-num-inference-steps 50 \
  --sample-guidance-scale 7 \
  --epochs 1
```

When W&B is enabled, the Stage 2 trainer logs loss / lr / gradient norm / trainable-parameter diagnostics / dtype counts / CUDA memory stats (when available) on the main process. For PixArt runs, `--sample-every` triggers local sample generation under `runs/stage2/train/.../samples/step_<step>/` and uploads those images to W&B. Prompt sources are resolved in this order: `--sample-prompt-file`, repeated `--sample-prompt`, then the first paired Stage 1 canonical captions if no explicit prompts were provided. The default training-time PixArt sampling knobs now match a more evaluation-like comparison setup: `--sample-num-inference-steps 50` and `--sample-guidance-scale 7`.

For detached comparison against step-0 / periodic training-path samples, you can now run standalone pretrained PixArt sampling with the same prompt-file flow:

```bash
bash scripts/stage2/run_pixart_stage2_baseline_sampling.sh
```

Direct CLI equivalent:

```bash
python -m cspd_stage2.cli sample-baseline \
  --dataset-root /path/to/imagefolder_dataset \
  --backbone-name PixArt-alpha/PixArt-Sigma-XL-2-512-MS \
  --sample-prompt-file configs/stage2/sample_prompts_imagenette.txt \
  --sample-num-prompts 8 \
  --sample-num-inference-steps 50 \
  --sample-guidance-scale 7
```

Standalone baseline outputs default to `runs/stage2/baseline_samples/<dataset_label>/<backbone_slug>/<timestamp>/`, with images saved under `samples/step_000000/` plus a `baseline_sampling_summary.json` metadata file.

Stage 2 no longer supports prompt-cache preprocessing or cached prompt/text embeddings. Training always runs live prompt encoding on the active backbone path during each step. For PixArt-Σ, the live prompt path keeps the PixArt-family prompt length consistent at 300 tokens, and the Stage 2 PixArt path now defaults closer to the official Sigma recipe: low LR, constant-with-warmup scheduling, tight grad clipping, and 0.1 prompt dropout on the canonical-caption conditioning stream. In LoRA mode, PixArt now defaults to FP32 adapter master/update weights even when the loaded backbone stays float16; if you explicitly want the older all-float16 LoRA adapter path, pass `--disable-lora-fp32-for-pixart`. The full-parameter PixArt path still keeps the safer FP32 trainable-parameter option too; disable that older override only with `--disable-full-update-fp32-for-pixart`.

For remote/server debugging, Stage 2 now emits concise rank-aware progress logs directly to stdout/stderr around the common stall points: backbone load, module freezing/selection, dataloader creation, each `accelerate.prepare(...)` boundary, first batch fetch, first text/VAE encode, first forward/backward/optimizer step, checkpoint writes, explicit non-finite loss detection, and early-step gradient diagnostics. It also writes per-rank JSONL diagnostics under the run directory, typically:
- `runs/stage2/train/.../rank00_memory_diagnostics.jsonl`
- `runs/stage2/train/.../rank01_memory_diagnostics.jsonl` (and so on for multi-process runs)
- `runs/stage2/train/.../training_metrics.json`

When a server run looks stuck, tail the launcher log for `[Stage2]` lines first, then inspect the latest `rank*_memory_diagnostics.jsonl` file to see the last completed phase on each rank.

Recommended 2-GPU PixArt conditioning-focused fallback command:

```bash
accelerate launch --num_processes 2 \
  -m cspd_stage2.cli train \
  --dataset-root /path/to/imagefolder_dataset \
  --render-input runs/stage1/render/my_dataset/qwen_local/2026-03-25_181500/records.jsonl \
  --backbone-name PixArt-alpha/PixArt-Sigma-XL-2-512-MS \
  --resolution 512 \
  --backbone-torch-dtype float16 \
  --training-parameterization full \
  --trainable-component-group conditioning_transformer \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5 \
  --lr-scheduler constant_with_warmup \
  --lr-warmup-steps 1000 \
  --max-grad-norm 0.01 \
  --adam-weight-decay 0.0 \
  --pixart-sigma-prompt-dropout-prob 0.1 \
  --epochs 1
```

`conditioning_transformer` resolves to conditioning-related transformer internals around `context_embedder`, `time_text_embed*`, `transformer_blocks.*.norm1_context*`, `transformer_blocks.*.attn.add_{q,k,v}_proj`, `transformer_blocks.*.attn.to_add_out`, and `ff_context*`. You can also combine finer groups such as `conditioning_context_embedder`, `conditioning_time_text_embed`, `conditioning_norm1_context`, `conditioning_added_kv_attention`, and `conditioning_ff_context`. In LoRA mode, these selectors define where adapters are injected; in full mode, they define which real parameters stay trainable. On PixArt, the recommended path is still the dual-GPU `full_transformer` LoRA route; if that does not fit or needs further narrowing, `conditioning_transformer` remains the smaller fallback. The full-parameter PixArt path also keeps the safer boundary-aware FP32 update strategy for targeted real-parameter training.

This Stage 2 CLI now implements pairing / manifest generation / run-directory setup plus minimal `accelerate`-managed real FLUX/PixArt training paths. The broad default policy still freezes non-transformer top-level modules and operates inside the transformer, but for PixArt-Σ the currently recommended practical route is the conservative dual-GPU `full_transformer` LoRA command above, with `conditioning_transformer` as the narrower fallback and full real-parameter updates retained as a targeted comparison/debug path. The runtime still attempts transformer gradient checkpointing when supported, but frozen VAE/text components now stay on the active training device instead of shuttling between CPU and GPU. The current implementation honestly remains limited: it uses `Accelerator` for process setup / dataloader sharding / backward / unwrap-model checkpointing, and the LoRA path is a conservative linear-layer injection path rather than a full PEFT feature-complete stack; it still does **not** implement heavier state-sharding/FSDP-style offload.

During real Stage 2 training, each process now also writes a per-rank GPU memory diagnostics artifact under the run directory, named like `rank00_memory_diagnostics.jsonl`. These JSONL events log rank / device identity plus current and peak CUDA allocated/reserved memory around key phases such as backbone load, module freeze/selection, gradient-checkpointing setup, frozen-component device setup, accelerate prepare, VAE encode, prompt encode, transformer forward, loss, backward, non-finite failure events, and gradient diagnostics.

The train CLI still exposes `--disable-gradient-checkpointing` and `--disable-full-update-fp32-for-pixart` if you need to back away from the safer defaults during debugging.

Stage 2 now also has a real diffusers-backed backbone load path for inspection when the environment actually supports it. Example:

```bash
cspd-stage2 inspect-targets \
  --backbone-name black-forest-labs/FLUX.1-Kontext-dev \
  --load-backbone \
  --local-files-only
```

If the model weights are not cached locally, or Hugging Face access/downloads are unavailable, the command reports the real runtime failure instead of pretending the backbone was loaded.

When you want to inspect the model's own module names before choosing trainable component groups, use the new dump command:

```bash
cspd-stage2 dump-modules \
  --backbone-name black-forest-labs/FLUX.1-Kontext-dev \
  --load-backbone \
  --component transformer \
  --output-dir runs/stage2/inspect/flux_dev/manual_dump \
  --local-files-only
```

This writes:
- `pipeline_top_level_components.txt` for explicit pipeline-level components (for example `transformer`, `vae`, `text_encoder` when exposed)
- `pipeline_named_children.txt` for the raw pipeline `named_children()` view
- `<component>_named_children.txt` for the selected focus module's direct children
- `<component>_named_modules.txt` for the full focus-module tree
- `filtered/keyword_*.txt` for quick keyword-focused review (for example `context`, `embed`, `attn`, `proj`)
- `dump_summary.json` for the artifact index and counts

If you use the provided shell helpers, the workflow can be driven end-to-end from prep through final Stage 1 canonical render, then into Stage 2 run scaffolding. The full workflow script uses only a small mock smoke subset by default (first 3 classes, first 10 images per class), and also supports `--skip-smoke`.

For routine ImageNet-1k / Imagenette reruns, you usually do not need to rerun Prep if you are happy using the repo-bundled `classes.json` plus `configs/stage1/class_to_archetype_imagenet1k_manual.json`.

Useful extraction options:
- `--disable-fast-processor`: use the slower processor path if the fast processor behaves oddly
- `--no-raw-response`: skip saving raw model text in success rows
- `--class-name-map /path/to/classes.json`: map raw folder labels such as `n01440764` to readable class names
- `--class-archetype-map /path/to/class_to_archetype.json`: freeze raw folder labels to explicit archetypes before slot schema selection
- resume is enabled by default when reusing an output directory; previously successful samples are skipped, while samples recorded in `failed_samples.jsonl` are retried. Use `--no-resume` to force overwrite/restart

## Prep taxonomy configuration

Prep now prefers a manually fixed taxonomy configuration instead of VLM-generated taxonomy discovery.

Primary taxonomy file:

```text
configs/stage1/archetype_taxonomy_manual.json
```

This file defines the manually curated archetype set used during Prep and Stage 1 schema decisions.

Bundled fixed ImageNet-1k class-to-archetype mapping:

```text
configs/stage1/class_to_archetype_imagenet1k_manual.json
```

This lets ImageNet-1k / Imagenette-style reruns skip re-running Prep when you just want a stable repo-bundled `classes.json` + `class_to_archetype.json` pair.

The repo now also bundles a conda environment file at `environment.yml`, so new servers can create the `cspd-dd` environment directly from the repo instead of reconstructing runtime dependencies by hand each time.

If you want Qwen to generate `class -> archetype` mappings, the recommended path is the multimodal class-level mapper:

```bash
python scripts/prep/generate_class_to_archetype_map_vlm.py \
  --input /path/to/classes.json \
  --dataset-root /path/to/imagefolder_dataset \
  --output /path/to/class_to_archetype.json \
  --detail-output /path/to/class_to_archetype_details.jsonl \
  --taxonomy configs/stage1/archetype_taxonomy_manual.json \
  --images-per-class 5
```

Server helper (uses the repo-bundled `classes.json` by default):

```bash
bash scripts/prep/generate_class_to_archetype_vlm.sh /path/to/imagefolder_dataset 5
```

## Stage 1 normalization helper

A conservative post-processing script is included for Stage 1 `attributes.jsonl` outputs. By default it now runs deterministic normalization first, then an inline constrained VLM review pass for only the ambiguous / review-required slots:

```bash
python scripts/stage1/normalize_stage1_attributes.py \
  --input /path/to/attributes.jsonl \
  --output-dir /path/to/normalized_artifacts
```

Disable the inline VLM review if you want a purely deterministic run:

```bash
python scripts/stage1/normalize_stage1_attributes.py \
  --input /path/to/attributes.jsonl \
  --output-dir /path/to/normalized_artifacts \
  --disable-vlm-review
```

Default rules live in:

```text
configs/stage1/normalization/stage1_attribute_normalization_rules.json
```

The script preserves the original row and writes these artifacts:
- `attributes_normalized.jsonl`: original row plus deterministic `normalized_attributes`, deterministic `attribute_normalization`, `effective_normalized_attributes` used by downstream render, and inline `vlm_review` metadata when enabled
- `normalization_audit.jsonl`: changed or review-flagged fields with rule ids/status from the deterministic pass
- `normalization_review_queue.jsonl`: only suspicious / review-required items from the deterministic pass
- `normalization_review_vlm.jsonl`: one structured constrained VLM decision per reviewed ambiguous slot
- `normalization_review_vlm_summary.json`: aggregate counts and contract metadata for the inline review pass
- `normalization_summary.json`: aggregate counts by status / slot / class / rule plus inline VLM review summary
- `normalization_rules_snapshot.json`: exact rule snapshot used for the run

`normalized_attributes` stays as the deterministic output for auditability. `effective_normalized_attributes` applies only the constrained inline VLM decisions (`replace_normalized` / `set_unknown`) and is what Stage 1 render uses by default when present.

## Server shell scripts

See `scripts/README.md` for the recommended order and detailed examples.

## Notes

- `mock` backend is for plumbing tests only.
- `qwen_local` is intended for server-side GPU execution.
- The pipeline enforces a unified schema and writes `unknown` / `not_applicable` / `null` when appropriate.
- Real large-scale runs should still start with a small dataset slice first to inspect speed, failure rate, and output quality.
tput quality.
