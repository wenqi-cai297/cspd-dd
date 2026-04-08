# Server helper scripts

These scripts are meant to reduce repeated manual CLI typing on the Linux GPU server.

## Recommended Prep + Stage 1 order

Prep now means `classes.json` generation plus `class -> archetype` mapping.
Stage 1 now means attribute extraction, normalization, and canonical render.

If you want the full workflow from environment checking to final canonical render, use these steps in order:

### 1. Create the conda environment from the repo and verify it

```bash
bash scripts/server/check_stage1_env.sh
```

This script:
- activates `cspd-dd`
- checks Python / torch / CUDA
- installs missing runtime packages such as `transformers` and `Pillow`
- runs `pip install -e .`
- verifies that `transformers` and `PIL` import correctly

### 2. Prep: materialize `classes.json` and `class_to_archetype.json`

If you start from a Python class mapping file, prepare metadata like this:

```bash
bash scripts/server/prepare_stage1_metadata.sh /path/to/classes.py /path/to/class_to_archetype.json IMAGENET2012_CLASSES
```

If you already have a JSON mapping file instead of a Python file:

```bash
bash scripts/server/prepare_stage1_metadata.sh /path/to/classes.json /path/to/class_to_archetype.json
```

This script:
- converts `classes.py` into `classes.json` when needed
- copies a fixed `class_to_archetype.json` into the prep run directory
- does not run VLM-based taxonomy discovery

For ImageNet-1k / Imagenette-style reruns, the repo now also bundles a fixed mapping you can reuse directly:

```text
configs/stage1/class_to_archetype_imagenet1k_manual.json
```

If you want VLM to produce `class_to_archetype.json`, the recommended path is the multimodal class-level mapper:

```bash
python scripts/data/generate_class_to_archetype_map_vlm.py \
  --input /path/to/classes.json \
  --dataset-root /path/to/imagefolder_dataset \
  --output /path/to/class_to_archetype.json \
  --detail-output /path/to/class_to_archetype_details.jsonl \
  --taxonomy configs/stage1/archetype_taxonomy_manual.json \
  --images-per-class 5
```

Or use the helper script (it uses the repo-bundled `classes.json` by default):

```bash
bash scripts/server/generate_class_to_archetype_vlm.sh /path/to/imagefolder_dataset 5
```

### 3. Run the full Prep + Stage 1 workflow end-to-end

```bash
bash scripts/server/run_stage1_full_workflow.sh \
  /path/to/dataset_root \
  /path/to/classes.py \
  /path/to/class_to_archetype.json \
  IMAGENET2012_CLASSES \
  256 \
  /path/to/sample_image.jpg
```

This script performs the full chain:
1. environment checks
2. Prep: `classes.py -> classes.json`
3. Prep: copy fixed `class_to_archetype.json`
4. Qwen load test
5. single-image inference test
6. small mock smoke run on the first 3 classes with the first 10 images per class
7. Stage 1 attribute extraction
8. Stage 1 normalization
9. Stage 1 canonical render

If you omit the final sample-image argument, the script auto-picks the first image under the dataset root.

## Individual helper scripts

### Install / refresh the project in the shared conda environment

```bash
bash scripts/server/setup_cspd_stage1.sh
```

This script:
- activates `cspd-dd`
- runs `pip install -e .`
- checks that `cspd-stage1` is available

### Run Stage 1 with the real local Qwen backend

```bash
bash scripts/server/run_stage1_qwen_local.sh /path/to/dataset [max_new_tokens] [class_name_map|DEFAULT] [flush_every] [class_archetype_map]
```

If `class_name_map` is omitted or passed as `DEFAULT`, the script uses the repo-bundled `classes.json` automatically.

Example:

```bash
bash scripts/server/run_stage1_qwen_local.sh /data/cifar10_small 256
bash scripts/server/run_stage1_qwen_local.sh /data/imagenette/train 256 DEFAULT 10 configs/stage1/class_to_archetype_imagenet1k_manual.json
```

The output directory is generated automatically as:

```text
runs/stage1/attributes/<dataset_name>/qwen_local/<timestamp>
```

### Run Stage 1 normalization

This now runs deterministic normalization first, then inline constrained VLM review for ambiguous slots by default.

```bash
bash scripts/server/run_stage1_normalization.sh /path/to/attribute_run_dir
```

You can also pass the `attributes.jsonl` path directly:

```bash
bash scripts/server/run_stage1_normalization.sh /path/to/attribute_run_dir/attributes.jsonl
```

The output directory is generated automatically as:

```text
<attribute_run_dir>/normalization/<timestamp>
```

To disable inline review manually:

```bash
bash scripts/server/run_stage1_normalization.sh /path/to/attribute_run_dir --disable-vlm-review
```

To override the inline review backend explicitly:

```bash
bash scripts/server/run_stage1_normalization.sh /path/to/attribute_run_dir qwen_local
```

The main normalized JSONL now keeps both deterministic `normalized_attributes` and `effective_normalized_attributes` plus per-slot `vlm_review` metadata; Stage 1 render prefers the effective attributes when present.

### Run optional Stage 1 normalization-review VLM fallback

This path is only for ambiguous cases already flagged by deterministic normalization (`status=review_required` or non-empty `review_reasons`). It does not replace the main deterministic normalization/render path.

```bash
bash scripts/server/run_stage1_normalization_review_vlm.sh /path/to/attributes_normalized.jsonl [backend] [max_new_tokens]
```

Example:

```bash
bash scripts/server/run_stage1_normalization_review_vlm.sh runs/stage1/attributes/ImageNette/qwen_local/2026-03-26_183111/normalization/2026-03-28_180021/attributes_normalized.jsonl qwen_local 256
```

The output directory is generated automatically as:

```text
<normalization_dir>/review_vlm/<timestamp>
```

### Run Stage 1 canonical rendering

```bash
bash scripts/server/run_stage1_render.sh /path/to/attributes_normalized.jsonl [renderer_version]
```

Example:

```bash
bash scripts/server/run_stage1_render.sh runs/stage1/attributes/ImageNette/qwen_local/2026-03-26_183111/normalization/2026-03-28_180021/attributes_normalized.jsonl
```

The output directory is generated automatically as:

```text
runs/stage1/render/<dataset_name>/<backend>/<timestamp>
```

Migration note:
- Canonical render code now lives under `src/cspd_stage1/`.
- Use `bash scripts/server/run_stage1_render.sh ...` or `cspd-stage1 render ...`.
- The old Stage 2 render compatibility entrypoints were removed because future Stage 2 will be different code.

### Run Stage 2 v1 training scaffold

Stage 2 now means generative-backbone adaptation / canonical-semantic-space familiarization.
It consumes:
- an ImageFolder dataset root as the visual source
- a Stage 1 render `records.jsonl` file as the canonical text-conditioning source

Recommended helper:

```bash
bash scripts/server/run_stage2_train.sh /path/to/dataset_root /path/to/stage1_render_records.jsonl
```

Example:

```bash
bash scripts/server/run_stage2_train.sh \
  /data/imagenette/train \
  runs/stage1/render/imagenette/qwen_local/2026-04-02_010203/records.jsonl
```

This helper currently:
- launches Stage 2 through `accelerate launch` by default for multi-GPU-friendly process semantics
- builds a Stage 2 run directory under `runs/stage2/train/...`
  - default dataset label is the dataset-root basename, except split-only roots like `.../train` become `<parent>_train` (same for `val`/`valid`/`validation`/`test`/`testing`)
  - optional override: set `STAGE2_DATASET_LABEL=...` before invoking the script
- pairs images with Stage 1 canonical captions conservatively by stable identifiers
- writes `train_manifest.jsonl` plus unmatched-record reports
- writes a Stage 2 config snapshot and trainer plan
- records full-transformer fine-tuning intent and module-selection metadata
- keeps non-transformer top-level modules frozen by default (`freeze_text_encoder=true`, `freeze_vae=true`)
- keeps frozen VAE/text modules resident on the active training device for the whole run

You can still pass through extra Stage 2 CLI options after the positional helper arguments. For example:

During real Stage 2 training, watch the helper log for `[Stage2]` progress lines. The run directory also keeps per-rank JSONL diagnostics such as `rank00_memory_diagnostics.jsonl` plus `training_metrics.json`. If a multi-GPU launch appears hung, the quickest check is usually: find the newest run under `runs/stage2/train/...`, then inspect the last `[Stage2]` stdout line and the tail of each `rank*_memory_diagnostics.jsonl` file to see which phase each rank last completed.

```bash
STAGE2_NUM_PROCESSES=2 bash scripts/server/run_stage2_train.sh \
  /data/imagenette/train \
  runs/stage1/render/imagenette/qwen_local/2026-04-02_010203/records.jsonl \
  black-forest-labs/FLUX.1-Kontext-dev \
  4 \
  1 \
  --gradient-accumulation-steps 1 \
  --backbone-local-files-only
```

Use the direct CLI mainly when you intentionally need full argument-level control. `--output-dir` is now optional there too; if you omit it, the CLI derives `runs/stage2/train/<dataset_label>/<backbone_slug>/<timestamp>` with the same dataset-label rule as this helper.

```bash
accelerate launch --num_processes 2 \
  -m cspd_stage2.cli train \
  --dataset-root /path/to/dataset_root \
  --render-input /path/to/stage1_render_records.jsonl \
  --backbone-name black-forest-labs/FLUX.1-Kontext-dev \
  --trainable-component-group full_transformer \
  --batch-size 4 \
  --epochs 1 \
  --gradient-accumulation-steps 1
```

If you want to pin a custom run directory, `--output-dir ...` still overrides the default.

For PixArt-Σ, the recommended next rerun path is full transformer training rather than LoRA. The full PixArt path now upcasts trainable transformer weights to FP32 before optimizer setup, aligns forward inputs to the actual entry-module dtypes instead of blindly casting everything to the first trainable dtype, and performs an immediate post-step trainable-parameter finiteness check, specifically to catch the observed "first optimizer step succeeded, next forward is NaN" failure mode earlier.

```bash
accelerate launch --num_processes 1 \
  -m cspd_stage2.cli train \
  --dataset-root /path/to/dataset_root \
  --render-input /path/to/stage1_render_records.jsonl \
  --backbone-name PixArt-alpha/PixArt-Sigma-XL-2-512-MS \
  --resolution 512 \
  --backbone-torch-dtype float16 \
  --training-parameterization full \
  --trainable-component-group full_transformer \
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

For the memory-reduced conditioning-focused path, keep the same CLI and swap the component group. For PixArt-Σ, this full-parameter `conditioning_transformer` command is now the recommended 2-GPU server fallback when `full_transformer` does not fit:

```bash
accelerate launch --num_processes 2 \
  -m cspd_stage2.cli train \
  --dataset-root /path/to/dataset_root \
  --render-input /path/to/stage1_render_records.jsonl \
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

`conditioning_transformer` resolves to conditioning-related transformer internals around `context_embedder`, `time_text_embed*`, `transformer_blocks.*.norm1_context*`, `transformer_blocks.*.attn.add_{q,k,v}_proj`, `transformer_blocks.*.attn.to_add_out`, and `ff_context*`. You can also compose narrower groups such as `conditioning_context_embedder`, `conditioning_time_text_embed`, `conditioning_norm1_context`, `conditioning_added_kv_attention`, and `conditioning_ff_context`. On PixArt, this path now works with the safer FP32 partial-full-update strategy because frozen entry modules such as `pos_embed` keep their native half-precision boundary, the boundary-sensitive `adaln_single` timestep block also stays at that native dtype, and the remaining selected conditioning modules can still be upcasted to FP32.

Important scope note:
- the pairing/manifest/run scaffold is implemented now
- the recorded default policy is to freeze non-transformer top-level modules and fine-tune the full `FluxTransformer2DModel`
- the real training path now uses `accelerate` for process setup, dataloader preparation, backward, and main-process checkpoint writes
- if memory is insufficient, the intended fallback is conditioning-related transformer submodules only
- the current trainer is still a practical first version rather than a fully optimized FLUX training stack
- use `STAGE2_DISABLE_ACCELERATE=1` only if you intentionally want the older single-process path
- use `--allow-placeholder-loop` only if you want a tiny PyTorch plumbing check rather than real model training

Real backbone-load inspection example:

```bash
cspd-stage2 inspect-targets \
  --backbone-name black-forest-labs/FLUX.1-Kontext-dev \
  --load-backbone \
  --local-files-only
```

If the weights are not present in the local Hugging Face cache, the command now returns an explicit real-load failure instead of a fake loaded module.

When you want to review the backbone's own module names before deciding what to fine-tune, use:

```bash
bash scripts/server/dump_stage2_backbone_modules.sh \
  black-forest-labs/FLUX.1-Kontext-dev \
  transformer \
  --local-files-only
```

This writes run artifacts under `runs/stage2/inspect/<backbone_slug>/<timestamp>/`, including `pipeline_top_level_components.txt` for explicit pipeline-level components, `pipeline_named_children.txt` for the raw top-level child view, the selected component's direct children, the full named-modules dump, keyword-filtered text files, and `dump_summary.json`.

## Dataset assumption

All Stage 1 run scripts assume an ImageFolder-style dataset layout:

```text
dataset_root/
  class_a/
    1.jpg
    2.jpg
  class_b/
    3.jpg
```
