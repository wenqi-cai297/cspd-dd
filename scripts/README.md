# Server helper scripts

These scripts are meant to reduce repeated manual CLI typing on the Linux GPU server.

## Recommended Prep + Stage 1 order

Prep now means `classes.json` generation plus `class -> archetype` mapping.
Stage 1 now means attribute extraction, normalization, and canonical render.

If you want the full workflow from environment checking to final canonical render, use these steps in order:

### 1. Create the conda environment from the repo and verify it

```bash
bash scripts/stage1/check_stage1_env.sh
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
bash scripts/prep/prepare_stage1_metadata.sh /path/to/classes.py /path/to/class_to_archetype.json IMAGENET2012_CLASSES
```

If you already have a JSON mapping file instead of a Python file:

```bash
bash scripts/prep/prepare_stage1_metadata.sh /path/to/classes.json /path/to/class_to_archetype.json
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
python scripts/prep/generate_class_to_archetype_map_vlm.py \
  --input /path/to/classes.json \
  --dataset-root /path/to/imagefolder_dataset \
  --output /path/to/class_to_archetype.json \
  --detail-output /path/to/class_to_archetype_details.jsonl \
  --taxonomy configs/stage1/archetype_taxonomy_manual.json \
  --images-per-class 5
```

Or use the helper script (it uses the repo-bundled `classes.json` by default):

```bash
bash scripts/prep/generate_class_to_archetype_vlm.sh /path/to/imagefolder_dataset 5
```

### 3. Run the full Stage 1 pipeline end-to-end

```bash
bash scripts/stage1/run_stage1_pipeline.sh /path/to/dataset_root [backend]
```

Performs: Stage 1A extraction → Stage 1B normalization (with inline VLM review) → Stage 1C render. Assumes Prep metadata (`classes.json`, `class_to_archetype.json`) is already in place.

## Individual helper scripts

### Install / refresh the project in the shared conda environment

```bash
bash scripts/stage1/setup_cspd_stage1.sh
```

This script:
- activates `cspd-dd`
- runs `pip install -e .`
- checks that `cspd-stage1` is available

### Run Stage 1 with the real local Qwen backend

```bash
bash scripts/stage1/run_stage1_qwen_local.sh /path/to/dataset [max_new_tokens] [class_name_map|DEFAULT] [flush_every] [class_archetype_map]
```

If `class_name_map` is omitted or passed as `DEFAULT`, the script uses the repo-bundled `classes.json` automatically.

Example:

```bash
bash scripts/stage1/run_stage1_qwen_local.sh /data/cifar10_small 256
bash scripts/stage1/run_stage1_qwen_local.sh /data/imagenette/train 256 DEFAULT 10 configs/stage1/class_to_archetype_imagenet1k_manual.json
```

The output directory is generated automatically as:

```text
runs/stage1/attributes/<dataset_name>/qwen_local/<timestamp>
```

### Run Stage 1 normalization

This now runs deterministic normalization first, then inline constrained VLM review for ambiguous slots by default.

```bash
bash scripts/stage1/run_stage1_normalization.sh /path/to/attribute_run_dir
```

You can also pass the `attributes.jsonl` path directly:

```bash
bash scripts/stage1/run_stage1_normalization.sh /path/to/attribute_run_dir/attributes.jsonl
```

The output directory is generated automatically as:

```text
<attribute_run_dir>/normalization/<timestamp>
```

To disable inline review manually:

```bash
bash scripts/stage1/run_stage1_normalization.sh /path/to/attribute_run_dir --disable-vlm-review
```

To override the inline review backend explicitly:

```bash
bash scripts/stage1/run_stage1_normalization.sh /path/to/attribute_run_dir qwen_local
```

The main normalized JSONL now keeps both deterministic `normalized_attributes` and `effective_normalized_attributes` plus per-slot `vlm_review` metadata; Stage 1 render prefers the effective attributes when present.


The output directory is generated automatically as:

```text
<normalization_dir>/review_vlm/<timestamp>
```

### Run Stage 1 canonical rendering

```bash
bash scripts/stage1/run_stage1_render.sh /path/to/attributes_normalized.jsonl [renderer_version]
```

Example:

```bash
bash scripts/stage1/run_stage1_render.sh runs/stage1/attributes/ImageNette/qwen_local/2026-03-26_183111/normalization/2026-03-28_180021/attributes_normalized.jsonl
```

The output directory is generated automatically as:

```text
runs/stage1/render/<dataset_name>/<backend>/<timestamp>
```

Migration note:
- Canonical render code now lives under `src/cspd_stage1/`.
- Use `bash scripts/stage1/run_stage1_render.sh ...` or `cspd-stage1 render ...`.
- The old Stage 2 render compatibility entrypoints were removed because future Stage 2 will be different code.

### Run Stage 2 v1 training scaffold

Stage 2 now means generative-backbone adaptation / canonical-semantic-space familiarization.
It consumes:
- an ImageFolder dataset root as the visual source
- a Stage 1 render `records.jsonl` file as the canonical text-conditioning source

Recommended helper:

```bash
bash scripts/stage2/run_stage2_train.sh /path/to/dataset_root /path/to/stage1_render_records.jsonl
```

For the SDXL official-diffusers path specifically, first check the environment and script resolution:

```bash
bash scripts/stage2/check_stage2_sdxl_env.sh
# or point it explicitly at the diffusers example script
bash scripts/stage2/check_stage2_sdxl_env.sh /path/to/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py
```

Example:

```bash
bash scripts/stage2/run_stage2_train.sh \
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

For SDXL runs in the same script-first style, use:

```bash
bash scripts/stage2/check_stage2_sdxl_env.sh
export DIFFUSERS_REPO_ROOT=/path/to/diffusers   # or export CSPD_STAGE2_SDXL_SCRIPT=/path/to/train_text_to_image_lora_sdxl.py
STAGE2_NUM_PROCESSES=2 bash scripts/stage2/run_sdxl_stage2_official.sh \
  /data/imagenette/train \
  runs/stage1/render/imagenette/qwen_local/2026-04-02_010203/records.jsonl \
  8 \
  9
```

This helper:
- activates the shared conda env
- runs the dedicated SDXL environment check first
- resolves the official diffusers script from `--sdxl-official-script`, `CSPD_STAGE2_SDXL_SCRIPT`, `DIFFUSERS_REPO_ROOT/examples/text_to_image/`, `DIFFUSERS_HOME/examples/text_to_image/`, or `PATH`
- writes an explicit SDXL launch preflight into `sdxl_official_launch_plan.json` before any long training launch
- keeps the run output under `runs/stage2/train/<dataset_label>/stabilityai__stable-diffusion-xl-base-1.0/<timestamp>/` unless `STAGE2_OUTPUT_DIR` is set

Mainline training config (produces the 63.27% IPC=10 baseline): rank=64 LoRA, cosine LR with 500-step warmup, noise_offset=0.05, snr_gamma=5.0, batch=8, epoch 9 on 2 GPUs at 512 resolution.

To inspect SDXL module names before choosing trainable-component groups:

```bash
bash scripts/stage2/dump_stage2_backbone_modules.sh \
  stabilityai/stable-diffusion-xl-base-1.0 \
  unet \
  --local-files-only
```

This writes run artifacts under `runs/stage2/inspect/<backbone_slug>/<timestamp>/`.

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
