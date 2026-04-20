# CSPD-DD

Full end-to-end pipeline for **class-aware semantic-prompt distillation**: from raw ImageFolder dataset to a distilled training set plus classifier evaluation.

- **Prep** = `classes.json` generation + `class -> archetype` mapping
- **Stage 1** = VLM attribute extraction + normalization + canonical semantic rendering
- **Stage 2** = SDXL LoRA adaptation on real images paired with Stage 1 canonical captions
- **Stage 3** = DINOv2 encoding + per-class HDBSCAN (K-Means fallback) clustering + medoid extraction
- **Stage 4** = distilled-image generation via text2img (optional img2img from medoid) using the Stage 2 LoRA
- **Eval** = ConvNet-6 / ResNet-18 / ResNetAP-10 classifiers trained on the distilled set, scored on the real val set

## Current repo scope

### Prep

- `classes.json` generation from Python or JSON class maps
- Fixed or VLM-assisted `class -> archetype` mapping
- Manually curated archetype taxonomy config under `configs/stage1/`
- Server helpers that materialize prep artifacts under `runs/prep/...`

### Stage 1

- Unified attribute schema
- Direct input from an **ImageFolder-style dataset root**
- Pluggable VLM client interface (`mock` for plumbing tests, `qwen_local` for real local GPU inference with Qwen2.5-VL)
- Explicit JSON prompting template plus narrow fallback parser for bullet-style pseudo-JSON outputs
- Optional class-name mapping for synset-style datasets (ImageNette / ImageNet subsets)
- Class-adaptive slot schemas chosen from the class semantic archetype
- Retry + validation + failure logging; incremental JSONL flushing during long runs
- Deterministic normalization plus inline constrained VLM review for ambiguous slots
- Deterministic canonical semantic rendering from normalized Stage 1 records

### Stage 2

- SDXL LoRA only (PixArt / FLUX / SD 1.5 surface was removed in the Stage 2 cleanup batches)
- Wraps the official `diffusers` `train_text_to_image_lora_sdxl.py` trainer
- Pairs ImageFolder images with Stage 1 canonical captions by stable identifier, materializing a diffusers imagefolder dataset under `sdxl_materialized_dataset/`
- Mainline config (baseline 63.27% on ImageNette IPC=10): rank=64, cosine LR with 500-step warmup, noise_offset=0.05, snr_gamma=5.0, batch=8, epoch 9, 2 GPUs, 512 resolution

### Stage 3

- DINOv2 image encoding (3A), per-class clustering (3B, HDBSCAN with K-Means fallback on small clusters, PCA-optional), medoid + semantic-mode extraction (3C)
- Output modes feed Stage 4 as visual anchors and semantic prompts

### Stage 4

- Distilled-image generation via SDXL text2img (default) or img2img from Stage 3 medoids
- Optional Stage 2 LoRA weights; optional SDXL refiner pass
- Writes distilled images under `runs/stage4/<dataset>/ipc<IPC>/lora/pipeline_<TS>/gen_seed<SEED>/images/`

### Eval

- Trains ConvNet-6, ResNet-18, or ResNetAP-10 classifiers on a Stage 4 distilled dataset
- Reports top-1 / top-5 on the real validation set; supports `EVAL_REPEAT` independent runs per architecture
- Output path mirrors the Stage 4 hierarchy so result JSON is self-identifying

### Main CLI entrypoints

- `cspd-stage1 run --dataset-root ... --output-dir ...`
- `cspd-stage1 render --input ... --output-dir ...`
- `cspd-stage2 train --dataset-root ... --render-input ... [--output-dir ...]`
- `cspd-stage3 encode | cluster | run ...`
- `cspd-stage4 generate ...`
- `cspd-eval run | run-all ...`

### Main server helper scripts

- `bash scripts/prep/prepare_stage1_metadata.sh ...`
- `bash scripts/stage1/run_stage1_pipeline.sh ...` (Stage 1A → 1B → 1C end-to-end)
- `bash scripts/stage2/check_stage2_sdxl_env.sh [optional_explicit_sdxl_script_path]`
- `bash scripts/stage2/run_sdxl_stage2_official.sh <dataset_root> <render_records.jsonl> <batch> <epochs>`
- `bash scripts/stage3/run_stage3_pipeline.sh <dataset_root> <render_records.jsonl> <ipc>`
- `bash scripts/stage4/run_stage4_pipeline.sh <stage3_modes_dir> <stage2_lora_weights|none>`
- `bash scripts/eval/run_eval_pipeline.sh <distilled_dir> <val_dir> <nclass> <ipc> [arch|all]`
- `bash scripts/pipelines/run_full_pipeline.sh <train_root> [val_root] [nclass]` (Stage 1 → Eval, 3×3 protocol by default)

### Default server-side output roots

- Prep metadata: `runs/prep/...`
- Stage 1 attributes: `runs/stage1/attributes/<dataset_name>/<backend>/<timestamp>`
- Stage 1 render: `runs/stage1/render/<dataset_name>/<backend>/<timestamp>`
- Stage 2 train: `runs/stage2/train/<dataset_label>/<backbone>/<timestamp>`
  - default dataset label is the dataset-root basename, except split-only roots like `.../train` become `<parent>_train` (same for `val`/`valid`/`validation`/`test`/`testing`)
  - optional override: set `STAGE2_DATASET_LABEL=...` before invoking the helper
- Stage 3 modes: `runs/stage3/<dataset>/ipc<IPC>/<timestamp>/modes/`
- Stage 4 distilled: `runs/stage4/<dataset>/ipc<IPC>/lora/pipeline_<TS>/gen_seed<SEED>/images/`
- Eval results: `runs/eval/<dataset>/ipc<IPC>/<arch>/<stage4_tag>/<eval_timestamp>/eval_<arch>.json`

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

Then run Stage 2 training. Stage 2 is SDXL LoRA only; the helper wraps the official diffusers trainer and writes to `runs/stage2/train/<dataset_label>/<backbone_slug>/<timestamp>`:

```bash
bash scripts/stage2/run_sdxl_stage2_official.sh \
  /path/to/imagefolder_dataset \
  runs/stage1/render/my_dataset/qwen_local/2026-03-25_181500/records.jsonl \
  8 \
  9
```

The mainline config (which produces the 63.27% IPC=10 baseline) is rank=64 LoRA, cosine LR with 500-step warmup, noise_offset=0.05, snr_gamma=5.0, batch=8, epoch 9 on 2 GPUs at 512 resolution. `--report_to` is deliberately omitted so the official diffusers script does not error on an unsupported tracker value.

After training, sample from the LoRA via `scripts/stage2/sample_sdxl_lora.py` for qualitative A/B checks against the baseline SDXL pipeline.

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
