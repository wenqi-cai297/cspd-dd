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
- `cspd-stage2 train --dataset-root ... --render-input ... --output-dir ...`
- Canonical Stage 1 render implementation now lives under `src/cspd_stage1/`
- Stage 2 now means generative-backbone adaptation / canonical-semantic-space familiarization; it no longer refers to render

### Main server helper scripts

- `bash scripts/server/prepare_stage1_metadata.sh ...`
- `bash scripts/server/run_stage1_qwen_local.sh ...`
- `bash scripts/server/run_stage1_normalization.sh ...`
- `bash scripts/server/run_stage1_render.sh ...`
- `bash scripts/server/run_stage2_train.sh ...`
- `bash scripts/server/run_stage1_full_workflow.sh ...`

### Default server-side output roots

- Prep metadata: `runs/prep/...`
- Stage 1 attributes: `runs/stage1/attributes/<dataset_name>/<backend>/<timestamp>`
- Stage 1 render: `runs/stage1/render/<dataset_name>/<backend>/<timestamp>`
- Stage 2 train scaffold: `runs/stage2/train/<dataset_label>/<backbone>/<timestamp>`
  - default dataset label is the dataset-root basename, except split-only roots like `.../train` become `<parent>_train` (same for `val`/`valid`/`validation`/`test`/`testing`)
  - optional override: set `STAGE2_DATASET_LABEL=...` before `bash scripts/server/run_stage2_train.sh ...`

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
bash scripts/server/check_stage1_env.sh
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
python scripts/data/normalize_stage1_attributes.py \
  --input runs/stage1/attributes/my_dataset/qwen_local/2026-03-25_170000/attributes.jsonl \
  --output-dir runs/stage1/attributes/my_dataset/qwen_local/2026-03-25_170000/normalization/2026-03-25_180000

cspd-stage1 render \
  --input runs/stage1/attributes/my_dataset/qwen_local/2026-03-25_170000/normalization/2026-03-25_180000/attributes_normalized.jsonl \
  --output-dir runs/stage1/render/my_dataset/qwen_local/2026-03-25_181500
```

Then build the Stage 2 v1 training scaffold from real images + Stage 1 canonical captions:

```bash
cspd-stage2 train \
  --dataset-root /path/to/imagefolder_dataset \
  --render-input runs/stage1/render/my_dataset/qwen_local/2026-03-25_181500/records.jsonl \
  --output-dir runs/stage2/train/my_dataset/flux_dev/2026-04-02_180000 \
  --backbone-name black-forest-labs/FLUX.1-Kontext-dev \
  --trainable-component-group conditioning_bridge \
  --trainable-component-group cross_attention \
  --adapter-type lora \
  --adapter-rank 16 \
  --batch-size 4 \
  --epochs 1 \
  --dry-run
```

This Stage 2 CLI already implements pairing / manifest generation / run-directory setup for a text-conditioning-focused adaptation plan. It now records trainable component groups, conservative module-selection patterns, and adapter/LoRA-style plan metadata for the current experimental FLUX.1 Kontext target. Full FLUX.1 Kontext fine-tuning is still a placeholder boundary in the current repo.

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
python scripts/data/generate_class_to_archetype_map_vlm.py \
  --input /path/to/classes.json \
  --dataset-root /path/to/imagefolder_dataset \
  --output /path/to/class_to_archetype.json \
  --detail-output /path/to/class_to_archetype_details.jsonl \
  --taxonomy configs/stage1/archetype_taxonomy_manual.json \
  --images-per-class 5
```

Server helper (uses the repo-bundled `classes.json` by default):

```bash
bash scripts/server/generate_class_to_archetype_vlm.sh /path/to/imagefolder_dataset 5
```

## Attribute analysis helper

To inspect slot/value distributions before writing normalization rules:

```bash
python scripts/data/analyze_attribute_values.py \
  --input /path/to/attributes.jsonl \
  --top-k 20 \
  --print-top-k 10
```

This writes a JSON summary report next to the input file and prints per-archetype, per-slot top values to stdout.

## Stage 1 normalization helper

A conservative post-processing script is included for Stage 1 `attributes.jsonl` outputs. By default it now runs deterministic normalization first, then an inline constrained VLM review pass for only the ambiguous / review-required slots:

```bash
python scripts/data/normalize_stage1_attributes.py \
  --input /path/to/attributes.jsonl \
  --output-dir /path/to/normalized_artifacts
```

Disable the inline VLM review if you want a purely deterministic run:

```bash
python scripts/data/normalize_stage1_attributes.py \
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

### Optional standalone VLM review helper for ambiguous normalization cases

For rows that were already flagged by deterministic normalization, you can still run the separate constrained VLM review helper:

```bash
python scripts/data/review_normalization_with_vlm.py \
  --input /path/to/attributes_normalized.jsonl \
  --output-dir /path/to/review_vlm_artifacts \
  --backend qwen_local
```

This helper only sends slots whose normalization metadata has `status=review_required` or non-empty `review_reasons`.
It still does **not** overwrite `attributes_normalized.jsonl`; instead it writes sidecar review artifacts:
- `normalization_review_vlm.jsonl`: one structured VLM decision per ambiguous slot
- `normalization_review_vlm_summary.json`: aggregate counts and contract metadata

Each VLM review decision is constrained to a fixed action set:
- `keep_normalized`
- `replace_normalized`
- `set_unknown`
- `defer`

The review schema also keeps the original `record_id`, `archetype`, and `slot`, and the parser forces fallback to `defer` if the model tries to change the archetype or slot.

## Server shell scripts

See `scripts/server/README.md` for the recommended order and detailed examples.

## Notes

- `mock` backend is for plumbing tests only.
- `qwen_local` is intended for server-side GPU execution.
- The pipeline enforces a unified schema and writes `unknown` / `not_applicable` / `null` when appropriate.
- Real large-scale runs should still start with a small dataset slice first to inspect speed, failure rate, and output quality.
tput quality.
