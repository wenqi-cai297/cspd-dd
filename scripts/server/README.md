# Server helper scripts

These scripts are meant to reduce repeated manual CLI typing on the Linux GPU server.

## Recommended Prep + Stage 1 order

Prep now means `classes.json` generation plus `class -> archetype` mapping.
Stage 1 now means attribute extraction, normalization, and canonical render.

If you want the full workflow from environment checking to final canonical render, use these steps in order:

### 1. Check the server environment and install missing runtime dependencies

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
bash scripts/server/run_stage1_qwen_local.sh /data/imagenette/train 256 DEFAULT 10 /data/imagenette/class_to_archetype.json
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

Legacy compatibility note:
- Canonical render code now lives under `src/cspd_stage1/`.
- `bash scripts/server/run_stage2_render.sh ...` still works, but now forwards to `run_stage1_render.sh`.
- `cspd-stage2 render ...` still works, but `cspd-stage1 render ...` is the preferred entrypoint.

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
