# Server helper scripts

These scripts are meant to reduce repeated manual CLI typing on the Linux GPU server.

## Recommended Stage 1 order

If you want the full workflow from environment checking to final attribute extraction, use these steps in order:

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

### 2. Prepare `classes.json` and use a fixed `class_to_archetype.json`

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
- copies a fixed `class_to_archetype.json` into the run prep directory
- does not run VLM-based taxonomy discovery

If you still want VLM to produce `class_to_archetype.json`, use the fixed manual taxonomy as the allowed label set:

```bash
python scripts/data/generate_class_to_archetype_map_vlm.py \
  --input /path/to/classes.json \
  --output /path/to/class_to_archetype.json \
  --detail-output /path/to/class_to_archetype_details.jsonl \
  --taxonomy configs/stage1/archetype_taxonomy_manual.json
```

The manually fixed taxonomy definition now lives in:

```text
configs/stage1/archetype_taxonomy_manual.json
```

### 3. Run the full Stage 1 workflow end-to-end

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
2. `classes.py -> classes.json`
3. copy fixed `class_to_archetype.json`
4. Qwen load test
5. single-image inference test
6. small mock smoke run on the first 3 classes with the first 10 images per class
7. final `qwen_local` attribute extraction run

The underlying `cspd-stage1 run` command now supports resume by default: if you rerun with the same output directory, it will read existing `attributes.jsonl` / `failed_samples.jsonl`, skip previously successful records, retry records listed in `failed_samples.jsonl`, and continue appending new results. Use `--no-resume` only if you intentionally want to restart from scratch.

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
bash scripts/server/run_stage1_qwen_local.sh /path/to/dataset [max_new_tokens] [class_name_map] [flush_every] [class_archetype_map]
```

Example:

```bash
bash scripts/server/run_stage1_qwen_local.sh /data/cifar10_small 256
bash scripts/server/run_stage1_qwen_local.sh /data/imagenette/train 256 /data/imagenette/classes.json 10 /data/imagenette/class_to_archetype.json
```

The output directory is generated automatically as:

```text
runs/attributes/<dataset_name>/qwen_local/<timestamp>
```

### Run Stage 1 with the mock backend

```bash
bash scripts/server/run_stage1_mock.sh /path/to/dataset /path/to/output_dir
```

### Run Stage 2 canonical rendering

```bash
bash scripts/server/run_stage2_render.sh /path/to/attributes_normalized.jsonl [output_dir] [renderer_version]
```

Example:

```bash
bash scripts/server/run_stage2_render.sh runs/stage1/example_normalized/attributes_normalized.jsonl
```

If `output_dir` is omitted, the script now writes under:

```text
runs/stage2/<normalized_parent_dir>/<timestamp>
```

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
th/to/dataset /path/to/output_dir
```

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
