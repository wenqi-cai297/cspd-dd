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

### 2. Prepare `classes.json` and propose an archetype taxonomy candidate

If you start from a Python class mapping file, first prepare metadata:

```bash
bash scripts/server/prepare_stage1_metadata.sh /path/to/classes.py IMAGENET2012_CLASSES heuristic
```

If you already have a JSON mapping file instead of a Python file:

```bash
bash scripts/server/prepare_stage1_metadata.sh /path/to/classes.json "" heuristic
```

This script:
- converts `classes.py` into `classes.json` when needed
- can still generate `class_to_archetype.json`
- supports two modes:
  - `heuristic`
  - `vlm`

If you want Qwen to first propose a better archetype taxonomy from the full class list, run:

```bash
python scripts/data/generate_archetype_taxonomy_candidate_vlm.py \
  --input /path/to/classes.json
```

This now creates a timestamped task directory under `runs/taxonomy_tasks/`, writes a per-round summary file, incrementally builds `archetype_taxonomy_candidate.json`, performs conflict checks for newly proposed archetypes, programmatically tracks uncovered semantic regions, and writes a final review JSON.

### 3. Run the full Stage 1 workflow end-to-end

```bash
bash scripts/server/run_stage1_full_workflow.sh \
  /path/to/dataset_root \
  /path/to/classes.py \
  IMAGENET2012_CLASSES \
  heuristic \
  256 \
  /path/to/sample_image.jpg
```

This script performs the full chain:
1. environment checks
2. `classes.py -> classes.json`
3. `classes.json -> class_to_archetype.json`
4. Qwen load test
5. single-image inference test
6. mock smoke run
7. final `qwen_local` attribute extraction run

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
