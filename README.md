# CSPD-DD

Minimal executable scaffold for **Stage 1: Attribute Extraction** in the CSPD-DD pipeline.

## Current Stage 1 scope

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
- Outputs:
  - `attributes.jsonl`
  - `failed_samples.jsonl`
  - `stage1_stats.json`

## Expected dataset layout

Stage 1 currently assumes a simple ImageFolder layout:

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
- each immediate subdirectory under `dataset_root` is treated as one class,
- class ids are assigned by sorting class directory names alphabetically,
- images are discovered recursively inside each class folder,
- supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

## Local development quick start

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

The repository now includes a real local backend:
- backend name: `qwen_local`
- default model: `Qwen/Qwen2.5-VL-7B-Instruct`

Example usage on the Linux GPU server:

```bash
cspd-stage1 run \
  --dataset-root /path/to/imagefolder_dataset \
  --output-dir runs/attributes/my_dataset/qwen_local/2026-03-25_170000 \
  --backend qwen_local \
  --model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --torch-dtype float16 \
  --device-map auto \
  --max-new-tokens 256
```

If you use the provided shell helper instead, you only need to pass the dataset path
(and optionally `max_new_tokens`); the output directory is generated automatically.

Useful options:
- `--disable-fast-processor`: use the slower processor path if the fast processor behaves oddly
- `--no-raw-response`: skip saving raw model text in success rows

## VLM smoke-test scripts

Two helper scripts are included under `scripts/vlm/`:

- `test_qwen_vl_load.py`: verify the local Qwen model loads on the server
- `test_single_image_infer.py`: run one image through the local VLM and inspect JSON output

## Server shell scripts

To avoid repeatedly typing the same CLI commands on the server, helper shell scripts are included under `scripts/server/`:

- `setup_cspd_stage1.sh`: activate `cspd_vlm`, install the repo with `pip install -e .`, and check the CLI
- `run_stage1_qwen_local.sh`: run Stage 1 on an ImageFolder dataset with the real local Qwen backend
- `run_stage1_mock.sh`: quick mock-backend plumbing run

## Notes

- `mock` backend is for plumbing tests only.
- `qwen_local` is intended for server-side GPU execution.
- The pipeline enforces a unified schema and writes `unknown` / `not_applicable` / `null` when appropriate.
- Real large-scale runs should still start with a small dataset slice first to inspect speed, failure rate, and output quality.
