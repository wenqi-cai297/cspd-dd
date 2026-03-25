# CSPD-DD

Minimal executable scaffold for **Stage 1: Attribute Extraction** in the CSPD-DD pipeline.

## Current Stage 1 scope

- Unified attribute schema
- JSONL dataset input
- Pluggable VLM client interface
- `mock` backend for local pipeline plumbing tests
- `qwen_local` backend for real local GPU inference with Qwen2.5-VL
- Retry + validation + failure logging
- Outputs:
  - `attributes.jsonl`
  - `failed_samples.jsonl`
  - `stage1_stats.json`

## Dataset input format

Input file must be a JSONL file with one sample per line:

```json
{"sample_id":"000001","image_path":"/path/to/image.jpg","class_id":0,"class_name":"cat"}
```

Required fields:
- `image_path`
- `class_id`
- `class_name`

Optional fields:
- `sample_id`

## Local development quick start

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .
```

Run with mock backend:

```bash
cspd-stage1 run --input data/samples.jsonl --output-dir runs/stage1_mock --backend mock
```

## Real local VLM backend

The repository now includes a real local backend:
- backend name: `qwen_local`
- default model: `Qwen/Qwen2.5-VL-7B-Instruct`

Example usage on the Linux GPU server:

```bash
cspd-stage1 run \
  --input data/samples.jsonl \
  --output-dir runs/stage1_qwen \
  --backend qwen_local \
  --model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --torch-dtype float16 \
  --device-map auto \
  --max-new-tokens 256
```

Useful options:
- `--disable-fast-processor`: use the slower processor path if the fast processor behaves oddly
- `--no-raw-response`: skip saving raw model text in success rows

## VLM smoke-test scripts

Two helper scripts are included under `scripts/vlm/`:

- `test_qwen_vl_load.py`: verify the local Qwen model loads on the server
- `test_single_image_infer.py`: run one image through the local VLM and inspect JSON output

## Notes

- `mock` backend is for plumbing tests only.
- `qwen_local` is intended for server-side GPU execution.
- The pipeline enforces a unified schema and writes `unknown` / `not_applicable` / `null` when appropriate.
- Real large-scale runs should start with a small batch first to inspect speed, failure rate, and output quality.
