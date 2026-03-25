# CSPD Stage 1

Minimal executable scaffold for **Stage 1: Attribute Extraction** in the CSPD pipeline.

## Current scope

- Unified attribute schema
- JSONL dataset input
- Pluggable VLM client interface
- Deterministic mock backend for local pipeline testing
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

Optional:
- `sample_id`

## Quick start

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .
```

Run with mock backend:

```bash
cspd-stage1 run --input data/samples.jsonl --output-dir runs/stage1 --backend mock
```

## Notes

- `mock` backend is for plumbing tests only.
- Real VLM integration should be implemented by adding a new client under `src/cspd_stage1/vlm/`.
- The pipeline enforces a unified schema and writes `unknown` / `not_applicable` / `null` when appropriate.
