# Server helper scripts

These scripts are meant to reduce repeated manual CLI typing on the Linux GPU server.

## 1. Install / refresh the project in the shared conda environment

```bash
bash scripts/server/setup_cspd_stage1.sh
```

This script:
- activates `cspd_vlm`
- runs `pip install -e .`
- checks that `cspd-stage1` is available

## 2. Run Stage 1 with the real local Qwen backend

```bash
bash scripts/server/run_stage1_qwen_local.sh /path/to/dataset /path/to/output_dir
```

Optional third argument:
- `max_new_tokens`

Example:

```bash
bash scripts/server/run_stage1_qwen_local.sh /data/cifar10_small runs/stage1_qwen 256
```

## 3. Run Stage 1 with the mock backend

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
