# CSPD Implementation Stage Spec

Status: repo-aligned implementation document
Owner: `wyy_coding_bot`
Scope: only implementation-facing facts, contracts, and next-step engineering guidance
Source of truth for code state: current repo at `E:\Project\2026-03-25`

---

## 0. Why this document exists

This file is **not** an idealized future spec.
It is a **repo-aligned implementation document**.

Its job is to let future coding agents answer these questions quickly:
- what stages are actually implemented in the repo right now,
- what each implemented stage concretely does,
- what artifacts and interfaces it currently exposes,
- what is only planned but not implemented yet,
- what mismatches or caveats must be remembered before continuing work.

If the repo changes, this document should be updated to match the repo.
Do not keep stale "should do" descriptions here when the code says otherwise.

---

## 1. Current repo status at a glance

### Implemented now
- **Prep metadata pipeline** is implemented.
- **Stage 1 extraction** is implemented and runnable.
- **Stage 1 normalization** is implemented as a deterministic-first canonicalization step with inline constrained VLM review enabled by default.
- **Stage 1 render** is implemented as a deterministic archetype-template renderer.
- **Stage 2 SDXL LoRA training** is implemented and has completed successful end-to-end runs on ImageNette.
- **Stage 2 inference / sampling** script is implemented for LoRA vs baseline A/B comparison.
- **Stage 3 visual/semantic mode discovery** is implemented: VAE/text/DINOv2 encoding + per-class clustering (K-Means or HDBSCAN) + mode extraction.
- **Stage 4 distilled dataset generation** is implemented: text-to-image generation using Stage 3 caption selection + Stage 2 LoRA backbone. Optional SDXL refiner support.
- **Evaluation** is implemented: train classifier (ConvNet-6/ResNet-18/ResNetAP-10) on distilled dataset, evaluate on real val set.
- Supporting server scripts, metadata prep, mock/regression runs, and full workflow wiring exist.

### Not implemented yet
- FID evaluation is not yet automated.

### Partially implemented / legacy exploratory
- **Stage 2 FLUX family**: training loop is only a stub; backbone loading and inspection work but end-to-end training is not wired.
- **Stage 2 PixArt family**: training loop is functional but has been deprioritized as an exploratory branch (text-to-image only, no img2img path).

### Important practical reading
Right now, the repo is best understood as:
- a working **Prep** pipeline for class metadata,
- a working **Stage 1** pipeline consisting of extraction -> normalization -> render,
- a working **Stage 2 SDXL LoRA** training pipeline that delegates to the official diffusers trainer,
- a working **Stage 2 inference** script for sampling from trained LoRA weights,
- a working **Stage 3** pipeline for VAE/text/DINOv2 encoding, per-class clustering (K-Means or HDBSCAN), and visual/semantic mode extraction,
- a working **Stage 4** pipeline for text-to-image distilled dataset generation with caption selection by visual clustering,
- a working **Evaluation** pipeline for training classifiers on distilled datasets and evaluating on real validation sets,
- where Stage 1 normalization is deterministic-first but can invoke constrained VLM review on ambiguous slots,
- plus planning/spec notes for Stage 4.

### Packaging / environment reality check
- The installable project in `pyproject.toml` is currently named **`cspd-stage1`**.
- The console scripts exposed there are now:
  - **`cspd-stage1`** with Stage 1 subcommands such as `run`, `normalize`, and `render`
  - **`cspd-stage2`** for Stage 2 scaffold / inspection / planning commands
  - **`cspd-stage3`** for Stage 3 encoding / clustering / mode extraction
  - **`cspd-stage4`** for Stage 4 dual-anchor distilled dataset generation
- The repo now also bundles `environment.yml` for the shared conda environment name **`cspd-dd`** used by the server shell helpers.
- Core dependencies: `torch`, `torchvision`, `numpy`, `tqdm`, `pillow`, `diffusers`, `transformers`, `accelerate`, `peft`, `sentencepiece`, `protobuf`, `tiktoken`, `safetensors`, `scikit-learn`.
- Optional dependencies (declared in `pyproject.toml`): `wandb` (W&B logging), `xformers` (memory-efficient attention), `bitsandbytes` (8-bit Adam).
- For new environment setup: `conda env create -f environment.yml && conda activate cspd-dd` installs everything needed.

---

## 2. Repo layout relevant to implemented workflow

### Main implemented packages
- `src/cspd_stage1/`
  - `__init__.py`
  - `cli.py`
  - `pipeline.py`
  - `schema.py`
  - `prompting.py`
  - `io_utils.py`
  - `render_pipeline.py`
  - `render_utils.py`
  - `templates.py`
  - `vlm/base.py`
  - `vlm/factory.py`
  - `vlm/json_utils.py`
  - `vlm/mock.py`
  - `vlm/qwen_local.py`
- `src/cspd_stage2/`
  - `__init__.py`
  - `cli.py`
  - `training.py` — main training orchestration, config, dispatch
  - `training_common.py` — shared training utilities (optimizer, scheduler, freeze logic)
  - `data.py` — pairing, manifest, dataloader
  - `backbone.py` — backbone loading, module inspection, LoRA injection
  - `families/flux/backbone.py`, `families/flux/training.py` — FLUX family (stub training)
  - `families/pixart/backbone.py`, `families/pixart/training.py` — PixArt family (functional but deprioritized)
  - `families/sdxl/backbone.py`, `families/sdxl/training.py` — SDXL family (**working end-to-end**)
  - implements Stage 2 pairing / planning / backbone inspection / SDXL LoRA training via official diffusers delegation

- `src/cspd_stage3/`
  - `__init__.py`
  - `encode.py` — VAE latent + text embedding + DINOv2 feature encoding (Stage 3A)
  - `cluster.py` — per-class clustering (K-Means or HDBSCAN) + visual/semantic mode extraction (Stage 3B+3C)
  - `cli.py` — CLI with `encode`, `cluster`, and `run` subcommands

- `src/cspd_stage4/`
  - `__init__.py`
  - `generate.py` — text2img distilled generation with optional refiner and legacy img2img paths
  - `cli.py` — CLI with `generate` subcommand

- `src/cspd_eval/`
  - `__init__.py`
  - `train.py` — classifier training + evaluation (ConvNet-6, ResNet-18, ResNetAP-10)
  - `train_utils.py` — training utilities (AverageMeter, accuracy, CutMix, etc.)
  - `models/convnet.py` — ConvNet-6 architecture
  - `models/resnet.py` — ResNet-18 architecture
  - `models/resnet_ap.py` — ResNetAP-10 architecture

### Inference scripts
- `scripts/inference/sample_sdxl_lora.py` — SDXL LoRA sampling with baseline comparison support

### Config / metadata
- `classes.json`
- `environment.yml`
- `pyproject.toml`
- `configs/stage1/archetype_taxonomy_manual.json`
- `configs/stage1/class_to_archetype_imagenet1k_manual.json`
- `configs/stage1/normalization/stage1_attribute_normalization_rules.json`

### Data / analysis scripts
- `scripts/data/convert_class_py_to_json.py`
- `scripts/data/generate_class_to_archetype_map.py`
- `scripts/data/generate_class_to_archetype_map_vlm.py`
- `scripts/data/generate_archetype_taxonomy_candidate_vlm.py`
- `scripts/data/analyze_attribute_values.py`
- `scripts/data/normalize_stage1_attributes.py`
- `scripts/data/review_normalization_with_vlm.py` (kept as the original prototype / reference path)
- `scripts/data/make_imagefolder_subset.py`

### VLM sanity / smoke test helpers
- `scripts/vlm/test_qwen_vl_load.py`
- `scripts/vlm/test_single_image_infer.py`

### Server-side execution scripts
- `scripts/server/check_stage1_env.sh`
- `scripts/server/setup_cspd_stage1.sh`
- `scripts/server/prepare_stage1_metadata.sh`
- `scripts/server/run_stage1_mock.sh`
- `scripts/server/run_stage1_qwen_local.sh`
- `scripts/server/run_stage1_normalization.sh`
- `scripts/server/run_stage1_render.sh`
- `scripts/server/run_stage1_full_workflow.sh`
- `scripts/server/generate_class_to_archetype_vlm.sh`
- `scripts/server/run_stage1_normalization_review_vlm.sh` (prototype / sidecar helper retained)
- `scripts/server/check_stage2_sdxl_env.sh` — Stage 2 SDXL environment preflight
- `scripts/server/stage2/run_sdxl_stage2_official.sh` — SDXL LoRA training launcher (default: 2 GPUs, 512 resolution)
- `scripts/server/stage2/run_pixart_stage2_baseline_sampling.sh`
- `scripts/server/stage2/run_pixart_stage2_wandb.sh`
- `scripts/server/run_stage2_train.sh`
- `scripts/server/dump_stage2_backbone_modules.sh`
- `scripts/server/README.md` documents the recommended Prep + Stage 1 + Stage 2 helper flow
- `scripts/server/stage1/run_stage1_pipeline.sh` — full Stage 1: extract → normalize → render
- `scripts/server/stage2/run_stage2_pipeline.sh` — Stage 2 training + checkpoint sampling
- `scripts/server/stage3/run_stage3_pipeline.sh` — Stage 3 encode + cluster
- `scripts/server/stage4/run_stage4_pipeline.sh` — Stage 4 generate distilled dataset
- `scripts/server/eval/run_eval_pipeline.sh` — train classifier + evaluate

### Stage 2 output-dir rule (must remember)
- The repo-standard Stage 2 run root is:
  - `runs/stage2/train/<dataset_label>/<backbone_slug>/<timestamp>`
- `scripts/server/run_stage2_train.sh` already derives this automatically.
- `cspd-stage2 train` should follow the same convention by default when `--output-dir` is omitted; do **not** force routine users to hand-type run directories.
- Dataset-label derivation rule:
  - default: `basename(dataset_root)`
  - if `dataset_root` ends with a split-only directory name in `{train,val,valid,validation,test,testing}`, use `<parent>_<split>`
- `--output-dir` remains only as an explicit override, not the normal required path.

### Stage 2 PixArt debugging / status note (must remember)
- As of 2026-04-09, the PixArt-Sigma Stage 2 path is best treated as a **debugged exploratory branch, not the forward mainline**.
- Confirmed working pieces on the real server include:
  - successful pairing when the exact Stage 1-compatible split root is used,
  - successful real diffusers PixArt backbone load,
  - successful VAE encode / prompt encode / first forward / first backward,
  - successful multi-step continuation beyond the first optimizer step,
  - optional W&B logging and periodic sampling wiring,
  - standalone baseline text-to-image sampling helper.
- The earlier dominant failure patterns were materially improved by repo changes:
  - prompt-cache path removed;
  - Stage 2 output-dir auto-derivation aligned with helper scripts;
  - PixArt frozen-module shuttle/offload complexity removed in favor of always-on-device frozen runtime;
  - PixArt full-update path given a safer FP32 trainable-parameter route;
  - PixArt LoRA path given FP32 adapter master/update weights by default;
  - first-step / post-step finite diagnostics added.
- Important limitation discovered during the latest debugging cycle:
  - standalone baseline text-to-image sampling can produce valid pretrained outputs,
  - but training-path `step=0` sampling is currently **not yet behavior-equivalent** to the clean pretrained baseline,
  - so any `step=0` mismatch should be treated as a code-path inconsistency to debug, not as evidence that LoRA initialization itself is corrupting the model.
- Current practical reading of the PixArt branch:
  - useful for preserving prior debugging lessons,
  - but **not** the recommended next family for the user's main CSPD direction.
- Why it is no longer the preferred mainline:
  - current runtime exposure is still fundamentally text-to-image-oriented,
  - no ready-made img2img branch is exposed in the present repo/runtime path,
  - this mismatches the user's longer-term goal of visual mode + semantic mode driven distilled-dataset creation, which is image-to-image flavored.
- Environment dependency reality check from real PixArt runs:
  - repo environment metadata must include `protobuf` and `tiktoken` for the current PixArt tokenizer / prompt path,
  - these were added to both `environment.yml` and `pyproject.toml` on 2026-04-09.
- Dataset-root contract is still critical for Stage 2 pairing:
  - use the exact Stage 1-compatible ImageFolder split root used by render records (e.g. `.../ImageNette/train`),
  - not the parent dataset root, or pairing may collapse to zero.
- Strategic decision after the 2026-04-09 review:
  - preserve PixArt as a separated family branch in the codebase,
  - but shift the next main investigation to **`stabilityai/stable-diffusion-xl-base-1.0`**.

---

## 3. Approved implementation-stage view of the project

For implementation tracking in this repo, use the following stage view:

### Prep
- `classes.json` generation / conversion
- `class -> archetype` mapping generation
  - including current multimodal class-level mapping

### Stage 1
1. **Stage 1A**: structured semantic extraction from real images
2. **Stage 1B**: deterministic-first normalization with optional inline VLM review for ambiguous cases
3. **Stage 1C**: canonical semantic rendering from normalized Stage 1 records

### Stage 2
- generative-backbone adaptation / canonical-semantic-space familiarization
- current working implementation: **SDXL base 1.0 UNet LoRA** via official diffusers trainer
- legacy exploratory families: FLUX (stub), PixArt (functional but deprioritized)

### Stage 3
- visual/semantic mode discovery via latent clustering
- encoding: VAE latents + CLIP text embeddings + DINOv2 CLS features
- clustering: K-Means (baseline) or HDBSCAN (density-based mode discovery)
- feature space: VAE latents (baseline) or DINOv2 features (better mode separation)
- outputs: visual mode centroids/medoids, semantic mode mean embeddings, representative captions

### Stage 4
- img2img distilled dataset generation from real medoid images
- visual clustering selects WHICH real images (medoids) to use as img2img init
- representative caption from each mode used as text conditioning
- Stage 2 LoRA weights as generation backbone, strength=0.8
- optional SDXL refiner for detail/sharpness
- text2img path preserved for ablation (visual_mode=none)

### Important naming caveat
Historically, render was treated as a Stage 2 compatibility surface.
In the current repo, the canonical render implementation lives under `src/cspd_stage1/`, and the old `cspd-stage2 render` compatibility entrypoint has been removed.
For current repo semantics and workflow docs, **render belongs to Stage 1**, not a separate later stage.

---

## 4. Prep — metadata and class-level setup

### Implementation status
**Implemented in repo.**

### What Prep currently includes
1. **Class metadata preparation**
   - conversion from `classes.py` to `classes.json`
   - repo-bundled `classes.json` is now tracked and is the default class-name map

2. **Class-to-archetype mapping**
   - fixed manual taxonomy is the target label set
   - current preferred path is **multimodal class-level mapping**
   - mapping uses:
     - class text / readable name
     - sampled class images
     - fixed archetype taxonomy
   - current implementation is designed to reduce pure text-label ambiguity on ImageNet-style classes

### Main files
- `classes.json`
- `scripts/data/convert_class_py_to_json.py`
- `scripts/data/generate_class_to_archetype_map_vlm.py`
- `scripts/server/prepare_stage1_metadata.sh`
- `scripts/server/generate_class_to_archetype_vlm.sh`

### Practical workflow note
Prep artifacts are class-level / dataset-level metadata.
They should not be confused with image-level Stage 1 semantic outputs.

---

## 5. Stage 1A — Structured semantic extraction

### Implementation status
**Implemented in repo.**
This is the model-facing image-level extraction step.

### Main code
- `src/cspd_stage1/cli.py`
- `src/cspd_stage1/pipeline.py`
- `src/cspd_stage1/schema.py`
- `src/cspd_stage1/prompting.py`
- `src/cspd_stage1/vlm/*`

### What Stage 1A actually does in the repo
The current Stage 1A pipeline:
- scans an **ImageFolder-style dataset**,
- resolves class labels,
- maps each class to a fixed semantic **archetype**,
- loads the slot schema associated with that archetype,
- prompts a VLM to fill only those slots,
- validates returned payload shape,
- fills missing requested slots conservatively with `unknown`,
- writes incremental JSONL artifacts,
- supports resume / skip of prior successful samples,
- retries previously failed samples.

### Current Stage 1A CLI
Primary implemented CLI path:
```bash
cspd-stage1 run --dataset-root ... --output-dir ...
```

### Current runtime config shape
`Stage1Config` in `src/cspd_stage1/pipeline.py` includes:
- `dataset_root`
- `output_dir`
- `backend`
- `max_retries`
- `save_raw_response`
- `model_name`
- `torch_dtype`
- `device_map`
- `use_fast_processor`
- `max_new_tokens`
- `class_name_map`
- `class_archetype_map`
- `flush_every`
- `resume`

### Implemented VLM backend status
From `src/cspd_stage1/vlm/factory.py`:

Implemented real backends:
- `mock`
- `qwen_local`

Recognized but currently placeholder / not implemented:
- `openai`
- `qwen-vl`
- `internvl`
- `llava`
- `claude-vision`

### Important reality check
The current repo’s real extraction path is **`qwen_local`**, not `openai`.

---

## 6. Stage 1 archetype system — actual repo state

### Current source of truth
- `configs/stage1/archetype_taxonomy_manual.json`
- `src/cspd_stage1/schema.py`

### Actual fixed taxonomy in repo
The current manual taxonomy file contains the following fine-grained archetypes:
- `animal`
- `plant_or_fungus`
- `food_and_drink`
- `vehicle`
- `clothing_and_wearable`
- `furniture`
- `container`
- `tool`
- `device_or_appliance`
- `instrument`
- `weapon`
- `sports_or_toy`
- `household_object`
- `structure_or_building`
- `natural_scene_or_landform`
- `human_or_person`
- `text_or_media_object`
- `decorative_or_symbolic_object`

### Archetype mapping revision (2026-04-12)
20 misplaced class-to-archetype mappings were fixed in `class_to_archetype_imagenet1k_manual.json`:
- 4 stores/shops moved from food_and_drink → structure_or_building (bakery, butcher shop, confectionery, grocery store)
- 6 containers moved from food_and_drink → container (beer/wine/pop bottle, soup bowl, plate, packet)
- rotisserie moved from food_and_drink → device_or_appliance
- nipple moved from food_and_drink → household_object
- bobsled/dogsled/go-kart moved from sports_or_toy → vehicle
- ballplayer moved from sports_or_toy → human_or_person
- patio moved from natural_scene → structure_or_building
- hay moved to plant_or_fungus, spider web to decorative_or_symbolic_object
- shopping cart moved from container → vehicle

These mapping changes require re-running Stage 1A for affected classes (archetype determines slot schema).

### Important current policy boundary
Current engineering direction is:
- **Prep can use class identity**, because Prep is explicitly class-level metadata construction.
- **Stage 1 normalization/render should prefer archetype-aware rules**, not expanding class-specific hard patches.
- **If VLM is used downstream, it should appear as constrained review/fallback, not as a full class-aware rewrite layer.**
- **All optimization must be at archetype level, never class level.** Optimizing for specific classes (e.g., special rules for "cassette player") is methodologically invalid and will not generalize.

That boundary matters for method cleanliness.

---

## 7. Stage 1A slot schema — actual repo state

### Source of truth
- `src/cspd_stage1/schema.py`

### Actual slot design
The current repo does **not** use a single shared universal slot set.
Instead, each archetype has its own slot schema.
Typical schema size is 7 slots per archetype.

Examples:

#### `animal`
- `species_or_category`
- `color_or_pattern`
- `body_trait`
- `pose_or_state`
- `background_or_habitat`
- `viewpoint`
- `salient_part_or_focus`

#### `vehicle`
- `vehicle_type`
- `color`
- `shape_or_structure`
- `state_or_action`
- `environment`
- `viewpoint`
- `salient_part_or_accessory`

#### `structure_or_building`
- `structure_or_building_type`
- `material_or_surface`
- `architectural_style_or_form`
- `scale_or_extent`
- `surrounding_environment`
- `viewpoint`
- `salient_structural_part`

#### `food_and_drink`
- `food_or_drink_type`
- `color`
- `shape_or_structure`
- `preparation_or_serving_style`
- `container_or_context`
- `viewpoint`
- `salient_topping_or_ingredient`

---

## 8. Stage 1A prompt/output contract — actual repo state

### Prompt source
- `src/cspd_stage1/prompting.py`

### Actual prompt behavior
The current prompt asks the VLM to return JSON in the shape:
```json
{
  "archetype": "...",
  "attributes": {
    "slot_name": "per-slot guidance string with examples"
  }
}
```

Each slot placeholder now contains **specific guidance** instead of generic `"short phrase"`. This is defined in `SLOT_GUIDANCE` dict in `prompting.py`. Examples:
- `background_or_habitat`: `"scene or place WHERE the subject is, e.g. grassy field, lake shore. Do NOT write just a color"`
- `operating_state_or_display_state`: `"device state with detail, e.g. playing music with display lit. Do NOT write just 'on' or 'off'"`
- `pose_or_state`: `"what the animal is doing, e.g. swimming, being held by person, curled up sleeping"`
- `viewpoint`: `"camera angle, e.g. front view, side view, top-down view"`

The prompt rules explicitly instruct:
- JSON only, no markdown or code fences,
- short phrases (2-5 words), not full sentences,
- describe ONLY what is visible in the specific image,
- for background/environment slots: describe the PLACE or SCENE, not just a color,
- for state/pose slots: describe the specific ACTION or CONDITION, not just 'on'/'off',
- each slot should have a SINGLE value, not a comma-separated list,
- prefer a coarse description over 'unknown'.

### Prompt revision history
- **Original**: generic `"short phrase"` placeholder for all slots. VLM frequently gave colors for backgrounds, bare "on"/"off" for states, and comma-separated lists.
- **2026-04-12**: per-slot guidance with examples and explicit anti-patterns. Addresses the three main VLM extraction quality issues (color-as-background, bare on/off, comma lists).

### Important reality check
Current Stage 1 extraction output is **flat string-valued attributes**.
There are no per-slot confidence objects.

---

## 9. Stage 1A artifacts — actual repo contract

### Implemented output files per run
Current Stage 1A writes:
- `attributes.jsonl`
- `failed_samples.jsonl`
- `stage1_stats.json`

### Successful row structure
Successful rows contain fields such as:
- `record_id`
- `dataset_root`
- `split`
- `sample_id`
- `relative_image_path`
- `image_path`
- `file_name`
- `class_id`
- `class_name_raw`
- `class_name`
- `archetype`
- `slot_schema`
- `backend`
- `model_name`
- `extracted_at`
- `attributes`
- `extraction_status`
- optionally `raw_response`

### Resume semantics
Implemented resume behavior:
- prior successful `record_id`s are skipped,
- prior failures are retried,
- incremental flush is supported via `flush_every`.

---

## 10. Stage 1B — Deterministic-first normalization with inline VLM review

### Implementation status
**Implemented in repo.**

### Main code and rules
- `scripts/data/normalize_stage1_attributes.py`
- `configs/stage1/normalization/stage1_attribute_normalization_rules.json`

### Main CLI / helper surface
Preferred current entrypoints:
```bash
cspd-stage1 normalize --input ... --output-dir ...
```

```bash
bash scripts/server/run_stage1_normalization.sh <attr_dir_or_jsonl>
```

Default helper output path:
```text
<attribute_run_dir>/normalization/<timestamp>
```

### Default behavior
Current default Stage 1B flow is:
1. deterministic normalization
2. inline constrained VLM review for ambiguous slots only
3. final effective normalized output

This inline VLM review is **enabled by default**.
It can be manually disabled via:
- `--disable-vlm-review`

### What Stage 1B does
Stage 1B loads `attributes.jsonl` from Stage 1A and performs:
- lexical cleanup,
- placeholder cleanup,
- slot-aware canonicalization,
- low-value slot suppression,
- archetype-aware review flagging,
- inline constrained VLM review for ambiguous / `review_required` slots,
- audit artifact generation.

### Important current policy boundary
Current approved direction is:
- deterministic-first,
- auditable,
- archetype-aware,
- with **VLM used only as constrained inline review/fallback**, not as a full free-form normalization replacement.

In other words:
- using class identity in Prep is acceptable,
- but Stage 1B should not drift into label-cheating behavior,
- and VLM review should stay limited to ambiguous slots, not all attributes.

### What triggers inline VLM review
Inline VLM review is invoked for:
1. Slots whose deterministic normalization metadata indicates ambiguity (`status == "review_required"` or non-empty `review_reasons`)
2. **Slots mapped to `"unknown"` by normalization** (`status == "mapped_to_unknown"`) — the VLM is given the image and asked to try providing a better value. This is the `review.mapped_to_unknown_recovery` trigger.

In ImageNette testing, this extended trigger increased VLM review items from 66 to 969, with 511 (52.7%) receiving a `replace_normalized` action — successfully recovering values that normalization had killed.

### Current action space for VLM review
The constrained review path only allows structured slot decisions such as:
- `keep_normalized`
- `replace_normalized`
- `set_unknown`
- `defer`

It is not intended to change archetypes or invent new slot schemas.

### Main output artifacts
When run, Stage 1B produces:
- `attributes_normalized.jsonl`
- `normalization_audit.jsonl`
- `normalization_review_queue.jsonl`
- `normalization_summary.json`
- `normalization_rules_snapshot.json`
- `normalization_review_vlm.jsonl`
- `normalization_review_vlm_summary.json`

### Important fields added by normalization
Normalized rows now include or may include:
- `normalized_attributes` — deterministic result, preserved for auditability
- `attribute_normalization` — deterministic per-slot normalization metadata
- `normalization_review_required`
- `effective_normalized_attributes` — deterministic result plus any inline constrained VLM overrides actually applied
- `vlm_review` — per-row constrained VLM review metadata

### Current behavior summary
Stage 1B is currently:
- deterministic-first,
- audit-preserving,
- archetype-aware,
- with inline constrained VLM review enabled by default,
- but still not a fully generative normalization stage.

---

## 11. Stage 1C — Canonical semantic rendering

### Implementation status
**Implemented in repo.**

### Main code
- `src/cspd_stage1/render_pipeline.py`
- `src/cspd_stage1/templates.py`
- `src/cspd_stage1/render_utils.py`
- `src/cspd_stage1/cli.py`

### Stage 2 package status
- `src/cspd_stage2/__init__.py` is now only a reserved scaffold package.
- It is not the implementation location for current render behavior.

### Current CLI / script surface
Preferred current entrypoints:
```bash
cspd-stage1 render --input ... --output-dir ...
```

```bash
bash scripts/server/run_stage1_render.sh /path/to/attributes_normalized.jsonl
```

There is no current `cspd-stage2 render` CLI entrypoint and no `scripts/server/run_stage2_render.sh` helper in the repo.
The installable console script exposed by `pyproject.toml` is `cspd-stage1`.
Operationally, the server-side full workflow also runs `scripts/vlm/test_qwen_vl_load.py` and `scripts/vlm/test_single_image_infer.py`, and `scripts/server/run_stage1_full_workflow.sh` performs a default mock smoke run on the first 3 classes and first 10 images per class unless `--skip-smoke` is passed.

### What Stage 1C does
Stage 1C converts normalized Stage 1 records into deterministic canonical captions using:
- archetype-specific fixed templates,
- slot filtering / dropping,
- slot cleanup / formatting,
- deterministic assembly.

### Effective input preference
Render now prefers, when present:
- `effective_normalized_attributes`

and otherwise falls back to:
- `normalized_attributes`

This lets Stage 1C consume the final effective Stage 1B output while preserving deterministic-first audit traces.

### Current render outputs
Successful render runs write:
- `records.jsonl`
- `render_summary.json`
- optionally `failures.jsonl` when failures exist

### Output row fields include
- `record_id`
- `sample_id`
- `class_name`
- `archetype`
- `canonical_caption`
- `renderer.renderer_version`
- `renderer.template_family`
- `renderer.template_id`
- `anchor_slot`
- `verbalized_slots`
- `dropped_slots`
- `drop_reasons`
- `used_normalized_attributes`
- `used_effective_normalized_attributes`
- `normalization_review_required`
- `vlm_review`
- `render_warnings`
- `render_status`

### Current output path convention
Preferred output root is now:
```text
runs/stage1/render/<dataset>/<backend>/<timestamp>
```

### Current render policy boundary
Current approved direction is:
- deterministic,
- normalized-first,
- archetype-specific,
- **diversity-preserving** — slot drop rules should retain information that provides intra-class distinction,
- auditable,
- and increasingly **archetype-aware rather than class-aware**.

That means render should keep using:
- archetype template families,
- slot-level drop rules,
- cleanup heuristics,
- class-name fallback only as a narrow anchor recovery path already present in the implementation,
- but avoid turning into a class-specific correction layer or a free-form VLM text generator.

### Render slot-drop policy revision (2026-04-11)
The original Stage 1C render was overly aggressive in dropping slots, causing ~30% caption duplication and low intra-class diversity. The following relaxations were applied:
- **Background**: reduced from 23 suppressed values to 5 (only truly uninformative: `neutral`, `unknown`, `indistinct`, `dark`, `cloth`). Removed `color_like_background` suppression. Relaxed `complex_background` rule for animals (allow commas/with).
- **Pose/state**: reduced from 21 suppressed values to 3 (`stationary`, `inactive`, `unplayed`). Bare `"on"`/`"off"` are dropped via a dedicated `STATE_DROP_VALUES` set instead of being treated as generic low-value poses. Meaningful poses like `"being held"`, `"standing"`, `"resting"`, `"deployed"` are now preserved.
- **Viewpoint**: `"front view"` and `"side view"` are no longer dropped — they provide composition information even if common.
- **Salient part**: reduced from 16 suppressed values to 6. Values like `"body"`, `"head"`, `"face"`, `"fish"`, `"tower"` are now preserved.
- **Trait/shape**: reduced from 14 suppressed values to 4. Geometric descriptors (`"rectangular"`, `"cylindrical"`, `"spherical"`) are now preserved. Hard-coded pre-anchor drops for shapes removed.
- **Food/container**: relaxed archetype-specific suppressions (food shape, food context, container fill).

### Normalization rules revision (2026-04-11)
The normalization rules in `configs/stage1/normalization/stage1_attribute_normalization_rules.json` were also relaxed to stop destroying upstream diversity before it reaches the renderer:
- **Background color contamination**: disabled `slot_contamination_color_background` markers (`"white"`, `"black"`, `"dark"` no longer auto-killed). VLM often uses colors as background descriptions (e.g., `"white"` for studio backdrop); these are now preserved.
- **Background low-value phrases**: reduced from 9 → 2 (`"neutral"`, `"indistinct"`). Values like `"indoor"`, `"outdoor"`, `"wall"`, `"living room"`, `"cloth"`, `"grassy area"` are no longer mapped to unknown.
- **Background phrase map**: removed `→ unknown` mappings for `"wall"`, `"living room"`, `"cloth"`, `"solid blue"`.
- **State normalization**: stopped collapsing descriptive states into bare `"on"`/`"off"`. `"powered on"` stays `"powered on"`, `"active"` stays `"active"`, `"inactive"` stays `"inactive"`, `"powered off"` stays `"powered off"`. Only bare `"on"`/`"off"` are dropped at render time.
- **Low-value state phrases**: reduced from 4 → 1 (only `"stationary"`). `"being held"`, `"resting"`, `"at rest"` are no longer treated as low-value.
- **Low-value shape phrases**: cleared entirely. Shapes like `"curved tubing"`, `"spherical"`, `"rectangular"` are no longer suppressed at normalization.
- **Design principle**: all rule changes are archetype-level (slot-type aware), never class-level. No rule references class names or class-level statistics.

---

## 12. Stage 2 — SDXL LoRA training (current mainline)

### Implementation status
**Implemented and successfully run on ImageNette.**

### Core purpose
Train the SDXL UNet via LoRA so the model's semantic space learns to recognize our Stage 1C canonical captions. The training pairs are `(real image, canonical_caption)` from Stage 1 render outputs.

### Architecture
Stage 2 SDXL delegates training to the **official diffusers `train_text_to_image_lora_sdxl.py`** script. The repo owns:
- **pairing**: matching ImageFolder images to Stage 1C render `records.jsonl` by `record_id`
- **dataset materialization**: copying images + generating `metadata.jsonl` in diffusers imagefolder format
- **launch orchestration**: building the `accelerate launch` command with config translation
- **preflight checks**: validating environment, script resolution, dataset integrity

### Main code
- `src/cspd_stage2/families/sdxl/training.py` — materialization, command building, launch
- `src/cspd_stage2/training.py` — dispatch (detects `sdxl` family, routes to official wrapper)
- `src/cspd_stage2/cli.py` — CLI with all SDXL-specific flags (`--sdxl-*`)
- `scripts/server/stage2/run_sdxl_stage2_official.sh` — server helper
- `scripts/server/check_stage2_sdxl_env.sh` — environment check

### Training configuration (current defaults)
- backbone: `stabilityai/stable-diffusion-xl-base-1.0`
- parameterization: LoRA (UNet attention layers: `to_k`, `to_q`, `to_v`, `to_out.0`)
- resolution: **512** (lowered from SDXL native 1024 for memory)
- GPUs: **2** (default `--sdxl-num-processes 2`)
- mixed precision: fp16
- gradient checkpointing: enabled
- lr: 2e-5, scheduler: constant, no warmup
- VAE + text encoders: frozen
- `--report_to` is omitted (not `"none"`) to avoid accelerate tracker init errors

### CLI usage
```bash
cspd-stage2 train \
  --dataset-root /path/to/ImageNette/train \
  --render-input /path/to/records.jsonl \
  --backbone-name stabilityai/stable-diffusion-xl-base-1.0 \
  --training-parameterization lora \
  --adapter-rank 64 \
  --batch-size 8 --epochs 5 \
  --resolution 512
```

### Server helper usage
```bash
bash scripts/server/stage2/run_sdxl_stage2_official.sh \
  <dataset_root> <render_records_jsonl> [batch_size] [epochs] [extra args...]
```

### Training output artifacts
```text
runs/stage2/train/<dataset_label>/<backbone_slug>/<timestamp>/
├── sdxl_materialized_dataset/       # copied images + metadata.jsonl
│   ├── images/
│   └── metadata.jsonl
├── official_output/                 # diffusers trainer output
│   ├── pytorch_lora_weights.safetensors   # final LoRA weights
│   └── checkpoint-*/                      # intermediate checkpoints
├── sdxl_official_launch_plan.json
├── sdxl_official_stdout.txt
├── sdxl_official_stderr.txt
├── trainer_plan.json
├── stage2_config_snapshot.json
├── stage2_run_summary.json
├── train_manifest_summary.json
├── unmatched_images.jsonl
└── unmatched_render_records.jsonl
```

### External dependency
The official diffusers repo must be cloned and pip-installed from source (`pip install -e .`). The training script is resolved via:
1. `--sdxl-official-script` CLI flag
2. `CSPD_STAGE2_SDXL_SCRIPT` env var
3. `DIFFUSERS_REPO_ROOT/examples/text_to_image/train_text_to_image_lora_sdxl.py`
4. `train_text_to_image_lora_sdxl.py` on PATH

### First successful training run (2026-04-10)
- dataset: ImageNette train (12,894 pairs, 100% pairing rate, 10 classes, 7 archetypes)
- config: rank=16, batch_size=1, 1 epoch (16,120 steps), 2 GPUs, ~6h49m
- result: LoRA weights (8.3MB) produced; qualitative A/B comparison shows clear improvement over baseline
- observation: model shifts from conceptual/illustration style toward realistic photo style matching training data
- known gap: generated images still differ from real dataset photos in detail, texture, and composition naturalness

### Inference / sampling script
- `scripts/inference/sample_sdxl_lora.py`
- loads base SDXL + optional LoRA weights
- generates images from canonical captions for visual A/B comparison
- supports `--no-lora` baseline mode with same seed for fair comparison
- default prompts cover all 7 ImageNette archetypes
- output: PNG images + `sample_metadata.json`

### Hyperparameter exploration results (2026-04-10 → 2026-04-11)
- **rank=16, epoch=20, batch=1**: learns semantic space but overfits (color drift, artifacts)
- **rank=64, epoch=5, batch=8**: better quality, less overfitting, 1h55m training time
- **rank=64, epoch=5/10/15/20 checkpoint comparison**: epoch=15 is the best overall — chain saw cleanest, spaniel normal, no overfitting artifacts
- **Selected configuration**: rank=64, epoch=15 (checkpoint-12090), batch=8, 2 GPUs
- LoRA weights at this config: ~355MB (safetensors)
- Remaining quality gap vs real photos: detail/texture, composition naturalness — acceptable for Stage 3/4 pipeline

### Important caveats
- the official diffusers script version must match the pip-installed diffusers version (use `pip install -e .` from the cloned repo)
- `--report_to none` must NOT be passed to the official script; latest accelerate rejects `"none"` as an unsupported tracker
- dataset-root must be the exact Stage 1-compatible ImageFolder split root (e.g. `.../ImageNette/train`), not the parent

---

## 13. What is deterministic vs learned right now

### Learned / model-driven
- Prep multimodal class-to-archetype mapping
- Stage 1A image-level attribute extraction (VLM with per-slot guided prompts)
- Stage 1B inline VLM review for ambiguous slots AND unknown-recovery
- Stage 2 SDXL UNet LoRA training (canonical-caption conditioning alignment)

### Deterministic / rule-driven
- Stage 1B first-pass normalization
- Stage 1C render
- Stage 2 pairing / dataset materialization / launch orchestration
- Stage 3 encoding / clustering / mode extraction

This separation is deliberate.
The repo currently uses VLMs where semantic proposal or ambiguity resolution is needed, uses diffusion model fine-tuning for semantic-space alignment, while keeping the main cleanup/render/orchestration paths auditable and deterministic.

---

## 14. Stage 3 — Visual/semantic mode discovery via latent clustering

### Implementation status
**Implemented in repo. Running experiments on ImageNette with DINOv2 + K-Means/HDBSCAN.**

### Core purpose
Discover representative visual and semantic modes per class via clustering. Visual clustering determines WHICH captions to use for Stage 4 generation (one representative caption per mode). Semantic modes provide the text embeddings and representative captions.

### Architecture

```
Stage 3A: Encode
  images → SDXL VAE → latents (N, 4, H/8, W/8)          [for mode extraction]
  captions → SDXL CLIP × 2 → text embeddings (N, 77, 2048) + pooled (N, 1280)
  images → DINOv2 (dinov2_vitb14) → CLS features (N, 768) [for clustering]

Stage 3B+3C: Cluster + Extract
  per class:
    K-Means (baseline): cluster on selected feature space, K = IPC
    HDBSCAN (mode discovery): discover natural density modes, allocate IPC proportionally
  per cluster:
    visual mode  = VAE latent centroid (+ medoid record_id)
    semantic mode = mean text embedding (+ representative caption from semantic medoid)
```

### Main code
- `src/cspd_stage3/__init__.py`
- `src/cspd_stage3/encode.py` — VAE latent + text embedding + DINOv2 feature encoding
- `src/cspd_stage3/cluster.py` — per-class clustering (K-Means or HDBSCAN), visual/semantic mode extraction
- `src/cspd_stage3/cli.py` — CLI with `encode`, `cluster`, and `run` subcommands

### CLI usage
```bash
# Full pipeline (encode + cluster)
cspd-stage3 run \
  --dataset-root /path/to/ImageNette/train \
  --render-input /path/to/records.jsonl \
  --output-dir runs/stage3/imagenette \
  --ipc 10 \
  --cluster-space dino \
  --cluster-method hdbscan \
  --min-cluster-size 15 \
  --min-samples 3 \
  --pca-dim 0

# Or step by step:
cspd-stage3 encode --dataset-root ... --render-input ... --output-dir runs/stage3/encoded
cspd-stage3 cluster --encode-dir runs/stage3/encoded --output-dir runs/stage3/modes --ipc 10 \
  --cluster-space dino --cluster-method kmeans
```

### Clustering parameters
- **--cluster-space**: `"vae"` (flatten VAE latents, 16384-dim, baseline) or `"dino"` (DINOv2 CLS features, 768-dim, better mode separation)
- **--cluster-method**: `"kmeans"` (baseline, K=IPC) or `"hdbscan"` (density-based mode discovery)
- **--min-cluster-size**: HDBSCAN parameter — minimum points for a subtree to count as a real cluster split in the condensed tree. Controls minimum legitimate cluster size. (ignored for kmeans)
- **--min-samples**: HDBSCAN parameter — k in k-NN for core distance computation. Controls density estimation smoothness. Lower = finer density landscape, more clusters. Higher = smoother, fewer clusters. (ignored for kmeans)
- **--pca-dim**: PCA dimensions for HDBSCAN pre-processing. `0` skips PCA. DINOv2 features (768-dim) usually don't need PCA.

### HDBSCAN mode discovery flow
1. Optional PCA dimensionality reduction on selected feature space
2. HDBSCAN discovers natural density modes (no preset K)
3. Noise points assigned to nearest discovered mode
4. If 0-1 modes found → fallback to K-Means
5. If modes > IPC → farthest-point sampling to select IPC most diverse modes
6. If modes <= IPC → proportional IPC allocation, sub-cluster large modes with K-Means
7. Mode extraction always uses VAE latents (visual) and text embeddings (semantic)

### DINOv2 encoding
- Model: `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`
- Input: images resized to 224×224, ImageNet normalization
- Output: CLS token features (N, 768)
- Purpose: DINOv2 produces semantically rich features with natural cluster structure, better suited for mode discovery than VAE latents (which are smooth and high-dimensional)

### Output artifacts
```text
runs/stage3/<output_dir>/
├── encoded/
│   ├── latents.pt              # (N, 4, H/8, W/8) VAE latents
│   ├── text_embeds.pt          # (N, 77, 2048) concatenated CLIP embeddings
│   ├── pooled_embeds.pt        # (N, 1280) pooled text embeddings
│   ├── dino_embeds.pt          # (N, 768) DINOv2 CLS features
│   └── encode_index.json       # per-sample metadata
├── modes_<method>/             # e.g. modes_dino_kmeans, modes_dino_hdbscan
│   ├── visual_modes.pt         # (total_modes, 4, H/8, W/8) centroid latents
│   ├── semantic_modes.pt       # (total_modes, 77, 2048) mean text embeddings
│   ├── pooled_modes.pt         # (total_modes, 1280) mean pooled embeddings
│   ├── modes_index.json        # per-mode metadata (class, archetype, captions, cluster sizes)
│   └── stage3_summary.json     # clustering summary
```

### Key design decisions
- **Clustering space vs mode extraction space**: clustering can use DINOv2 or VAE features, but mode extraction always uses VAE latents (visual) and text embeddings (semantic) for Stage 4 compatibility
- **DINOv2 for clustering**: 768-dim CLS features have natural cluster structure; VAE latents (16384-dim) are smooth and cause HDBSCAN to fail (6-8/10 classes fall back to K-Means in VAE space vs 1/10 in DINOv2 space)
- **Visual mode**: cluster centroid in VAE latent space; medoid record_id recorded for reference
- **Semantic mode**: mean of text embeddings within cluster; **representative caption** from semantic medoid is the primary output used by Stage 4
- **IPC as K**: for K-Means, number of clusters per class = IPC. For HDBSCAN, IPC is the target budget allocated proportionally across discovered modes
- Uses the same SDXL VAE + text encoders as Stage 2 for space consistency

### Experiment observations (ImageNette, IPC=10)
- **VAE K-Means**: baseline, uniform cluster sizes
- **DINO K-Means**: more uneven cluster sizes (max/min ratio 2.4x-9.0x), reflecting real density variation
- **DINO HDBSCAN** (min_cluster_size=15, min_samples=3, pca_dim=50): 9/10 classes discovered modes independently, only chain saw fell back to K-Means. gas pump had a 2-member micro-cluster (noise artifact from low min_cluster_size/min_samples)

---

## 15. Stage 4 — Img2img distilled dataset generation

### Implementation status
**Implemented in repo. Running experiments on ImageNette.**

### Core purpose
Generate the final distilled dataset using img2img generation from real medoid images. Stage 3 visual clustering selects the most representative real image (medoid) per mode as the img2img starting point, with the representative caption as text conditioning. This preserves the diversity of real images while applying the Stage 2 LoRA-tuned generation quality.

### Generation flow per mode (recommended: visual_mode="medoid")
```
Stage 3 mode → visual medoid (real image) + representative_caption (text string)
  → SDXL img2img pipeline (+ Stage 2 LoRA), strength=0.8
  → optional SDXL refiner pass
  → distilled image (PNG)
```

### Main code
- `src/cspd_stage4/__init__.py`
- `src/cspd_stage4/generate.py` — text2img generation with optional refiner and legacy img2img paths
- `src/cspd_stage4/cli.py` — CLI with `generate` subcommand
- `scripts/server/stage4/run_stage4_pipeline.sh` — server pipeline script

### CLI usage
```bash
# Recommended: text2img with LoRA (visual clustering selects captions)
cspd-stage4 generate \
  --modes-dir runs/stage3/.../modes_dino_kmeans \
  --lora-weights runs/stage2/.../checkpoint-12090/pytorch_lora_weights.safetensors \
  --output-dir runs/stage4/imagenette/ipc10

# With SDXL refiner for added detail/sharpness
cspd-stage4 generate \
  --modes-dir runs/stage3/.../modes_dino_kmeans \
  --lora-weights runs/stage2/.../pytorch_lora_weights.safetensors \
  --output-dir runs/stage4/imagenette/ipc10 \
  --refiner-model stabilityai/stable-diffusion-xl-refiner-1.0

# Without LoRA (baseline SDXL)
cspd-stage4 generate \
  --modes-dir runs/stage3/.../modes_dino_kmeans \
  --output-dir runs/stage4/imagenette/ipc10/baseline
```

### Key parameters
- **--visual-mode**: `"medoid"` (recommended) uses real medoid image as img2img init. `"centroid"` uses decoded VAE centroid. `"none"` for pure text2img.
- **--strength**: Img2img denoising strength. Default `0.8`. Higher = more regeneration from LoRA, lower = closer to original image. Ignored when visual-mode=none.
- **--resolution**: Output image resolution. Default `1024` (SDXL native).
- **--guidance-scale**: CFG strength. Default `9.0`. Higher = sharper but less diverse.
- **--num-inference-steps**: Diffusion sampling steps. Default `50`.
- **--refiner-model**: Optional SDXL refiner model ID. When set, runs refiner pass after base generation for added detail/sharpness.
- **--refiner-strength**: Denoising strength for refiner pass (0-1). Default `0.3`.
- **--semantic-mode** (hidden, default `"caption"`): `"caption"` uses representative caption text as prompt. `"embedding"` uses mean text embedding (legacy baseline, blurry).

### Output artifacts
```text
runs/stage4/<dataset>/<ipc>/<lora_tag>/<timestamp>/
├── images/
│   ├── <class_raw>/
│   │   ├── <class_raw>_mode000.png
│   │   ├── <class_raw>_mode001.png
│   │   └── ...
│   └── ...
├── distilled_metadata.json    # per-image metadata with mode info
└── stage4_summary.json        # generation summary
```

### Design decisions
- **Img2img from medoid over pure text2img**: pure text2img produced homogeneous images because captions within a class are too similar. Img2img from real medoid images preserves the visual diversity of the original dataset while applying LoRA-tuned generation quality. Visual clustering selects WHICH real images to use as anchors and which captions to pair them with.
- **High strength (0.8)**: allows significant regeneration so the LoRA backbone influences the output, while the real image provides composition and diversity.
- **Per-mode seeding**: `seed + mode_index` for reproducibility with diversity
- **ImageFolder output structure**: images organized by class for downstream classifier training
- **SDXL refiner**: optional second pass that adds detail/sharpness after base generation

### Evolution of generation strategy
1. img2img + mean embedding → all-black (custom loop incompatible with SDXL)
2. img2img + mean embedding + official pipeline → blurry (averaged embedding not real caption)
3. img2img + representative caption → quality OK but Stage 2 vs 4 mismatch
4. text2img + representative caption → matches Stage 2 inference but homogeneous output
5. **img2img from medoid + representative caption + refiner** → current approach, preserves diversity

---

## 16. What future coding agents should not get wrong

### 16.1 All optimization must be archetype-level, never class-level
This is a hard methodological boundary. Do not write normalization rules, render drop rules, or prompt guidance that references specific class names or class-level statistics. The only place class identity is used is in Prep (class-to-archetype mapping). Everything downstream operates on archetype + slot name only.

### 16.2 All four stages + evaluation are implemented
The repo covers Prep, Stage 1 (1A+1B+1C), Stage 2 (SDXL LoRA + inference), Stage 3 (encoding + clustering + mode extraction), Stage 4 (text2img distilled generation), and Evaluation (classifier training + accuracy measurement). FID evaluation is not yet automated.

### 16.3 Stage 1A prompt now uses per-slot guidance
The prompt template no longer uses generic `"short phrase"` placeholders. Each slot has specific guidance with examples and anti-patterns defined in `SLOT_GUIDANCE` dict in `prompting.py`. This was added on 2026-04-12.

### 16.4 Stage 1B VLM review now triggers on mapped-to-unknown slots
VLM review is no longer limited to `review_required` slots. It also triggers `review.mapped_to_unknown_recovery` on all slots that normalization mapped to unknown, giving the VLM a chance to provide a better value by looking at the image.

### 16.5 Archetype mapping was revised on 2026-04-12
20 classes in `class_to_archetype_imagenet1k_manual.json` were remapped. If reusing old Stage 1A extraction results, check whether any affected classes are present — those records need re-extraction because the slot schema changed.

### 16.6 Do not pass --report_to none to the official SDXL script
Latest accelerate rejects `"none"` as an unsupported tracker. The repo already handles this by omitting `--report_to` when the value is `"none"`.

### 16.7 Stage 2 best known config is rank=64, epoch=15
From checkpoint comparison on ImageNette. The checkpoint at step 12090 (epoch 15 of 20) gives the best quality/overfitting tradeoff.

---

## 17. Immediate next implementation work

Given current repo state (as of 2026-04-14):

1. **Resolve Stage 2 ↔ Stage 4 resolution mismatch**
   - Stage 2 LoRA trains at 512, Stage 4 generates at 1024
   - Options: retrain Stage 2 at 1024 (expensive), or generate at 512 and improve quality via other means
   - 1024 generation produces sharper images but LoRA features are 512-calibrated

2. **Tune HDBSCAN parameters for better mode discovery**
   - Current min_samples=3 is too low (creates micro-clusters like 2-member gas pump mode)
   - Need to experiment with min_samples and min_cluster_size on DINOv2 space
   - Both parameters now exposed as CLI args

3. **Improve generation quality**
   - Test SDXL refiner (implemented, not yet evaluated)
   - Tune guidance_scale (currently 9.0, may cause artifacts on human subjects)
   - Consider LoRA rank increase (currently 64, could try 128)
   - Address human body artifacts in captions with "being held" etc.

4. **ImageNet-1k full pipeline**
   - Stage 1 full run on ImageNet-1k is in progress on server
   - Once complete: Stage 2 → Stage 3 → Stage 4 → Eval on full 1000 classes
   - This is the target evaluation for the method

5. **Evaluation benchmarking**
   - Compare DINO K-Means vs DINO HDBSCAN distilled datasets via classifier accuracy
   - Compare against baselines (random selection, SRe2L, etc.)
   - Eval code is implemented (ConvNet-6, ResNet-18, ResNetAP-10)

---

## 18. Completed experiment log (for context recovery)

### Stage 1 optimization arc (2026-04-11 → 2026-04-12)
- **Problem**: ImageNette captions had 70.6% unique rate, heavy duplication (top: 209x), trailing "on/off" (1080 cases)
- **Fix 1** (render relaxation): unique rate → 83.7%, trailing on/off → 6
- **Fix 2** (normalization relaxation): background unknown rate dropped ~70%
- **Fix 3** (VLM unknown recovery): 511/903 unknown slots recovered via VLM re-examination
- **Fix 4** (prompt per-slot guidance): pending re-run evaluation
- **Final ImageNette**: 84.6% unique, 14.0 avg words, 85x top duplicate
- **ImageNet-1k 5-shot validation**: 99.7% unique across 4999/5000 records (1 render failure → fixed via archetype mapping)

### Stage 2 training arc (2026-04-10 → 2026-04-11)
- rank=16, epoch=20: learns style but overfits (color drift on french horn, spaniel limb issues)
- rank=64, epoch=5: better quality, less overfitting, 1h55m
- rank=64, epoch=5/10/15/20 comparison: **epoch=15 is best** — cleanest results, no overfitting
- All runs on ImageNette (12,894 pairs), 2 GPUs, resolution=512, lr=2e-5

### Stage 3 clustering arc (2026-04-14)
- **VAE K-Means** (baseline): uniform cluster sizes, works but doesn't discover real modes
- **VAE HDBSCAN**: 6-8/10 classes fell back to K-Means — VAE latent space too smooth for density-based clustering
- **DINOv2 added**: 768-dim CLS features with natural cluster structure
- **DINO K-Means**: uneven cluster sizes reflecting real density variation (max/min ratio 2.4x-9.0x)
- **DINO HDBSCAN** (min_cluster_size=15, min_samples=3, pca_dim=50): 9/10 classes discovered modes independently (only chain saw fell back). But gas pump had a 2-member micro-cluster — min_samples=3 too low.

### Stage 4 generation arc (2026-04-13 → 2026-04-14)
- **img2img + mean embedding**: all-black images (custom denoising loop incompatible with SDXL scheduler)
- **img2img + official pipeline + mean embedding**: blurry/gray (averaged embedding doesn't correspond to real caption)
- **img2img + representative caption**: better quality, but Stage 2 vs Stage 4 output mismatch (different generator device, call signature)
- **text2img + representative caption** (current recommended): matches Stage 2 inference quality exactly
- **text2img at 1024 + guidance 9.0**: sharper, but human anatomy artifacts on "being held" captions; resolution mismatch with 512-trained LoRA

### Server environment
- Path: `/media/4T_HDD/cai/cspd-dd/cspd-dd`
- GPU: 2x GPU, CUDA 12.1
- Conda env: `cspd-dd`
- ImageNette: `/media/4T_HDD/cai/datasets/ImageNette/train` (12,894 images, 10 classes)
- ImageNet-1k: `/media/4T_HDD/cai/datasets/ImageNet1k/train`
- ImageNet-1k 5-shot: `/media/4T_HDD/cai/datasets/ImageNet1k_5shot/train` (5,000 images, 1000 classes)
- Diffusers repo: `./diffusers` (pip install -e . required)

---

## 19. If this document needs updating later

When updating this file:
- prefer repo-truth over older chat memory,
- name implemented files explicitly,
- mark unimplemented stages honestly,
- separate “implemented now” from “planned next”,
- update stage status as soon as code lands,
- keep the boundary between Prep class-level metadata and Stage 1 archetype-aware downstream logic clear,
- explicitly note whether Stage 1B review is deterministic-only or deterministic+inline-VLM in the current repo state,
- and avoid reviving removed Stage 2 render compatibility entrypoints in the implementation-facing description unless they actually return to the repo.

This file should help a new coding agent continue the project without needing to reverse-engineer the entire repo from scratch.
