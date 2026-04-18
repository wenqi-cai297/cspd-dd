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
- **Stage 2 training** is implemented for SDXL LoRA (mainline, 63.27% baseline). Other families are out of scope.
- **Stage 2 inference / sampling** script is implemented for LoRA vs baseline A/B comparison.
- **Stage 3 mode discovery** is implemented: DINOv2 encoding + per-class HDBSCAN clustering (with internal K-Means fallback / sub-clustering) + medoid caption extraction.
- **Stage 4 distilled dataset generation** is implemented: SDXL LoRA text2img by default (current baseline) with an optional img2img-from-medoid path and an optional SDXL refiner pass.
- **Evaluation** is implemented: train classifier (ConvNet-6/ResNet-18/ResNetAP-10) on distilled dataset, evaluate on real val set.
- Supporting server scripts, metadata prep, mock/regression runs, and full workflow wiring exist.

### Not implemented yet
- FID evaluation is not yet automated.

### Important practical reading
Right now, the repo is best understood as:
- a working **Prep** pipeline for class metadata,
- a working **Stage 1** pipeline consisting of extraction → normalization → render,
- a working **Stage 2 SDXL LoRA** training pipeline that delegates to the official diffusers trainer,
- a working **Stage 2 inference** script for sampling from trained LoRA weights,
- a working **Stage 3** pipeline for DINOv2 encoding, per-class HDBSCAN mode discovery, and medoid caption extraction,
- a working **Stage 4** pipeline for text-to-image distilled dataset generation (with optional img2img-from-medoid and optional SDXL refiner),
- a working **Evaluation** pipeline for training classifiers on distilled datasets and evaluating on real validation sets,
- where Stage 1 normalization is deterministic-first but can invoke constrained VLM review on ambiguous slots.

### Packaging / environment reality check
- The installable project in `pyproject.toml` is currently named **`cspd-stage1`**.
- The console scripts exposed there are now:
  - **`cspd-stage1`** with Stage 1 subcommands `run`, `normalize`, `render`
  - **`cspd-stage2`** with the single `train` subcommand (delegates to the official SDXL LoRA trainer)
  - **`cspd-stage3`** with `encode`, `cluster`, `run` subcommands
  - **`cspd-stage4`** with the single `generate` subcommand (text2img default, optional `--visual-mode medoid` and `--refiner-model`)
  - **`cspd-eval`** with `run` and `run-all` subcommands for classifier training + validation
- The repo now also bundles `environment.yml` for the shared conda environment name **`cspd-dd`** used by the server shell helpers.
- Core dependencies: `torch`, `torchvision`, `numpy`, `tqdm`, `pillow`, `diffusers`, `transformers`, `accelerate`, `peft`, `sentencepiece`, `protobuf`, `tiktoken`, `safetensors`, `scikit-learn`, `hdbscan`, `qwen-vl-utils`.
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
  - `cli.py` — CLI with the `train` subcommand
  - `training.py` — Stage 2 orchestration: pairing → manifest → SDXL LoRA training dispatch
  - `training_common.py` — small helpers (`derive_stage2_output_dir`, `_safe_write_json`)
  - `data.py` — pairing + manifest writing
  - `backbone.py` — single helper `infer_backbone_family` (SDXL only)
  - `families/sdxl/backbone.py`, `families/sdxl/training.py` — SDXL LoRA trainer wrapper around the official diffusers script

- `src/cspd_stage3/`
  - `__init__.py`
  - `encode.py` — DINOv2 feature encoding (Stage 3A)
  - `cluster.py` — per-class HDBSCAN clustering + K-Means fallback/sub-clustering + medoid caption extraction (Stage 3B+3C)
  - `cli.py` — CLI with `encode`, `cluster`, `run` subcommands

- `src/cspd_stage4/`
  - `__init__.py`
  - `generate.py` — text2img distilled generation with optional img2img-from-medoid and optional SDXL refiner
  - `cli.py` — CLI with `generate` subcommand

- `src/cspd_eval/`
  - `__init__.py`
  - `train.py` — classifier training + evaluation (ConvNet-6, ResNet-18, ResNetAP-10)
  - `train_utils.py` — metrics (AverageMeter, accuracy), CutMix helpers (random_indices, rand_bbox), and the tensor-space ColorJitter + Lighting augmentations required to match the MGD³ reference eval
  - `models/convnet.py` — ConvNet-6 architecture
  - `models/resnet.py` — ResNet-18 architecture
  - `models/resnet_ap.py` — ResNetAP-10 architecture

### Inference scripts
- `scripts/stage2/sample_sdxl_lora.py` — SDXL LoRA sampling with baseline comparison support

### Config / metadata
- `classes.json`
- `environment.yml`
- `pyproject.toml`
- `configs/stage1/archetype_taxonomy_manual.json`
- `configs/stage1/class_to_archetype_imagenet1k_manual.json`
- `configs/stage1/normalization/stage1_attribute_normalization_rules.json`

### Data / analysis scripts
- `scripts/prep/convert_class_py_to_json.py`
- `scripts/prep/generate_class_to_archetype_map_vlm.py`
- `scripts/stage1/normalize_stage1_attributes.py`

### Server-side execution scripts
- `scripts/stage1/check_stage1_env.sh`
- `scripts/stage1/setup_cspd_stage1.sh`
- `scripts/prep/prepare_stage1_metadata.sh`
- `scripts/stage1/run_stage1_mock.sh`
- `scripts/stage1/run_stage1_qwen_local.sh`
- `scripts/stage1/run_stage1_normalization.sh`
- `scripts/stage1/run_stage1_render.sh`
- `scripts/prep/generate_class_to_archetype_vlm.sh`
- `scripts/stage2/check_stage2_sdxl_env.sh` — Stage 2 SDXL environment preflight
- `scripts/stage2/run_sdxl_stage2_official.sh` — SDXL LoRA training launcher (default: 2 GPUs, 512 resolution)
- `scripts/stage2/run_stage2_train.sh`
- `scripts/README.md` documents the recommended Prep + Stage 1 + Stage 2 helper flow
- `scripts/stage1/run_stage1_pipeline.sh` — full Stage 1: extract → normalize → render
- `scripts/stage2/run_stage2_pipeline.sh` — Stage 2 training + checkpoint sampling
- `scripts/stage3/run_stage3_pipeline.sh` — Stage 3 encode + cluster
- `scripts/stage4/run_stage4_pipeline.sh` — Stage 4 generate distilled dataset
- `scripts/eval/run_eval_pipeline.sh` — train classifier + evaluate
- `scripts/pipelines/run_full_pipeline.sh <train_root> [val_root] [nclass]` — end-to-end driver: Stage 1 → Stage 2 → Stage 3 → Stage 4 → Eval. Idempotent per stage (skips work that already exists on disk). `PIPELINE_IPC="10 20 50"` controls the IPC sweep at the end.
- `scripts/pipelines/run_baseline_3x3.sh <train_root> [val_root] [nclass]` — 3×3 measurement protocol (three paired seeds, each `(cluster, generate, eval)`). Assumes Stage 1 / 2 / 3A already done by the full-pipeline script; auto-detects the latest LoRA checkpoint under `STAGE2_BEST_EPOCH` (default 9).

### Stage 2 output-dir rule (must remember)
- The repo-standard Stage 2 run root is:
  - `runs/stage2/train/<dataset_label>/<backbone_slug>/<timestamp>`
- `scripts/stage2/run_stage2_train.sh` already derives this automatically.
- `cspd-stage2 train` should follow the same convention by default when `--output-dir` is omitted; do **not** force routine users to hand-type run directories.
- Dataset-label derivation rule:
  - default: `basename(dataset_root)`
  - if `dataset_root` ends with a split-only directory name in `{train,val,valid,validation,test,testing}`, use `<parent>_<split>`
- `--output-dir` remains only as an explicit override, not the normal required path.

### Stage 2 dataset-root contract (must remember)
- Use the exact Stage 1-compatible ImageFolder split root that the render records point at (e.g. `.../ImageNette/train`), not the parent dataset root. Passing the parent can collapse pairing to zero.

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

### Stage 3
- visual/semantic mode discovery via latent clustering
- encoding: VAE latents + CLIP text embeddings + DINOv2 CLS features
- clustering: HDBSCAN mode discovery on DINOv2 features (K-Means is the internal fallback / sub-clustering path)
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
- `scripts/prep/convert_class_py_to_json.py`
- `scripts/prep/generate_class_to_archetype_map_vlm.py`
- `scripts/prep/prepare_stage1_metadata.sh`
- `scripts/prep/generate_class_to_archetype_vlm.sh`

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
- `scripts/stage1/normalize_stage1_attributes.py`
- `configs/stage1/normalization/stage1_attribute_normalization_rules.json`

### Main CLI / helper surface
Preferred current entrypoints:
```bash
cspd-stage1 normalize --input ... --output-dir ...
```

```bash
bash scripts/stage1/run_stage1_normalization.sh <attr_dir_or_jsonl>
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
bash scripts/stage1/run_stage1_render.sh /path/to/attributes_normalized.jsonl
```

There is no current `cspd-stage2 render` CLI entrypoint and no `scripts/stage2/run_stage2_render.sh` helper in the repo.
The installable console script exposed by `pyproject.toml` is `cspd-stage1`.

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

## 12. Stage 2 — Diffusion model LoRA training

### Implementation status
**Implemented for SDXL LoRA only (mainline).**

### Core purpose
Train the diffusion model's UNet so it learns to recognize our Stage 1 canonical captions. Training pairs are `(real image, canonical_caption)` from Stage 1 render outputs.

### Backbone choice
- **SDXL** (`stabilityai/stable-diffusion-xl-base-1.0`): **primary**. Native 1024 but trained at 512 (resolution mismatch); 2.6B params; dual CLIP text encoders. LoRA fine-tuning via `train_text_to_image_lora_sdxl.py`. Current best: **63.27% ± 0.19** on ImageNette IPC=10 under the 3×3 protocol (checkpoint-7254, rank=64, epoch 9 with cosine LR).

### Architecture
Stage 2 delegates training to official diffusers training scripts. The repo owns:
- **pairing**: matching ImageFolder images to Stage 1C render `records.jsonl` by `record_id`
- **dataset materialization**: copying images + generating `metadata.jsonl` in diffusers imagefolder format
- **launch orchestration**: building the `accelerate launch` command with config translation
- **preflight checks**: validating environment, script resolution, dataset integrity

### Main code
- `src/cspd_stage2/families/sdxl/training.py` — SDXL LoRA materialization, command building, launch (**primary**)
- `src/cspd_stage2/training.py` — dispatch (SDXL only)
- `src/cspd_stage2/cli.py` — CLI with all SDXL-specific flags (`--sdxl-*`)
- `scripts/stage2/run_sdxl_stage2_official.sh` — server helper (SDXL)
- `scripts/stage2/check_stage2_sdxl_env.sh` — environment check

### Training configuration (current defaults, SDXL mainline)
- backbone: **`stabilityai/stable-diffusion-xl-base-1.0`**
- parameterization: LoRA, **rank=64** (UNet attention: `to_k`, `to_q`, `to_v`, `to_out.0`)
- resolution: **512** (non-native for SDXL, but empirically fine)
- GPUs: **2** (default `--sdxl-num-processes 2`)
- mixed precision: fp16
- gradient checkpointing: enabled
- lr: 2e-5, scheduler: **cosine**, warmup: **500 steps**
- noise offset: **0.05** (improves contrast/brightness range)
- Min-SNR gamma: **5.0** (balances loss weighting across timesteps)
- VAE + text encoders: frozen
- `--report_to` is omitted (not `"none"`) to avoid accelerate tracker init errors
- **best checkpoint**: epoch 9 → `checkpoint-7254`

### CLI usage
```bash
cspd-stage2 train \
  --dataset-root /path/to/ImageNette/train \
  --render-input /path/to/records.jsonl \
  --adapter-rank 64 \
  --batch-size 8 --epochs 9 \
  --resolution 512
```

`--backbone-name` defaults to `stabilityai/stable-diffusion-xl-base-1.0` (the only supported family). `--training-parameterization` / `--trainable-component-group` / `--wandb` / `--sample-*` / `--pixart-*` / `inspect-targets` / `dump-modules` / `sample-baseline` were removed in the 2026-04-18 Stage 2 cleanup.

### Server helper usage
```bash
bash scripts/stage2/run_sdxl_stage2_official.sh \
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
├── stage2_config_snapshot.json
├── stage2_run_summary.json
├── train_manifest.jsonl
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
- `scripts/stage2/sample_sdxl_lora.py`
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

## 14. Stage 3 — Mode discovery via DINOv2 clustering

### Implementation status
**Implemented in repo. Current baseline: HDBSCAN + medoid caption, 63.27% ± 0.19 on ImageNette IPC=10 (3×3 protocol, best-of-3 per seed).**

### Core purpose
Discover representative modes per class via DINOv2 clustering. Each mode's medoid sample contributes its canonical caption to Stage 4 (one caption per mode generates one image).

### Architecture

```
Stage 3A: Encode
  images → DINOv2 (dinov2_vitb14) → CLS features (N, 768)


Stage 3B: Cluster + Extract
  per class: HDBSCAN discovers natural density modes, allocates IPC
             proportionally; K-Means is used internally as the <=1-mode
             fallback and as the sub-clustering strategy.
  per cluster:
    medoid = real sample closest to the DINOv2 cluster centroid
    representative_caption = medoid's canonical caption
```

### Main code
- `src/cspd_stage3/encode.py` — DINOv2 feature encoding
- `src/cspd_stage3/cluster.py` — per-class HDBSCAN clustering + K-Means fallback/sub-clustering + medoid caption extraction
- `src/cspd_stage3/cli.py` — CLI with `encode`, `cluster`, and `run` subcommands

### CLI usage
```bash
# Encode once
cspd-stage3 encode \
  --dataset-root /path/to/ImageNette/train \
  --render-input /path/to/records.jsonl \
  --output-dir runs/stage3/.../encoded

# Cluster (HDBSCAN is the only method; the flag was removed 2026-04-18)
cspd-stage3 cluster \
  --encode-dir runs/stage3/.../encoded \
  --output-dir runs/stage3/.../modes_hdbscan \
  --ipc 10
```

### Clustering parameters
- **--min-cluster-size** (HDBSCAN): minimum points for a real cluster split. Default 15.
- **--min-samples** (HDBSCAN): k-NN for core distance. Default 3.
- **--pca-dim** (HDBSCAN): PCA dims, 0 to skip. Default 50. Reduces DINOv2 768-dim before HDBSCAN.


### HDBSCAN mode discovery flow
1. Optional PCA dimensionality reduction (seeded)
2. HDBSCAN discovers natural density modes (no preset K; deterministic given input)
3. Noise points assigned to nearest discovered mode
4. If 0-1 modes found → fallback to seeded K-Means with K=IPC
5. If modes > IPC → farthest-point sampling selects IPC most diverse modes; unselected mode members are absorbed into the nearest selected mode
6. If modes ≤ IPC → proportional IPC allocation (every mode gets ≥1 slot); parents that receive >1 slot are sub-clustered with seeded K-Means
7. Medoid caption from each final cluster is the representative caption

### DINOv2 encoding
- Model: `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`
- Input: images resized to 224×224, ImageNet normalization
- Output: CLS token features (N, 768)
- Purpose: DINOv2 produces semantically rich features with natural cluster structure, better suited for mode discovery than VAE latents (which are smooth and high-dimensional).

### Output artifacts
```text
runs/stage3/<output_dir>/
├── encoded/
│   ├── dino_embeds.pt              # (N, 768) DINOv2 CLS features
│   └── encode_index.json           # per-sample metadata + provenance
└── modes_hdbscan/
    ├── modes_index.json            # per-mode metadata (captions, cluster_sizes, weight, density)
    └── stage3_summary.json         # method + hyperparameters + provenance + per-class diagnostics
```

### Key design decisions
- **DINOv2 for clustering**: architecture-agnostic 768-dim features.
- **Medoid caption**: the real sample closest to the cluster centroid contributes its canonical caption. Jaccard-distance "caption diversification" was tested and removed on 2026-04-18 (it hurt accuracy).
- **HDBSCAN + internal K-Means**: HDBSCAN is the top-level method; K-Means remains as the fallback and as the sub-clustering strategy inside the HDBSCAN allocator.

---

## 15. Stage 4 — Distilled dataset generation

### Implementation status
**Implemented in repo. Current baseline: SDXL LoRA + text2img, 63.27% ± 0.19 on ImageNette IPC=10 (3×3 protocol).**

### Core purpose
Generate the final distilled dataset. Default is text-to-image: use Stage 3 representative caption as prompt, generate one image per mode via Stage 2 LoRA-tuned SDXL.

### Generation flow (default: text2img)
```
Stage 3 mode → representative_caption (from medoid)
  → SDXL text2img pipeline + Stage 2 LoRA
  → distilled image (PNG)
```

### Main code
- `src/cspd_stage4/generate.py` — generation orchestration (text2img default, optional img2img-from-medoid, optional SDXL refiner)
- `src/cspd_stage4/cli.py` — CLI with the `generate` subcommand
- `scripts/stage4/run_stage4_pipeline.sh` — server pipeline script

### CLI usage
```bash
cspd-stage4 generate \
  --modes-dir runs/stage3/.../modes_hdbscan \
  --lora-weights runs/stage2/.../checkpoint-7254/pytorch_lora_weights.safetensors \
  --output-dir runs/stage4/.../output
```

### Key parameters
- **--visual-mode**: `"none"` (default, baseline) for text2img. `"medoid"` for img2img starting from the real medoid image.
- **--strength**: Img2img denoising strength. Default `0.8`. Only used when `--visual-mode medoid`.
- **--resolution**: Output image resolution. Default `512` (matches the Stage 2 LoRA training resolution).
- **--guidance-scale**: CFG strength. Default `7.5`.
- **--num-inference-steps**: Diffusion sampling steps. Default `50`.
- **--refiner-model**: Optional SDXL refiner model ID. When set, runs a refiner pass after the base generation for added detail / sharpness.
- **--refiner-strength**: Denoising strength for the refiner pass (0-1). Default `0.3`.

### Output artifacts
```text
runs/stage4/<dataset>/<ipc>/<lora_tag>/<timestamp>/
├── images/
│   ├── <class_raw>/
│   │   ├── <class_raw>_mode000.png
│   │   └── ...
│   └── ...
├── distilled_metadata.json    # per-image metadata with mode info
└── stage4_summary.json        # generation summary
```

### Design decisions
- **Text2img is the default**: eval showed text2img significantly outperforms img2img for classifier training accuracy. Img2img kept for ablation only.
- **Per-image seeding, shared base per round**: image `i` uses `torch.Generator().manual_seed(base_seed + mode_idx)`. The 3×3 protocol varies `base_seed` across three rounds; the `base_seed=42` round reproduces the pre-3×3 baseline dataset byte-for-byte.
- **ImageFolder output structure**: images organized by class for downstream classifier training.
- **SDXL refiner**: optional second pass that can be chained to either text2img or img2img.

### Evolution of generation strategy (historical; cleanup summary at end)
1. img2img + mean embedding → all-black (custom loop incompatible with SDXL)
2. img2img + mean embedding + official pipeline → blurry (averaged embedding not a real caption)
3. img2img + representative caption → quality OK but Stage 2 vs 4 mismatch
4. text2img + representative caption → matches Stage 2 inference, best eval accuracy
5. img2img from medoid + representative caption → more diverse but eval accuracy significantly worse (kept as `--visual-mode medoid` for ablation)
6. text2img + caption diversity selection (Jaccard greedy) → hurt accuracy; **code removed 2026-04-18**
7. text2img + mode guidance (MGD³-style) → failed: detailed captions dominate, no usable sweet spot (see 16.11); **code removed 2026-04-18**
8. text2img + multi-candidate selection, per-mode (DINOv2 prototype + diversity) → did not beat single-medoid baseline at IPC=10; **code removed 2026-04-18**
9. **text2img + HDBSCAN + medoid caption, 3×3 protocol** → current baseline (63.27% ± 0.19 on ImageNette IPC=10)
10. text2img + multi-candidate set-level moments, DINOv2 L2-normalized → 59.53% ± 0.38 (−2.80%); **code removed 2026-04-18**
11. text2img + multi-candidate set-level moments, VAE latents (16384-dim, no L2-norm) → 59.07% ± 0.25 (−3.26%); **code removed 2026-04-18**

---

## 16. What future coding agents should not get wrong

### 16.1 All optimization must be archetype-level, never class-level
This is a hard methodological boundary. Do not write normalization rules, render drop rules, or prompt guidance that references specific class names or class-level statistics. The only place class identity is used is in Prep (class-to-archetype mapping). Everything downstream operates on archetype + slot name only.

### 16.2 All stages + evaluation are implemented
The repo covers Prep, Stage 1 (1A+1B+1C), Stage 2 (SDXL LoRA + inference), Stage 3 (DINOv2 encoding + HDBSCAN clustering + medoid caption), Stage 4 (text2img distilled generation, optional img2img-from-medoid, optional refiner), and Evaluation (classifier training + accuracy measurement). FID evaluation is not yet automated.

### 16.8 Do not enrich captions at Stage 3 level
Stage 3 VLM recaption (enriching only medoid captions) was tested and degraded accuracy by ~5% because enriched captions are out-of-distribution for the LoRA trained on template captions. If caption enrichment is ever revisited, it must be applied at Stage 1 so the LoRA trains on the same caption distribution Stage 4 generates from.

### 16.9 Caption format and LoRA training must stay in sync
The LoRA can only generate well from captions in the same format it was trained on. Changing caption format in later stages without retraining the LoRA will degrade generation quality.

### 16.3 Stage 1A prompt now uses per-slot guidance
The prompt template no longer uses generic `"short phrase"` placeholders. Each slot has specific guidance with examples and anti-patterns defined in `SLOT_GUIDANCE` dict in `prompting.py`. This was added on 2026-04-12.

### 16.4 Stage 1B VLM review now triggers on mapped-to-unknown slots
VLM review is no longer limited to `review_required` slots. It also triggers `review.mapped_to_unknown_recovery` on all slots that normalization mapped to unknown, giving the VLM a chance to provide a better value by looking at the image.

### 16.5 Archetype mapping was revised on 2026-04-12
20 classes in `class_to_archetype_imagenet1k_manual.json` were remapped. If reusing old Stage 1A extraction results, check whether any affected classes are present — those records need re-extraction because the slot schema changed.

### 16.6 Do not pass --report_to none to the official SDXL script
Latest accelerate rejects `"none"` as an unsupported tracker. The repo already handles this by omitting `--report_to` when the value is `"none"`.

### 16.7 Stage 2 best known config is rank=64, epoch=9
From the 2026-04-14 → 2026-04-15 checkpoint sweep over epochs 5–15 with cosine LR on ImageNette: epoch 9 (step 7254) gave the best eval accuracy; epochs 10–15 did not help and started overfitting. `run_full_pipeline.sh` therefore trains **9 epochs total** by default (`STAGE2_EPOCHS=9`) and consumes the final checkpoint. Historical `checkpoint-7254` on disk from older 15-epoch runs is still valid — it was produced by a 15-epoch cosine schedule that happened to pass through the same step, and the baseline 63.27% number comes from that checkpoint.

### 16.10 Non-SDXL generative backbones underperformed or were never usable
Settled 2026-04-18. SD v1.5 full fine-tuning was tested end-to-end and eval'd worse than SDXL LoRA (61.3% vs 62.33% at the time). FLUX.1 and PixArt-Sigma never had a working training path on our stack despite the exploratory code. All three family subpackages (`families/{sd15,flux,pixart}`) and their server helpers were removed from the repo in the 2026-04-18 cleanup; the rule is simply: Stage 2 uses SDXL LoRA only.

### 16.11 Mode guidance (MGD³-style) is incompatible with detailed captions
Tested on 2026-04-16: MGD³ latent centroid guidance works when text conditioning is weak (class name only) but fails with our detailed structured captions. With strong text conditioning (CFG=7.5 + detailed caption), the UNet locks onto the caption's content. Mode guidance either has no effect (scale ≤ 0.1) or destroys image quality (scale ≥ 0.2). There is no sweet spot. The fundamental issue: text conditioning and latent guidance compete for control over the same features. MGD³ works because its text is weak ("tench"), leaving room for guidance. Our text is strong ("a brown speckled long and flat body tench being held in riverbank..."), leaving no room. Code removed on 2026-04-18 (`mode_guidance.py`, `--mode-guidance-scale`, `--mode-guidance-stop-step`, and the `use_mode_guidance` branches in `generate.py`).

---

## 17. Immediate next implementation work

Method is locked in as of 2026-04-18 (HDBSCAN + medoid text2img SDXL LoRA; 3×3 protocol at IPC=10 → 63.27% ± 0.19). The repo has just finished a deep cleanup pass that removed every experimental side-branch. The remaining to-do list is entirely about running experiments, not building new machinery:

1. **IPC sweep on the 3×3 baseline (in progress)**
   - Protocol: for each seed in {42, 123, 456}, re-cluster Stage 3 → Stage 4 generate with `base_seed + mode_idx` per image → eval × 3 repeats. Aggregation: best-of-3 per seed, then mean/std/min/max across the 3 per-seed bests.
   - **IPC=10 done: 63.27% ± 0.19** (per-seed bests 63.4 / 63.0 / 63.4; replaces the old single-run 62.33%).
   - **IPC=20 and IPC=50 pending**. Run:
     ```bash
     IPC=20 bash scripts/pipelines/run_baseline_3x3.sh /path/to/ImageNette/train
     IPC=50 bash scripts/pipelines/run_baseline_3x3.sh /path/to/ImageNette/train
     ```
   - Compare against published IPC-scaling baselines (MGD³, DD-VLCP, RDED, SRe2L).

2. **Multi-architecture benchmarking**
   - Run all three eval architectures (ConvNet-6, ResNet-18, ResNetAP-10), 3 repeats each.
   - The eval pipeline already supports this via `cspd-eval run-all` or `bash scripts/eval/run_eval_pipeline.sh ... all`. Report mean ± std, not just ResNetAP-10.

3. **ImageNet-1k full pipeline**
   - Stage 1 full run on ImageNet-1k is in progress.
   - Once it finishes, point the full-pipeline driver at it:
     ```bash
     bash scripts/pipelines/run_full_pipeline.sh /path/to/ImageNet1k/train
     ```
   - Idempotent stage-by-stage skipping means this tolerates interruptions.

4. **Novel method exploration** (Phase 4 from plan.md)
   - Early vision-language fusion (EVLF-style) — lightweight visual-semantic adapter.
   - The biggest research-value direction remaining after Phase 2 (per-mode prototype+diversity) and Phase 3 (set-level distribution matching) were exhausted at IPC=10 and the code was removed.

### Closed (for now)
- **Phase 2 multi-candidate selection**, **Phase 3 set-level selection**, and **MGD³-style mode guidance** were all tested, all regressed vs the single-medoid baseline at IPC=10, and their code was removed on 2026-04-18. See §18 experiment log for the numbers. If IPC=20 / IPC=50 ever behave differently enough to reopen any of these, the code can be restored from git history.

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

### Stage 4 generation arc (2026-04-13 → 2026-04-15)
- **img2img + mean embedding**: all-black images (custom denoising loop incompatible with SDXL scheduler)
- **img2img + official pipeline + mean embedding**: blurry/gray (averaged embedding doesn't correspond to real caption)
- **img2img + representative caption**: better quality, but Stage 2 vs Stage 4 output mismatch (different generator device, call signature)
- **text2img + representative caption**: matches Stage 2 inference quality exactly
- **text2img at 1024 + guidance 9.0**: sharper, but human anatomy artifacts on "being held" captions; resolution mismatch with 512-trained LoRA
- **img2img from medoid (strength=0.8)**: more diverse images but eval accuracy significantly worse than text2img
- **Conclusion**: text2img > img2img for classifier training; K-Means > HDBSCAN for mode selection
- **Stage 3 recaption experiment**: VLM re-captioned medoid images with free-form descriptions → accuracy dropped from ~62% to ~57%. Cause: enriched captions are OOD for the LoRA trained on template captions. Not pursued further; Stage 1 caption distribution is the committed design.

### Stage 2 cosine LR training arc (2026-04-14 → 2026-04-15)
- Added cosine LR, warmup=500, noise_offset=0.05, snr_gamma=5.0
- Epoch sweep (5-15): **epoch 9 is best** with cosine LR (peaks earlier than constant LR's epoch 15)
- Checkpoint-7254 selected as the standard LoRA

### Mode guidance experiment arc (2026-04-16)
- **Goal**: combine structured caption conditioning (text) + VAE latent centroid guidance (visual) — a combination not in prior work
- **Implementation**: EulerModeGuidanceScheduler subclasses EulerDiscreteScheduler, injects guidance in step()
- **Attempt 1** (manual denoising loop): all-black images — SDXL scheduler incompatibility
- **Attempt 2** (callback_on_step_end): gray images — no access to pred_x0, sigma too large
- **Attempt 3** (custom scheduler step()): images OK but guidance has no content effect at scale=0.1
- **Root cause**: DINOv2 cluster VAE means are too similar → switched to VAE-space K-Means centroids
- **Attempt 4** (VAE-native centroids): images OK, color/contrast changes but still no content diversity
- **Scale sweep** (0.1 → 0.18): either no effect or image quality collapses, no sweet spot
- **Conclusion**: mode guidance is fundamentally incompatible with detailed text conditioning. Strong CFG + detailed caption locks content; guidance can only affect low-level features (color/contrast) before breaking. MGD³ works because its text is weak ("tench"), ours is strong ("a brown speckled long and flat body tench being held in...").

### Multi-candidate selection experiment v1 (2026-04-16, SDXL)
- **Approach v1**: DINOv2 linear probe (discriminative) + cosine diversity
- **Results**: 10 candidates, beta=0.5 → 58.3%; beta=0.0 → 60.8% (both worse than baseline 61.3%)
- **Root cause**: proxy classifier (DINOv2 probe) doesn't match eval classifiers; architecture-specific bias
- **Approach v2** (implemented, not yet tested): architecture-agnostic scoring with prototype similarity (cosine to class mean DINOv2) + diversity (cosine distance to accepted set). No proxy classifier. IPC-dependent beta (0.3/0.5/0.7 for IPC=10/20/50).
- Inspired by D³HR (representativeness), IGDS (IPC-dependent balance), DAP (feature-space alignment)

### SD v1.5 backbone experiment (2026-04-16 → 2026-04-17, resolved)
- **Hypothesis**: SDXL at non-native 512 is the bottleneck; SD v1.5 (native 512, ~860M params, DD-VLCP validated) should beat it.
- **Implemented**: full fine-tuning of SD v1.5 UNet via `train_text_to_image.py` (not LoRA). Stage 4 auto-detects SD v1.5 vs SDXL and loads via `from_pretrained`.
- **Training**: 8 epochs, batch=8, 2 GPUs, cosine LR, noise_offset=0.05, snr_gamma=5.0.
- **Result**: sample quality looks good visually but eval accuracy was **worse** than SDXL baseline (61.3%). Hypothesis falsified.
- **Conclusion**: SDXL LoRA remains mainline. SDXL's dual CLIP text encoders + larger capacity compensate for non-native resolution.
- **PixArt-alpha considered**: 256 native + text-conditional, but no DD validation and poor fine-tuning ecosystem. Not tested.
- **Follow-up (2026-04-18)**: SD v1.5, FLUX, and PixArt family subpackages + their server helpers removed from the repo entirely; see 16.10.

### Candidate selection v2 + representativeness scoring (2026-04-16)
- **Phase 2 (candidate selection v2)**: architecture-agnostic scoring — prototype similarity (cosine to class mean DINOv2) + diversity (cosine distance to accepted set). No proxy classifier. IPC-dependent beta.
- **Phase 3 (representativeness scoring)**: set-level evaluation after generation
  - MMD with linear kernel (DAP ICLR 2026, Table 8: linear > RBF)
  - Moment matching: mean + std + 0.1×skewness (D³HR ICML 2025, exact formula from `evaluate_distribution_batch`)
  - Coverage: diagnostic metric (not from a specific paper)
  - Composite: 0.4×MMD + 0.4×moments + 0.2×coverage
  - Gap detection: `find_gap_modes()` identifies which modes need regeneration
- **Enriched mode metadata**: Stage 3 now outputs per-mode weight, density, DINOv2 centroid
- **Paper alignment verified**: D³HR source code (GitHub), DAP paper (arXiv), RDED source code (GitHub)
- **Status**: Phase 2 alone did not improve on the single-medoid baseline at IPC=10. Phase 3 scoring path was kept briefly as a diagnostic (`--eval-representativeness`) but removed with the rest of Phase 3 on 2026-04-18.

### Phase 3 refinement — set-level candidate selection (2026-04-17)
- **Motivation**: Phase 3 scoring alone was unused in the mainline. Upgraded to a *refinement* loop that couples Phase 2 and Phase 3: generate N candidates per mode, then pick the set that best matches the real class distribution.
- **Implementation**: `RepresentativenessScorer.select_set_greedy()` — greedy per-class selection with a 1-per-mode constraint (preserves Stage 3 mode structure). Two objectives:
  - `moments`: D³HR-style `‖μ_synth − μ_real‖ + ‖σ_synth − σ_real‖ + 0.1·‖skew_synth‖`
  - `mmd`: DAP-style linear-kernel MMD² between real and synthetic feature sets
  - Both operate on L2-normalized DINOv2 features (fixes the earlier normalization mismatch between `dino_embeds.pt` and `scorer.encode_image`).
- **Wiring**: new CLI flags `--set-level-selection` and `--set-objective {moments,mmd}` in `cspd-stage4 generate`; writes `set_level_selection_report.json` alongside `distilled_metadata.json`.
- **Design decisions**:
  - 1-per-mode constraint (not free-pool) so IPC stays balanced across modes.
  - Greedy in original mode listing order, which for HDBSCAN is cluster-id order (i.e. discovery order from the HDBSCAN label assignment), **not** weight-desc. Processing large modes first is likely better in theory (more slack to compensate downstream); this is a candidate follow-up.
  - Requires `--num-candidates > 1` and `--visual-mode none`; raises on misuse.
- **A/B #1 (2026-04-17, DINOv2 CLS space, L2-normalized, `moments`, N=10, HDBSCAN modes, ResNetAP-10 × 3)**: **59.53% ± 0.38** (59.0, 59.8, 59.8) — **−2.80% vs 62.33% baseline**. `distilled_dir` = `runs/stage4/ImageNette_train/ipc10/lora/setlevel_moments_n10_2026-04-17_195030`.
- **A/B #2 (2026-04-17, SDXL VAE latent space, unnormalized, 16384-dim, `moments`, N=10, same modes/seed)**: **59.07% ± 0.25** (59.4, 59.0, 58.8) — **−3.26% vs baseline**. `distilled_dir` = `runs/stage4/ImageNette_train/ipc10/lora/setlevel_vae_moments_n10_2026-04-17_222847`. Changing to the model-native feature space and dropping L2-norm did **not** help; if anything it regressed slightly further (both differences are well beyond each other's std, but practically similar).
- **Interpretation**: both regressions are consistent (std ≤ 0.38 across 3 repeats). Since the feature space swap addressed the two a-priori most-likely causes (proxy-space mismatch + L2-normalization wiping magnitude) and the result did **not** move toward the baseline, the bottleneck is **not** the feature space. The most likely remaining explanation is the objective itself:
  - Medoid baseline: each mode contributes the real image closest to its own cluster centroid → diversity comes from the mode structure (inter-mode variation).
  - Set-level moment matching: greedy pulls the whole set toward the *class* mean/std. The first pick anchors near the class centroid, later picks compensate but the inter-mode spread gets smoothed out, producing a more homogeneous set than medoid → worse classifier training signal.
- **Status — set-level line closed and code removed 2026-04-18**. Both feature spaces regressed at IPC=10, the feature-space swap did not change direction, and the objective itself (greedy class-mean matching) was judged harmful to inter-mode diversity. `representativeness.py` / `candidate_selection.py` / `mode_guidance.py` deleted from the repo along with the corresponding CLI flags.

### Eval output layout (2026-04-18)
Eval runs now write their JSON into a hierarchical directory that mirrors Stage 4, so results are traceable to their distilled source without opening the JSON:

```text
runs/eval/<dataset>/ipc<IPC>/<arch>/<stage4_tag>/<eval_timestamp>/eval_<arch>.json
```

`<stage4_tag>` is computed from the Stage 4 `distilled_dir` path by stripping `runs/stage4/<dataset>/ipc<IPC>/` and joining the remaining segments with `__` — for example:

| Stage 4 output | `<stage4_tag>` |
| --- | --- |
| `runs/stage4/ImageNette_train/ipc10/lora/2026-04-17_150048/images` | `lora__2026-04-17_150048` |
| `runs/stage4/ImageNette_train/ipc10/lora/baseline_3x3_TS/gen_seed42/images` | `lora__baseline_3x3_TS__gen_seed42` |

Old flat layout (`runs/eval/<timestamp>_ipc<IPC>_<arch>/eval_<arch>.json`) still on disk from pre-2026-04-18 runs is untouched; the 3×3 aggregator in `scripts/pipelines/run_baseline_3x3.sh` accepts both old and new layouts when looking up per-seed eval files.

### Repo cleanup pass (2026-04-18)
With the method locked in, the repo went through a multi-stage cleanup to drop every experimental side-branch and leave only the code that the locked-in pipeline actually reaches. Summary of what was removed:

- **Stage 1D** (VLM caption enrichment): never integrated. Deleted `src/cspd_stage1/enrich.py` and the `cspd-stage1 enrich` subcommand.
- **Stage 1 obsolete scripts**: sidecar VLM review (`scripts/data/review_normalization_with_vlm.py` + its shell wrapper), the pre-consolidation `run_stage1_full_workflow.sh`, VLM smoke tests (`scripts/vlm/test_*.py`), and the one-off `analyze_attribute_values.py`.
- **Stage 2 non-SDXL families**: `families/{sd15,flux,pixart}/` subpackages, `mock_backbones.py`, their server helpers, and all the related CLI flags / config fields. The self-built LoRA injection machinery (`LoRALinearAdapter`, `inject_lora_adapters`, `inspect_target_modules`, `apply_trainable_parameter_selection`) and the orphan PixArt/FLUX dispatch wrappers / placeholder loop in `training.py` went with them. `training.py` went from 1839 lines to ~190; `backbone.py` from 759 to 18; `cli.py` from 814 to ~145; `training_common.py` from 272 to ~50. `inspect-targets`, `dump-modules`, `sample-baseline` CLI subcommands removed.
- **Stage 3**: `_diversify_captions` + `--diversify-captions` CLI flag (tested, hurt accuracy); `--cluster-method` flag (HDBSCAN is the only top-level method, K-Means kept as internal fallback); `mode_centroids.pt` generation + VAE-latent threading (only consumer was the now-deleted mode guidance).
- **Stage 4**: `candidate_selection.py` (Phase 2 per-mode scorer), `representativeness.py` (Phase 3 set-level scorer), `mode_guidance.py` (MGD³-style scheduler). Related CLI flags (`--num-candidates`, `--candidate-beta`, `--candidate-probe-dir`, `--eval-representativeness`, `--set-level-selection`, `--set-objective`, `--set-feature-space`, `--mode-guidance-*`). `generate.py` went from 814 to ~320 lines; kept text2img + img2img-medoid + optional refiner.
- **Stage 3 VAE encoding** (`--encode-vae` / `--vae-model-name` flags, the VAE branch of `encode.py`): last consumer (Stage 4 set-level VAE feature space) is gone.
- **Eval `train_utils.py`**: unused `Logger`, `TimeStamp`, `Plotter`, `CutOut`, custom `Normalize`, `dist_l2`, `get_time`. Kept only what `train.py` imports plus the internals `ColorJitter` composes.
- **Pipeline scripts**: `run_ipc_sweep.sh`, `run_candidate_sweep.sh`, `run_setlevel_phase3.sh` deleted. `run_full_pipeline.sh` and `run_baseline_3x3.sh` rewritten to accept `<train_root> [val_root] [nclass]` as positional arguments (instead of hardcoded ImageNette paths) with sensible env overrides.
- **Script layout**: `scripts/server/` flattened; everything now lives under stage-specific folders (`scripts/{prep,stage1,stage2,stage3,stage4,eval,pipelines}/`).

Cumulative net change across all cleanup commits: roughly **−7700 lines** of source + scripts + spec dead weight.

### Current best configuration (as of 2026-04-18)
- **Stage 2**: SDXL rank=64 LoRA, cosine LR (2e-5), warmup=500, noise_offset=0.05, snr_gamma=5.0, epoch 9 (checkpoint-7254)
- **Stage 3**: DINOv2 HDBSCAN + medoid caption (no diversity selection), IPC=10
- **Stage 4**: text2img (visual_mode=none), resolution=512, guidance=7.5, steps=50
- **Baseline accuracy (new 3×3 protocol, IPC=10, 2026-04-18)**: **63.27% ± 0.19** on ImageNette (ResNetAP-10). Per-seed best-of-3: seed=42 → 63.4, seed=123 → 63.0, seed=456 → 63.4; min=63.0, max=63.4. This replaces the old 62.33% ± 1.47 as the comparison baseline going forward.
- **Baseline accuracy (old protocol, for reference)**: 62.33% ± 1.47 (HDBSCAN + medoid, single generation × mean of 3 eval repeats, per-image `seed + mode_idx` seeding). Note: the new seed=42 round (63.4) and the old 62.33 are **on the same dataset** — the only differences are (a) old used mean of 3 repeats, new uses max of 3; (b) old reported the population std of the 3 repeats, new reports spread across the 3 per-seed bests.
- **Key insights (empirical)**:
  - Medoid caption (default) > diversity selection — kept medoid as default, diversity opt-in
  - HDBSCAN ≈ K-Means for medoid caption selection at IPC=10
  - Mode guidance (MGD³-style) incompatible with detailed text conditioning (see 16.11)
  - Multi-candidate selection with DINOv2 probe/prototype doesn't beat baseline (Phase 2)
  - Multi-candidate **set-level** selection with D³HR moments **regresses** in both DINOv2 space (59.53% ± 0.38, −2.80%) and VAE latent space (59.07% ± 0.25, −3.26%) at IPC=10. Feature-space swap did not change the direction → the objective itself (greedy class-mean matching) over-smooths inter-mode diversity at low IPC (Phase 3, 2026-04-17)
  - SD v1.5 full fine-tuning underperforms SDXL LoRA (see 16.10)
- **Eval**: standard protocol — RRC → ToTensor → HFlip → custom ColorJitter → Lighting → Normalize

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
