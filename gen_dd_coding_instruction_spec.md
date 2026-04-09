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
- Supporting server scripts, metadata prep, mock/regression runs, and full workflow wiring exist.

### Not implemented yet
- **Later research stages after current Stage 2 planning/scaffold work** are not implemented in code yet.
  - Stage 3 visual clustering / mode discovery
  - Stage 3 visual anchor estimation
  - Stage 3 semantic anchor aggregation
  - Stage 3 semantic anchor rendering
  - Stage 4 dual-anchor conditioned distilled generation

### Partially implemented now
- **Stage 2 generative-backbone adaptation / canonical-caption conditioning alignment** now has a substantial scaffold in code, but not a runnable end-to-end real training loop yet.
- Implemented Stage 2 pieces now include:
  - image + Stage 1 canonical-caption pairing / manifest generation
  - Stage 2 run-directory setup and trainer-plan/config snapshot writing
  - diffusers-backed real FLUX/Kontext backbone load path for inspection
  - module targeting inspection utilities
  - optional LoRA-target planning / injection scaffolding on explicitly provided real modules
  - module-tree dump helpers for top-level pipeline components and transformer submodules
- Current Stage 2 default recorded policy in repo:
  - freeze non-transformer top-level modules
  - fine-tune the full `FluxTransformer2DModel`
  - if memory is insufficient, fall back to conditioning-related transformer submodules only
- Important boundary:
  - the repo records this policy honestly in Stage 2 metadata/artifacts,
  - but **executable real FLUX.1 Kontext / FLUX transformer fine-tuning is still not wired end-to-end yet**.

### Important practical reading
Right now, the repo is best understood as:
- a working **Prep** pipeline for class metadata,
- a working **Stage 1** pipeline consisting of extraction -> normalization -> render,
- where Stage 1 normalization is deterministic-first but can invoke constrained VLM review on ambiguous slots,
- plus planning/spec notes for later stages,
- but **not** a multi-stage end-to-end CSPD implementation yet.

### Packaging / environment reality check
- The installable project in `pyproject.toml` is currently named **`cspd-stage1`**.
- The console scripts exposed there are now:
  - **`cspd-stage1`** with Stage 1 subcommands such as `run`, `normalize`, and `render`
  - **`cspd-stage2`** for Stage 2 scaffold / inspection / planning commands
- The repo now also bundles `environment.yml` for the shared conda environment name **`cspd-dd`** used by the server shell helpers.

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
  - `training.py`
  - `data.py`
  - `backbone.py`
  - currently implements Stage 2 pairing / planning / backbone inspection scaffolding and partial training-path plumbing

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
- `scripts/server/README.md` documents the recommended Prep + Stage 1 helper flow

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

### Later planned stages
2. generative-backbone adaptation / canonical-semantic-space familiarization
3. visual mode discovery / clustering
4. visual anchor estimation
5. semantic anchor aggregation
6. semantic anchor rendering
7. dual-anchor conditioned distilled generation

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

### Important current policy boundary
Current engineering direction is:
- **Prep can use class identity**, because Prep is explicitly class-level metadata construction.
- **Stage 1 normalization/render should prefer archetype-aware rules**, not expanding class-specific hard patches.
- **If VLM is used downstream, it should appear as constrained review/fallback, not as a full class-aware rewrite layer.**

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
    "slot_name": "short phrase"
  }
}
```

The prompt explicitly asks for:
- JSON only,
- no markdown,
- short phrases rather than sentences,
- `unknown` for unclear values,
- `not_applicable` when needed.

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
Inline VLM review is not full-row and not full-dataset free rewriting.
It is only invoked for slots whose deterministic normalization metadata indicates ambiguity, i.e. slots with:
- `status == "review_required"`, or
- non-empty `review_reasons`

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
- conservative,
- auditable,
- and increasingly **archetype-aware rather than class-aware**.

That means render should keep using:
- archetype template families,
- slot-level drop rules,
- cleanup heuristics,
- class-name fallback only as a narrow anchor recovery path already present in the implementation,
- but avoid turning into a class-specific correction layer or a free-form VLM text generator.

---

## 12. What is deterministic vs learned right now

### Learned / model-driven
- Prep multimodal class-to-archetype mapping
- Stage 1A image-level attribute extraction
- Stage 1B inline VLM review for ambiguous slots only

### Deterministic / rule-driven
- Stage 1B first-pass normalization
- Stage 1C render

This separation is deliberate.
The repo currently uses VLMs where semantic proposal or ambiguity resolution is needed, while keeping the main downstream cleanup/render path auditable.

---

## 13. Later stages — current implementation status

### Visual clustering / prototype / generation / assembly stages
Status: **not implemented in repo**.

No current modules are present in the repo yet for:
- visual mode discovery / clustering
- visual prototype estimation
- semantic prototype aggregation
- semantic prototype rendering
- dual-prototype conditioned generation
- final distilled dataset assembly

---

## 14. What future coding agents should not get wrong

### 14.1 Do not misread the repo as multi-stage-complete
It is not.
Currently the repo is mostly:
- Prep metadata,
- Stage 1 extraction,
- Stage 1 normalization,
- Stage 1 render,
- helper scripts,
- planning notes.

### 14.2 Do not ignore render anymore
Render is implemented and belongs to Stage 1 workflow semantics.
Its canonical implementation now lives under `src/cspd_stage1/`, while `src/cspd_stage2/` is only future scaffolding.

### 14.3 Do not silently treat class-aware correction as the main downstream strategy
Current intended boundary is:
- class-aware logic is acceptable in Prep,
- downstream normalization/render should stay mostly archetype-aware and auditable.

### 14.4 Do not describe Stage 1B as purely deterministic anymore
Current Stage 1B is **deterministic-first with inline constrained VLM review by default**.
The deterministic result is preserved for auditability, but the effective result may include reviewed overrides.

### 14.5 Do not assume per-slot confidence/state objects exist
They do not exist in current Stage 1 extraction outputs.

### 14.6 Do not spec future stages against stale numbering
Current implementation-facing numbering is:
- Prep
- Stage 1A extraction
- Stage 1B normalization (+ inline review)
- Stage 1C render
- later research stages remain unimplemented

---

## 15. Immediate next implementation work

Given current repo state, the most sensible next coding work is:

1. **Use the current clean Stage 1 outputs as the fixed Stage 2 input surface**
   - the latest ImageNette normalization/render run has already been pushed to a no-render-failure baseline
   - Stage 1 should no longer be the main blocker unless a concrete upstream bug reappears

2. **Push Stage 2 from mixed-family scaffold status to a clean family-split implementation surface**
   - Stage 2 pairing / manifest / trainer-plan infrastructure is already implemented
   - FLUX and PixArt training / backbone logic have now been split into family subpackages
   - current engineering rule is: only neutral shared code should remain outside family packages
   - before substantial new SDXL work, keep reducing leftover mixed family logic in shared modules

3. **Immediate next mainline engineering target**
   - move the next serious Stage 2 investigation to **SDXL base 1.0**, not PixArt
   - treat Stage 2 as an image-to-image-flavored generative adaptation problem aligned with later visual+semantic distilled-data creation
   - use the current Stage 2 family-split refactor as the structural base for adding an SDXL family cleanly rather than patching old mixed modules
   - initial SDXL tuning hypothesis should prioritize **UNet LoRA** (attention-only first, broader UNet LoRA second if needed), not text encoder or VAE

4. **Keep method wording generic, implementation wording honest**
   - method level: generative-backbone adaptation / canonical-semantic-space familiarization
   - current exploratory legacy families in repo: FLUX and PixArt
   - next preferred implementation target family: **SDXL base 1.0**
   - current practical framing: move toward image-to-image-capable Stage 2 rather than staying on a text-to-image-only branch that mismatches the downstream project need

5. **Stage 3 remains future work**
   - visual mode discovery / clustering
   - visual anchor estimation
   - cluster-wise semantic aggregation
   - semantic anchor rendering

---

## 16. If this document needs updating later

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
