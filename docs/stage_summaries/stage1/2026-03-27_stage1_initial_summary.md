# Stage 1 Summary — initial executable version

Date: 2026-03-27  
Scope: CSPD pipeline Stage 1 status snapshot from current repo state

## 1) What Stage 1 was supposed to do
Stage 1 is the attribute extraction step.

Target behavior:
- read an ImageFolder-style dataset,
- map each class to a semantic archetype,
- choose the archetype-specific slot schema,
- run a VLM per image to fill only those slots,
- validate / normalize the returned payload shape,
- write incremental extraction artifacts that can survive long runs and resume cleanly.

Expected primary output per sample:
- image/sample metadata,
- chosen archetype + slot schema,
- extracted attribute dictionary,
- raw response / error info when useful.

## 2) What is implemented now
Core Stage 1 executable scaffold is in place and usable.

Implemented pieces:
- CLI entrypoint: `cspd-stage1 run`
- dataset ingestion for ImageFolder-style roots
- optional class-name mapping for synset-style folder names (`classes.json`)
- optional fixed `class -> archetype` mapping
- archetype-specific slot schemas in code
- pluggable VLM backend interface
- `mock` backend for plumbing/regression checks
- `qwen_local` backend for real local GPU inference with Qwen2.5-VL
- explicit JSON-oriented prompting with fallback parsing for messy pseudo-JSON output
- payload validation + slot-level normalization to expected schema keys
- retry logic on failed samples
- incremental JSONL flushing during long runs
- resume support: skip prior successes, retry items recorded in `failed_samples.jsonl`
- server helper scripts for env check, metadata prep, smoke run, and full workflow
- deterministic post-processing normalization script for Stage 1 outputs

## 3) Key implementation-relevant fixes already landed
These are the fixes / improvements that materially changed execution reliability or output usability.

1. **Fixed taxonomy flow**
   - Stage 1 now prefers a manually curated taxonomy (`configs/stage1/archetype_taxonomy_manual.json`) instead of open-ended taxonomy discovery.
   - This reduces drift in slot-schema selection and makes class-to-archetype assignment more stable.

2. **Constrained class-to-archetype mapping path**
   - Metadata prep scripts now assume a fixed taxonomy and support explicit `class_to_archetype.json` input.
   - Better fit for controlled production runs.

3. **Richer sample metadata in output rows**
   - Output rows now carry record ids, dataset-relative paths, raw/readable class names, backend/model info, archetype, and slot schema.
   - This makes downstream auditing and Stage 2 joining much easier.

4. **Incremental flush + resume behavior**
   - Partial results are visible during long runs.
   - Reusing an output directory no longer forces a restart: successful records are skipped and previously failed records are retried.

5. **Parser robustness improvements**
   - Handling for malformed/truncated VLM JSON was improved, including truncated arrays and relaxed list parsing.
   - This matters for local VLM execution where responses are not always perfectly formatted.

6. **Workflow/script hardening**
   - Smoke workflow defaults were reduced to a small subset.
   - Server helper scripts were added/fixed so environment check -> metadata prep -> load test -> single-image test -> smoke run -> final extraction is reproducible.

7. **Post-extraction normalization v1 + v2 refinement**
   - A deterministic normalization script and ruleset were added.
   - v2 specifically addresses several high-value failure modes: low-value background color contamination, person leakage into object slots, and simple mixed state phrases.

## 4) Current outputs / artifacts available
### Extraction artifacts
Per run, Stage 1 currently writes:
- `attributes.jsonl`
- `failed_samples.jsonl`
- `stage1_stats.json`

Current repo contains multiple mock/plumbing runs under `runs/`, including:
- `runs/stage1_smoke/`
- `runs/stage1_imagefolder_mock/`
- `runs/stage1_progress_check/`
- `runs/stage1_classmap_mock/`
- `runs/stage1_flush_mock/`
- `runs/stage1_mock_regression/`

These runs confirm the pipeline path for:
- dataset scan,
- class-name mapping,
- class/archetype handling,
- flushing,
- regression-level end-to-end execution.

### Metadata / config artifacts
Relevant checked-in metadata/configs:
- `configs/stage1/archetype_taxonomy_manual.json`
- `configs/stage1/class_to_archetype_imagenet1k_auto.json`
- `configs/stage1/normalization/stage1_attribute_normalization_rules.json`
- top-level `classes.json`

### Analysis / normalization utilities
- `scripts/data/analyze_attribute_values.py`
- `scripts/data/normalize_stage1_attributes.py`
- planning note: `docs/stage1_attribute_normalization_plan.md`

Normalization outputs (when the script is run on a real Stage 1 result file):
- `attributes_normalized.jsonl`
- `normalization_audit.jsonl`
- `normalization_review_queue.jsonl`
- `normalization_summary.json`
- `normalization_rules_snapshot.json`

## 5) Current quality / result status
Execution status:
- **Stage 1 extraction scaffold is operational.**
- **Local real-backend path exists (`qwen_local`) and server workflow scripts are in place.**
- **Artifact format is already good enough for downstream consumption.**

Quality status:
- mock/plumbing coverage is solid enough to trust the pipeline mechanics,
- extraction outputs now carry enough metadata for traceability,
- parser/retry/resume behavior is good enough for long local runs,
- deterministic normalization has moved from plan to implementation,
- **extraction + normalization v2 are good enough to proceed to Stage 2**, with the understanding that Stage 2 should consume normalized outputs conservatively and preserve review hooks rather than assuming perfect labels.

Practical reading: Stage 1 is not “final ontology solved”; it is “usable upstream producer with reviewable outputs”. That is enough for the next stage.

## 6) Known limitations
Keep these explicit so Stage 2 is designed defensively.

1. **Repo-visible run artifacts are mostly mock/plumbing runs**
   - The checked-in `runs/` directory does not yet show a large real extraction run artifact set.
   - Real qwen_local throughput/error profile still needs validation on the actual target dataset.

2. **Input assumption is still narrow**
   - Stage 1 assumes ImageFolder-style layout.
   - Other dataset organizations will need adapters or preprocessing.

3. **Archetype assignment still depends on external mapping quality**
   - The best path is explicit `class_to_archetype.json`.
   - If mapping quality is off, the chosen slot schema will be off too.

4. **Normalization is intentionally conservative**
   - Good for safety/auditability, but it will leave some lexical variance and review queue volume.
   - Do not assume all long-tail noise is removed.

5. **Some slot contamination/hallucination remains expected**
   - Background slots can still carry weak/noisy context.
   - Person mentions and mixed narrative phrases are reduced, not eliminated.
   - Wrong-object hallucinations in type-like slots remain possible.

6. **No evidence yet of end-to-end Stage 1 -> Stage 2 integration contract in code**
   - Artifact schema is usable, but the formal consumption contract for Stage 2 should now be frozen explicitly.

## 7) Handoff notes for Stage 2
Stage 2 should start from the current Stage 1 outputs as they are, not wait for a perfect extractor.

Recommended Stage 2 assumptions:
- primary consumption target should be **normalized Stage 1 outputs**,
- raw Stage 1 values should still be retained for audit/debug,
- rows with `normalization_review_required=true` should remain traceable and optionally down-weighted / flagged,
- Stage 2 should key joins by `record_id` and retain `class_name_raw`, `class_name`, `archetype`, and `slot_schema`.

Recommended Stage 2 input contract:
- required fields from extraction side:
  - `record_id`
  - `sample_id`
  - `relative_image_path`
  - `class_name_raw`
  - `class_name`
  - `archetype`
  - `slot_schema`
  - `attributes`
- preferred normalized additions:
  - `normalized_attributes`
  - `attribute_normalization`
  - `normalization_review_required`

Recommended immediate Stage 2 tasks:
1. Freeze the exact Stage 2 input schema against the normalized Stage 1 row format.
2. Decide whether Stage 2 consumes raw attributes, normalized attributes, or both (recommended: both, normalized-first).
3. Build explicit handling for review-required fields instead of silently trusting them.
4. Validate Stage 2 on one real normalized Stage 1 artifact bundle before scaling.
5. Keep normalization snapshots with every Stage 2 experiment for reproducibility.

## 8) Bottom line
Stage 1 initial version is past the “toy scaffold” stage.

What is true now:
- extraction pipeline exists and runs,
- metadata and artifact structure are serviceable,
- real local backend path exists,
- normalization v2 addresses the most obvious high-volume noise,
- **current Stage 1 output quality is good enough to unblock Stage 2 work**.

What is not true yet:
- Stage 1 is not fully production-hardened on a large real run,
- normalization is not a complete ontology solution,
- Stage 2 still needs to consume outputs defensively.
