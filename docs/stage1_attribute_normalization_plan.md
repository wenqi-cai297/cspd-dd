# Stage 1 Attribute Normalization Plan

## Goal
Build a deterministic post-processing step for Stage 1 `attributes.jsonl` that reduces obvious lexical variance, collapses near-duplicates, flags noisy outputs, and produces reviewable normalized artifacts without hiding uncertainty.

Primary source for planning: `attributes_slot_value_summary.json` from the current ImageNette-like run.

## Scope
In scope now:
- slot-value normalization for the current 10-class ImageNette-like subset
- canonicalization of obvious formatting / synonym / plurality / casing variants
- limited class-aware normalization where the class label makes the intended canonical value nearly certain
- separation of:
  - auto-normalizable values
  - review-needed values
  - values that should remain as-is

Out of scope for first pass:
- open-ended ontology building across all future datasets
- embedding/LLM-based fuzzy merging in the main pipeline
- aggressive semantic rewriting of context/background text

## Current classes in this run
- `n01440764`: tench
- `n02102040`: English springer spaniel
- `n02979186`: cassette player
- `n03000684`: chainsaw
- `n03028079`: church
- `n03394916`: French horn
- `n03417042`: garbage truck
- `n03425413`: gas pump
- `n03445777`: golf ball
- `n03888257`: parachute

## Normalization principles
1. **Do deterministic cleanup first.** Lowercase, trim, normalize separators, collapse repeated whitespace, standardize commas / slashes / hyphens where safe.
2. **Preserve semantics over style.** Merge `frontal view` -> `frontal`, but do not rewrite rich background descriptions into a guessed ontology too early.
3. **Prefer slot-local rules.** The same phrase can mean different things in different slots.
4. **Use class-aware rules only when the class distribution is extremely concentrated.** Example: `instrument_type` for French horn should collapse strongly; `background_or_context` should not.
5. **Never silently erase uncertainty.** Keep raw value, normalized value, and a normalization status/reason.
6. **Review tails, not heads.** High-frequency obvious variants can be auto-fixed; low-frequency odd values should be surfaced for review.

## Recommended normalization statuses
Each normalized field should carry one of:
- `unchanged`
- `canonicalized` (format/synonym cleanup only)
- `class_inferred` (class-aware canonicalization)
- `mapped_to_unknown` (garbage/placeholder collapsed to `unknown`)
- `review_required`

## Slot-wise rule categories

### 1) Viewpoint-like slots
Applies to: all `viewpoint` slots.

Auto rules:
- collapse variants:
  - `front`, `front view`, `frontal view` -> `frontal`
  - `side`, `side angle` -> `side view`
  - `from above`, `overhead` -> `top-down`
  - `ground-level` -> `ground level`
- keep distinct when semantically meaningful:
  - `low angle`, `rear view`, `close-up`, `aerial`, `interior`
- mark review:
  - obviously misplaced technical/textual junk, e.g. tool viewpoint value `45 cc high performance, 2-stroke engine`
  - multi-angle or narrative phrases: `multiple angles`, `from audience perspective`, `sidewalk perspective`

### 2) Type/category/species slots
Applies to:
- `species_or_category`
- `device_or_appliance_type`
- `instrument_type`
- `sports_or_toy_type`
- `structure_or_building_type`
- `tool_type`
- `vehicle_type`

This is the highest-value normalization area.

Auto rules:
- lowercase/casefold and punctuation cleanup
- singularization of obvious plurals where slot expects a type
- class-aware canonical maps, e.g.:
  - tench class: `tench, tinca tinca`, `tench`, `Tench`, `Tench (Tinca tinca)`, `tinca tinca`, typo `tinch` -> `tench`
  - English springer class: `english springer`, `english springer spaniel`, `english springer, english springer spaniel` -> `english springer spaniel`
  - chainsaw class: `chain saw`, `chain saw, chainsaw` -> `chainsaw`
  - French horn class: `french horn`, `French Horn`, `French horn, horn`, `horn` -> `french horn`
  - golf ball class: `golf`, `golf balls`, `golf ball` -> `golf ball`
  - church class: `religious`, `worship space` -> review by default; possibly map to `church` only if user wants stronger class-conditioned collapse
  - gas pump class: collapse `gas pump`, `gasoline pump`, `petrol pump`, `island dispenser` variants if they appear in raw data
  - garbage truck / parachute classes: map close synonyms to canonical class name when clearly same object
- map placeholders to `unknown`: `unknown`, `not_applicable` when the slot should contain a type only if visible/identifiable

Review rules:
- clearly wrong alternate objects: `pike`, `catfish`, `trout` in tench class; `airplane` in parachute class; `generator` in chainsaw class
- decorative/accessory mentions instead of type: `golf accessory`, `golf ball holder`

### 3) Material / finish slots
Applies to `material`, `material_or_finish`, `material_or_surface`.

Auto rules:
- ordering / separator normalization:
  - `plastic and metal`, `metal and plastic`, `metal/plastic`, `plastic/metal`, `metal, plastic`, `plastic, metal` -> `metal+plastic`
  - `wood, stone`, `wood and stone` -> `wood+stone`
- synonym collapse where safe:
  - `metallic` -> `metal`
  - `wooden` -> `wood`
  - `weathered metal`, `rusty metal`, `metallic with rust` -> preserve detail under review tag or map to canonical primary `metal` plus modifier if modifier schema is added
- placeholders -> `unknown`

Recommendation:
- store **primary material** canonical value now
- defer richer condition/finish modifiers (`rusty`, `paint-chipped`, `weathered`) to a later modifier field instead of flattening too aggressively

### 4) Color / pattern slots
Applies to `color`, `color_or_pattern`.

Auto rules:
- canonical base colors: `gray`/`grey`, case normalization, whitespace cleanup
- conjunction normalization:
  - `red, white, blue` -> `red+white+blue`
  - `black and white` / `white and black` -> `black+white`
- simple pattern cleanup:
  - `white with logo`, `white with logos` -> keep as-is for now or split later into base color + marking
- normalize approximate variants if lossless enough:
  - `gold`, `golden` -> `gold`
  - `greenish`, `greenish-brown`, `yellowish` should **not** be flattened in v1 unless a color-family field is introduced

Review rules:
- non-color labels in color slot: `brass`, `metallic`, `silhouette`, `sepia tone`, `rainbow` depending on desired ontology
- decide whether `brass` belongs in material/finish rather than color for French horn

### 5) Shape / structure slots
Applies to `shape_or_structure`, `architectural_style_or_form` partly separate.

Auto rules:
- merge obvious morphological variants:
  - `sphere`, `spherical`, `smooth spherical` -> `spherical`
  - `curved tube`, `curved tubes`, `curved tubing` -> `curved tubing`
  - `large boxy`, `large, boxy` -> `large boxy`
- normalize singular/plural when describing shape rather than count

Review rules:
- values that are actually object identity, not shape:
  - vehicle `dump truck`, `dumpster-like`, `umbrella-like`
  - sports `golf ball`
  - structure form values mixing style and parts
- values containing dimensions/specs or junk text

### 6) State / pose / usage slots
Applies to:
- `pose_or_state`
- `operating_state_or_display_state`
- `playing_state_or_pose`
- `activity_or_usage_state`
- `usage_state`
- `state_or_action`

Auto rules:
- merge obvious synonyms:
  - `held`, `being held`, `held in hand`, `held by hand`, `held by person` -> `being held`
  - `off`, `powered off`, `inactive` -> `off` for device/tool operating-state slots
  - `active`, `powered on`, `on` -> `on`
  - `in operation`, `operational`, `in use` -> `in use`
  - `resting on ground`, `resting on surface`, `resting on floor`, `resting on grass` -> canonical parent `resting` unless location is worth preserving elsewhere
- placeholders:
  - `not_applicable`, `unknown` -> canonical placeholder

Review rules:
- human-centric phrases: `man playing`, `held by man`
- mixed state + context: `active, displaying prices`, `inactive, old`
- class-dependent actions that may deserve separate ontology (`deployed`, `ascending`, `emptying` for vehicles)

### 7) Salient part / focus slots
Applies to `salient_part_or_focus`, `salient_structural_part`, `salient_part_or_accessory`.

Auto rules:
- plural normalization where harmless:
  - `faces` -> `face`
  - `towers` -> `tower`
  - `crosses` -> `cross`
- merge exact near-duplicates:
  - `dumping mechanism` / `dump mechanism`
  - `entire fish` / `fish` only if you want coarse canonicalization; otherwise keep separate

Review rules:
- values dominated by people or scene composition rather than object parts:
  - `man holding fish`, `person`, `person attached`
- accessory/container mentions in vehicle slot may be useful but should likely be typed separately from part labels

### 8) Background / environment slots
Applies to `background_or_*`, `environment`, `surrounding_environment`.

Low-aggression normalization only.

Auto rules:
- whitespace/casing cleanup
- obvious duplicate merges:
  - `outdoor`, `outdoors`, `outdoor setting` -> `outdoor`
  - `indoor`, `indoor setting` -> `indoor`
  - `grass field`, `green grass` maybe keep separate; do not over-collapse
  - `clear blue sky` -> `clear sky` only if acceptable
- placeholder mapping: `unknown`, `indistinct`, `neutral` -> either `unknown` or keep separate depending on analysis needs

Review rules:
- values that are actually foreground surfaces instead of environment: `white`, `black`, `dark`, `couch`, `carpet`
- mixed/narrative phrases: `outdoor, possibly abandoned`, `outdoor, near water`

## Class-aware rules justified for current data
These are safe enough to implement now because the current run is extremely concentrated by class:

1. **Type/species slot canonicalization by class**
   - strongest win, lowest risk
2. **Expected-shape canonicalization for single-object classes**
   - golf ball: `sphere` / `spherical` / `round` family
   - French horn: `curved tube` / `curved tubing` family
   - chainsaw: `chain saw` type variants and blade/handle structure families
3. **Expected-material canonicalization where overwhelmingly dominant**
   - French horn -> `metal`
   - golf ball -> `rubber` may still need caution because some outputs mention composition/noise
4. **Expected-state normalization for specialized classes**
   - device/tool operating states (`off`/`inactive`, `on`/`active`)
   - parachute action family (`deployed`, `inflated`, `in flight`) should stay distinct but normalized lexically

Avoid strong class-aware rewrites for background/environment/focus slots.

## What can be normalized automatically now
High confidence:
- lowercase / trim / punctuation / spacing cleanup
- synonym collapse for viewpoint labels
- type/species canonicalization for all 10 current classes
- placeholder normalization: `unknown`, `not_applicable`, null-like variants
- separator normalization for multi-material / multi-color values
- singular/plural normalization for common part labels and type labels
- obvious typo repair with explicit whitelist (`tinch` -> `tench`)
- exact or near-exact duplicate lexical variants in state slots (`held` family, `off` family, `on` family)

## What should be review-gated
- values that look like wrong-object hallucinations
- values mixing multiple concepts in one string (`active, displaying prices`)
- values that belong to the wrong slot (`brass` in color; engine spec in viewpoint)
- long-tail background descriptions
- shape labels that are actually class names or metaphors (`umbrella-like`, `dumpster-like`)
- human/person mentions in part/focus/state slots

## Especially obvious noisy patterns from the summary
- cross-slot contamination exists: some `background/context` values are just surface colors (`white`, `black`, `dark`).
- some slots contain full narrative phrases instead of attributes, especially background and state fields.
- a few values are plainly misplaced/junk, e.g. tool `viewpoint = "45 cc high performance, 2-stroke engine"`.
- class-type slots still contain wrong-object hallucinations in the long tail (`generator` for chainsaw, `airplane` for parachute, fish species mismatches for tench).
- person/scene mentions leak into object-part slots (`man holding fish`, `person attached`).

## Suggested output artifacts
For an input `attributes.jsonl`, produce:
- `attributes_normalized.jsonl`
  - original row + per-slot normalized values
- `normalization_audit.jsonl`
  - one record per changed/reviewed field with raw value, normalized value, rule id, status
- `normalization_summary.json`
  - counts by slot/status/rule/class
- `normalization_review_queue.jsonl`
  - only `review_required` items, sorted/grouped later by slot and frequency
- optional: `normalization_rules_snapshot.json`
  - exact compiled rule tables used for reproducibility

## Recommended implementation order

### Stage A: infrastructure
- add a standalone normalizer module/script, e.g. `scripts/data/normalize_stage1_attributes.py`
- define a small rule engine:
  - global cleanup rules
  - slot-specific canonical maps
  - class+slot-specific canonical maps
  - review triggers
- keep rules in data files when possible, e.g. `configs/stage1/normalization/*.json`

### Stage B: safest high-value rules
- placeholder normalization
- viewpoint normalization
- type/species canonicalization for the 10 current classes
- material separator normalization
- state-family normalization (`off`/`inactive`, `held` family, etc.)

### Stage C: review pipeline
- emit audit + review queue
- aggregate unseen tail values by slot/class/frequency
- manually inspect top review buckets and promote safe rules into configs

### Stage D: moderate semantic cleanup
- shape family canonicalization
- part/accessory canonicalization
- limited background/environment consolidation

### Stage E: optional ontology refinement
- split compound attributes into structured subfields where useful:
  - color + marking
  - material + finish/condition
  - state + action + holder/person context

## Practical rule format suggestion
Use explicit rule ids and keep them explainable.

Example categories:
- `global.casefold`
- `slot.viewpoint.front_to_frontal`
- `slot.material.separator_unify`
- `class.n01440764.species.tench_family`
- `class.n03000684.tool_type.chain_saw_to_chainsaw`
- `review.wrong_object_candidate`
- `review.slot_contamination`

## v2 refinement notes
A conservative v2 refinement is justified for the current review queue because a large share of flags are not semantically useful disagreements, just contamination:
- person mentions leaking into `salient_part_or_focus` / `salient_part_or_accessory`
- pure color/lighting words leaking into background/context slots
- short two-clause mixed-state strings where the head state is already a good coarse label

For these buckets, prefer mapping low-value contamination to `unknown` or collapsing to the primary clause, rather than inventing a richer rewrite.

## Minimal success criterion
The first normalization pass is good enough if it:
- sharply reduces unique value counts for `*_type`, `species_or_category`, `viewpoint`, and major state slots
- does not destroy raw information
- produces a compact review queue dominated by genuinely ambiguous/noisy cases rather than formatting noise
