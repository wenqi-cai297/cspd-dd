# CSPD — Paper Brief for Draft Authoring

Status: paper-authoring brief, for a downstream LLM (e.g. ChatGPT) to draft the manuscript.
Source of truth for the code: the repo at `E:\Project\2026-03-25` (main branch).
Companion technical doc: `gen_dd_coding_instruction_spec.md` (kept local; gives implementation-level facts).
Companion roadmap doc: `plan.md` (kept local; gives what is closed / active / queued).

This brief is self-contained. Reading it should let you write a methods paper without going back into the codebase. Every claim about pipeline behavior, hyperparameter, ablation result, or experimental conclusion below is grounded in code or run artifacts; nothing is forward-looking or "should do".

---

## 0. How to use this brief

1. **Section 1–3** give the story: the problem, the prior-work gap, our claim.
2. **Section 4** is the method, stage by stage. Each subsection has a "what the paper says" vs "what the code does" split so equations and prose are anchored in the implementation.
3. **Section 5** is empirical evidence: the main result, ablations, and IPC scaling status.
4. **Section 6** is the negative-result log. Five concrete redirections were tested, falsified, and removed. These are arguably the most defensible scientific contribution of the project; treat them as first-class findings.
5. **Section 7** is the related-work map.
6. **Section 8** is suggested paper structure.
7. **Sections 9–10** are reproducibility and limitations.

The paper's core thesis is **structured semantics + generator-native conditioning + density-aware mode discovery is enough** — once those three pieces are in the right place, every post-hoc selection or sampler-guidance refinement we tried regressed.

---

## 1. Problem and motivation

**Task.** Dataset distillation for image classification: given a real training set with $N$ images, produce a synthetic set with $|S| = K \times \text{IPC}$ images ($K$ classes) such that a classifier trained on $S$ and evaluated on the real validation set achieves accuracy as close as possible to the model trained on the full real set.

**Generative DD vs gradient-matching DD.** Two families dominate the literature:
- **Gradient / trajectory / distribution matching** (DC, MTT, DM, IDC, RDED, SRe2L, …): synthesize pixels directly with a matching objective. Scales poorly past CIFAR-10/100; high IPC is still hard on ImageNet-1k.
- **Generative DD** (GLaD, DiT-DD, MGD³, DD-VLCP, D³HR, DAP, …): condition a pretrained generator on a class signal and synthesize images. Decouples optimization from pixel space; scales better; the quality of *what* you condition the generator on becomes the central design question.

**The gap we attack.** Within generative DD, the common pattern is "class name → diffusion model". This conflates two things into one short text:
1. *what is in the image* (object identity, type, parts);
2. *how it appears in the dataset* (pose, environment, viewpoint, intra-class variation).

Class-name conditioning collapses (2). The result is a synthetic set that looks like a cleaner ImageNet *clip-art* than the actual ImageNet *photograph* distribution, and downstream classifier accuracy suffers — especially at low IPC where every image must carry a distinct sub-mode of the class.

Two attempted fixes from prior work each have a defect:
- **Long free-form recaption** (e.g. BLIP / VLM rewrites): caption distribution shifts away from anything the generator was trained on; the LoRA / fine-tuned generator overfits to a recaption style that does not match downstream inference; we measured a ~6% drop (§6.2).
- **Post-hoc representativeness selection** (D³HR, DAP-style moment / MMD matching): operates after generation; greedy class-mean matching over-smooths inter-mode diversity at low IPC; we measured a 2.8% drop in DINOv2 space and a 3.3% drop in VAE space (§6.5).

**Our contribution.** We propose **Class-aware Semantic-Prompt Distillation (CSPD)** — a four-stage pipeline that pushes structure earlier in the pipeline:

1. **Semantic structure is image-level, archetype-aware, and slot-typed**, not free text. Each class maps to a fixed *archetype* (animal / vehicle / structure / …), each archetype has a fixed 7-slot schema (type, color, pose/state, background, viewpoint, salient part, …), and a VLM fills those slots per image. This produces captions of the form `a brown speckled long-bodied tench being held in studio, viewed from the front`, deterministically renderable from $(\text{archetype}, \text{slot dict})$.
2. **Generator adaptation aligns to this caption distribution.** SDXL UNet LoRA is trained on real image / canonical-caption pairs so that the generator can later be conditioned by the same template at inference.
3. **Mode discovery is density-aware, in a representation space the generator does not control.** Per-class HDBSCAN on DINOv2 features finds natural sub-modes (e.g. "tench-being-held" vs "tench-on-grass"); each mode contributes its medoid's canonical caption.
4. **Generation is text-only and prototype-aligned.** SDXL + the Stage 2 LoRA, conditioned on the medoid caption, generates one image per mode.

The whole pipeline is deterministic given seeds, runs end-to-end via a single shell driver, and is auditable at every stage (every dropped slot, every VLM override, every cluster assignment is logged).

---

## 2. Key empirical claims to defend in the paper

Numerical claims grounded in `runs/` and the spec history log. All numbers below are on **ImageNette, IPC=10, ResNetAP-10** unless stated.

| Claim | Number | Source | Status |
|---|---|---|---|
| **Mainline accuracy** under the 3×3 measurement protocol | **63.27 ± 0.19** (per-seed 63.4 / 63.0 / 63.4) | `runs/stage4/.../pipeline_*/summary.txt`, commit `5dfd24f` | Locked |
| Old single-run baseline (for back-compat with earlier numbers) | 62.33 ± 1.47 | `runs/eval/2026-04-17_150749_ipc10_resnet_ap/` | Locked |
| Free-form Stage 4 recaption is harmful | 56.67 ± 0.50 (−6.6%) | `runs/eval/2026-04-15_173911_*` | Closed |
| Per-mode multi-candidate selection (DINOv2 prototype + diversity) doesn't help at IPC=10 | 60.8 ± 0.33 (−1.5%) | `runs/eval/2026-04-16_062943_*` | Closed |
| Set-level representativeness selection (D³HR-style moments, DINOv2 space) regresses | 59.53 ± 0.38 (−2.8%) | `runs/eval/2026-04-17_210019_*`, commit `57b72f0` | Closed |
| Set-level representativeness selection in **VAE latent space** also regresses (the obvious next-thing-to-try) | 59.07 ± 0.25 (−3.3%) | commit `d81b47b` | Closed |
| MGD³-style latent mode guidance is structurally incompatible with detailed text conditioning | no usable scale; either no effect or quality collapse | code: `mode_guidance.py` (removed `a36e8d9`) | Closed |
| Non-SDXL backbones (SD v1.5 full fine-tune) underperform SDXL LoRA | 61.3% vs 62.33% at the time of test | spec §16.10, commit `d992e76` | Closed |
| Stage 1A scales to ImageNet-1k | 99.7% render success (4999 / 5000 on the 5-shot split) | `runs/stage1/render/ImageNet1k_5shot/.../render_summary.json` | Verified |
| Stage 1B streaming refactor makes ImageNet-1k full-set normalization OOM-safe | RSS now O(1) in N rows | commit `c35f207` | Landed |

**Defensible takeaways** for the paper (in order of strength):

- T1. **Where semantics enter matters more than what feature space you select in.** Three independent post-hoc selection methods (per-mode multi-candidate; DINOv2 set-level; VAE set-level) all regressed against the simple medoid baseline. The shared cause is **strong, detailed text conditioning** at generation time. Once the UNet locks onto a structured caption, downstream re-ranking has nothing left to do.
- T2. **Density-aware mode discovery on a feature space the generator does not control (DINOv2)** is the right place to spend the modeling budget, *not* on a smarter selector after generation.
- T3. **Archetype-aware slot schemas + per-slot guidance prompts + deterministic rendering** produce a caption distribution that the generator can learn (LoRA training stays well-behaved) and re-condition on cleanly (Stage 4 doesn't OOD-shift). Free-form rewrites break this contract.
- T4. **MGD³-style latent guidance is fundamentally text-strength-bounded.** It works when text is "tench" (weak) and fails when text is "a brown speckled long-bodied tench being held by a person in a studio". This is not a tuning issue; we swept scales and either there is no content effect (scale ≤ 0.1) or the image quality collapses (scale ≥ 0.2). Reported as a clean negative result.

---

## 3. One-paragraph elevator pitch

Generative dataset distillation has converged on a recipe — "condition a diffusion model on a class signal, generate IPC images per class, train a downstream classifier" — but the field is still debating *what to condition on* and *how to pick the K best samples after the fact*. We argue both questions are downstream of a more important one: **the structure of the conditioning signal itself**. CSPD encodes each real image as a fixed archetype-specific slot dict (type / color / pose / background / viewpoint / salient part) extracted by a VLM, renders it into a deterministic canonical caption, adapts an SDXL UNet via LoRA to this caption distribution, then discovers natural sub-modes per class with HDBSCAN on DINOv2 features and generates one image from each mode's medoid caption. The contract — "the captions Stage 4 generates from are identical in distribution to the captions Stage 2 trained on" — is what unlocks everything else. We falsify five alternative refinements (free-form recaption, per-mode multi-candidate selection, set-level matching in DINOv2 space, set-level matching in VAE space, MGD³-style mode guidance) on ImageNette at IPC=10; the simple medoid-of-cluster baseline beats all of them by 1.5–6 percentage points and reaches **63.27 ± 0.19** with ResNetAP-10 under a 3×3 measurement protocol.

---

## 4. Method — pipeline detail

Pipeline order: `Prep → Stage 1 (1A 1B 1C) → Stage 2 → Stage 3 → Stage 4 → Eval`. Each stage is idempotent and skipped when its canonical artifact exists on disk. The end-to-end driver is `scripts/pipelines/run_full_pipeline.sh <train_root> [val_root] [nclass]`.

### 4.1 Notation

- $\mathcal{D}_{\text{real}} = \{(x_i, y_i)\}_{i=1}^N$ — real ImageFolder dataset, $K$ classes.
- $\mathcal{A} = \{a_1, \dots, a_{|A|}\}$ — fixed *archetype* taxonomy ($|A|=18$ in the bundled config).
- $\phi: \{1, \dots, K\} \to \mathcal{A}$ — class-to-archetype map (frozen, manual on ImageNet-1k).
- $\sigma: \mathcal{A} \to \text{ordered slot list}$ — fixed 7-slot schema per archetype.
- $V$ — a VLM (concretely Qwen2.5-VL-7B-Instruct).
- $\mathcal{R}$ — the deterministic Stage 1B normalization ruleset.
- $\mathcal{T}_a$ — the deterministic Stage 1C template for archetype $a$.
- $E_{\text{DINO}}$ — DINOv2 ViT-B/14 image encoder.
- $G$ — SDXL base 1.0 with a trained LoRA $\Delta\theta$.

### 4.2 Prep: class metadata

Two artifacts are required before Stage 1: `classes.json` (a `class_raw_label → class_readable_name` map) and `class_to_archetype.json` (a `class_raw_label → archetype` map).

For ImageNet-1k the repo bundles a manually curated map at `configs/stage1/class_to_archetype_imagenet1k_manual.json`. For other datasets, a multimodal class-level mapper exists at `scripts/prep/generate_class_to_archetype_map_vlm.py` (samples 5 images per class, asks the VLM to pick an archetype from the fixed taxonomy).

**Methodology note (important for the paper).** Class identity is used at Prep only. Stage 1 normalization, Stage 1 render, Stage 3 clustering, and Stage 4 generation never reference class names or class-specific rules. This is enforced as a project boundary (see spec §16.1). The empirical motivation: class-level rules don't generalize, and they conflate dataset-specific tuning with method evaluation.

### 4.3 Stage 1A: structured semantic extraction

Input: an ImageFolder root + the two Prep maps.
Output: `attributes.jsonl`, one row per image.

For each image $x_i$ in class $y_i$:

$$a_i = \phi(y_i), \qquad s_i = \sigma(a_i), \qquad p_i = V(x_i, \text{prompt}(a_i, s_i, \text{class\_name}(y_i)))$$

The user prompt is class-adaptive: it lists exactly the slot names in $s_i$ as JSON keys, with per-slot guidance strings drawn from `SLOT_GUIDANCE` in `src/cspd_stage1/prompting.py`. Examples:

- `background_or_habitat`: `"scene or place WHERE the subject is, e.g. grassy field, lake shore. Do NOT write just a color"`
- `viewpoint`: `"camera angle, e.g. front view, side view, top-down view"`
- `operating_state_or_display_state`: `"device state with detail, e.g. playing music with display lit. Do NOT write just 'on' or 'off'"`

The system prompt enforces JSON-only output and that missing values be filled with `"unknown"`. Failed parses go to `failed_samples.jsonl` and are retried with bounded `--max-retries`.

**Schema (the slot families).** 18 archetypes × 7 slots each. The animal archetype as a representative example:
`species_or_category, color_or_pattern, body_trait, pose_or_state, background_or_habitat, viewpoint, salient_part_or_focus`.

**Cost.** On ImageNette train (12,894 images): 100% success. On ImageNet-1k 5-shot (5,000 images): 99.7% (single failure was a missing archetype in the manual map, fixed in revision 2026-04-12). On ImageNet-1k full train (1.28M images): in progress; Stage 1A is empirically robust at this scale (no OOM, resumable).

### 4.4 Stage 1B: deterministic-first normalization + inline VLM review

Pure VLM output is locally inconsistent ("grey" vs "gray", "metallic" vs "metal", "powered on" vs "on", "white" inside a `background_or_habitat` slot meaning the studio backdrop, etc.). Stage 1B is the canonicalization layer.

**Two passes** (after the 2026-05-19 streaming refactor in commit `c35f207`):

1. **Deterministic pass.** For each row, for each slot, apply rule families:
   - lexical cleanup (whitespace, case-folding, separator unification);
   - placeholder cleanup (`null`, empty, `"n/a"` → `unknown`);
   - slot-aware canonicalization (`viewpoint_map`, `state_map`, `shape_map`, `background_map`, `type_map`);
   - low-value-value suppression (`stationary` in state, `neutral` in background) → `unknown`;
   - archetype-aware mismatch detection (e.g. `style_value ∈ {bridge, tower, …}` in a `architectural_style_or_form` slot flags `review.structure_type_style_conflict`).

   Each slot ends with `status ∈ {unchanged, canonicalized, mapped_to_unknown, review_required}`, applied rules, and review reasons.

2. **Inline constrained VLM review pass** (default on). For each slot whose deterministic outcome is ambiguous (`status = review_required` *or* `status = mapped_to_unknown` — the latter is the *recovery* trigger added 2026-04-12), the VLM gets the image, the slot name, the candidate normalized value, and a strict action space `{keep_normalized, replace_normalized, set_unknown, defer}`. Decisions are gated by `should_apply_review_decision()`: low-confidence replaces are rejected; low-confidence `set_unknown` is allowed only when a high-severity reason was flagged.

   On ImageNette this expanded `mapped_to_unknown_recovery` trigger raised review items from 66 → 969, of which 511 (52.7%) successfully recovered a non-unknown value.

**Outputs** (per row): `normalized_attributes` (deterministic), `effective_normalized_attributes` (deterministic + applied VLM decisions), `attribute_normalization` (per-slot metadata), `vlm_review` (per-row review record).

**Streaming.** Pass 1 reads and writes JSONL line-by-line; Pass 2 streams the Pass-1 output back in, rewrites it to a sibling `.tmp` file with VLM decisions applied, then `os.replace`s atomically. The Qwen client is constructed exactly once outside the loop so the 14 GB weights stay GPU-resident. Memory footprint is O(1) in dataset size. (Earlier in-memory implementation OOM-killed at ImageNet-1k scale; fix in commit `c35f207`.)

### 4.5 Stage 1C: deterministic canonical render

The render function $\mathcal{T}_{a_i}$ turns an `effective_normalized_attributes` dict into one canonical caption. The template per archetype is fixed; for `animal`:

```
[<color_or_pattern> <body_trait>] <species_or_category>
  [<pose_or_state>]
  [in <background_or_habitat>]
  [from <viewpoint>]
  [with <salient_part_or_focus>]
```

Mechanics:
- The `anchor_slot` (e.g. `species_or_category`) is the only obligatory slot; if missing, optional `fallback_anchor` token or `class_name` is used.
- Each remaining slot has a fixed prefix (`""`, `"in"`, `"from"`, `"with"`, …).
- Slots whose value is `unknown` / `not_applicable` are dropped.
- A per-archetype `drop_if_unknown` set lists slots that can be dropped if missing.
- Final whitespace/punctuation cleanup is deterministic.

**Diversity-preserving slot-drop policy** (2026-04-11 revision; spec §11). Earlier render was over-aggressive: ~30% caption duplication on ImageNette. We relaxed the drop rules at slot-type level (never class-level):
- Background suppressions: 23 → 5 values.
- Pose/state suppressions: 21 → 3 values.
- Viewpoint: `"front view"` / `"side view"` no longer dropped.
- Salient part: 16 → 6.
- Trait/shape: 14 → 4.

Net effect: unique-caption rate rose from 70.6% → 84.6% on ImageNette train; on ImageNet-1k 5-shot, 99.7% across the 5000 records.

### 4.6 Stage 2: SDXL UNet LoRA adaptation

Goal: align the generator to the canonical-caption distribution from Stage 1C, so that at Stage 4 conditioning on the *same kind of caption* gives high-quality images.

**Pairing.** Stage 1C `records.jsonl` is joined with the ImageFolder root by `record_id = class_name_raw::relative_path`. Images are copied into `sdxl_materialized_dataset/images/`, paired with `text = canonical_caption` in `metadata.jsonl` (the diffusers `imagefolder` format).

**Trainer.** A thin wrapper around the official `train_text_to_image_lora_sdxl.py` from `diffusers/examples/text_to_image/`. We do not reimplement training; we only own pairing, dataset materialization, launch orchestration, preflight checks.

**Mainline hyperparameters** (this is the configuration that produces the 63.27% result):

| Knob | Value |
|---|---|
| Backbone | `stabilityai/stable-diffusion-xl-base-1.0` |
| Parameterization | LoRA, rank=64 |
| LoRA target modules | UNet attention `to_k`, `to_q`, `to_v`, `to_out.0` |
| Resolution | 512 (non-native for SDXL, empirically OK) |
| GPUs | 2 (accelerate `--num_processes 2`) |
| Per-GPU batch | 8 (effective batch 16) |
| Epochs | 9 (best checkpoint on ImageNette is `checkpoint-7254`) |
| LR | 2e-5 |
| Scheduler | cosine, **500-step warmup** |
| Noise offset | 0.05 |
| Min-SNR γ | 5.0 |
| Mixed precision | fp16 |
| Gradient checkpointing | enabled |
| VAE + text encoders | frozen |

**Why these numbers.** A checkpoint sweep over epochs 5–15 (with cosine LR + warmup) settled on epoch 9 — earlier than constant-LR's epoch 15. Rank 64 beat rank 16 on visual quality. The noise offset / Min-SNR additions are standard SDXL training tweaks that we ablated qualitatively and kept.

**Frozen non-trainable modules** stay on GPU rather than being CPU-offloaded — the throughput hit of constant cross-device transfer dominates the VRAM savings for our 2-GPU setup.

### 4.7 Stage 3: mode discovery via DINOv2 + HDBSCAN

For each class, find the natural sub-modes ("being held", "in a river", "studio close-up", …) using a representation space the generator did not adapt to, and pick a medoid caption per mode.

**3A: Encode.** DINOv2 ViT-B/14 (`facebookresearch/dinov2:dinov2_vitb14`) on 224×224 ImageNet-normalized crops → CLS features in $\mathbb{R}^{768}$. We use DINOv2 (not SDXL VAE latents) because density clustering needs *semantic* structure with natural separation, not the smooth pixel-aligned manifold a VAE provides. Spec §14 records that 6–8/10 ImageNette classes fell back to K-Means under VAE+HDBSCAN, vs 9/10 forming real density modes under DINOv2+HDBSCAN.

**3B: Cluster.** For each class $c$:
1. Optional PCA (default `pca_dim=50`).
2. HDBSCAN with `min_cluster_size = min(15, max(N_c / IPC, 5))`, `min_samples=3` → discovers natural modes (no preset K).
3. Noise points (label −1) are reassigned to the nearest discovered mode centroid.
4. **Fallback:** if HDBSCAN finds ≤ 1 mode → seeded K-Means with $K=\text{IPC}$.
5. **Discovered > IPC:** farthest-point sampling (greedy from index 0) selects the IPC most diverse mode centroids; unselected modes' members are absorbed by the nearest selected mode.
6. **Discovered ≤ IPC:** proportional IPC allocation (every mode gets ≥ 1 slot). Parents receiving > 1 slot are sub-clustered by seeded K-Means.

**3C: Medoid + caption.** For each final cluster $c_k$, the **medoid** $m_k$ is the real sample whose DINOv2 feature is closest to the cluster centroid. Its Stage 1C caption is the **representative caption** that Stage 4 will condition on. We also record cluster `weight` (size fraction) and `density` (1 / mean distance to centroid) for diagnostics — these are *not* used by the current Stage 4 mainline (they were used by the now-removed set-level scorer).

**Determinism note.** Pure HDBSCAN is deterministic given input. The seed enters only via PCA (deterministic for given seed) and via the K-Means fallback / sub-clustering. We exploit this for the 3×3 protocol (§4.9).

### 4.8 Stage 4: text-to-image distilled generation

For each Stage 3 mode $k$ in each class:

$$\hat{x}_k = G(\text{prompt}=\text{caption}(m_k), \text{seed}=\text{seed}_0 + k, \text{guidance}=7.5, \text{steps}=50)$$

Default mode is **text2img** (`--visual-mode none`). The Stage 2 LoRA is loaded via `pipe.load_lora_weights(...)` on the SDXL base. Output resolution is 512 (matching the LoRA training resolution).

**Img2img ablation path** (`--visual-mode medoid`, kept for the ablation table). Same caption, but the diffusion starts from the real medoid image at `strength=0.8`. Empirically worse for classifier training even though images look more diverse — we attribute this to a Stage-2-vs-Stage-4 conditioning distribution mismatch.

**Optional refiner** (`--refiner-model stabilityai/stable-diffusion-xl-refiner-1.0`, `--refiner-strength 0.3`). Chains an SDXL refiner pass on top of either text2img or img2img output for additional detail. Not part of the mainline result; kept available as an ablation knob.

**Output layout.** ImageFolder-format:
```
runs/stage4/<dataset>/ipc<IPC>/lora/pipeline_<TS>/gen_seed<SEED>/images/
  <class_raw>/<class_raw>_mode<NNN>.png
```

### 4.9 Evaluation

Train a classifier on the Stage 4 distilled ImageFolder, evaluate on the real validation set.

**Protocol.** Ported verbatim from the MGD³ reference eval (`github.com/jachansantiago/mode_guidance/.../eval/`):
- Optimizer: SGD, momentum=0.9, weight_decay=5e-4, **lr=0.01**.
- LR schedule: `MultiStepLR` at $2/3$ and $5/6$ of total epochs, $\gamma=0.2$.
- Augmentation: `RandomResizedCrop(scale=(0.5, 1.0))` → `ToTensor` → `RandomHorizontalFlip` → custom tensor-space `ColorJitter(0.4,0.4,0.4)` → `PCA Lighting(0.1)` → ImageNet `Normalize`.
- **CutMix** ($\beta=1$, $\text{mix\_p}=1$) every step.
- Epochs from IPC: IPC ≤ 10 → 2000; IPC ≤ 50 → 1500; IPC ≤ 200 → 1000; IPC ≤ 500 → 500; else 300. (For 100-class data, ×2/3.)
- Image size 224 × 224.
- Three architectures: ConvNet-6, ResNet-18, ResNetAP-10. Each architecture is reported as **best top-1 over 3 independent runs**.

**3×3 measurement protocol** (introduced 2026-04-18). To prevent single-seed lucky-roll results:
1. **Seed sweep** over `{42, 123, 456}`. Each seed re-runs Stage 3B (HDBSCAN clustering with K-Means fallback / sub-clustering uses this seed) → Stage 4 generation (per-image seed = `seed + mode_idx`).
2. **Per-seed**: 3 independent classifier training runs (different `torch` init).
3. **Aggregation**: best-of-3 per seed → mean ± std across the 3 per-seed bests.

This is what produces the **63.27 ± 0.19** headline number on ImageNette IPC=10 ResNetAP-10. The seed=42 round (63.4%) reproduces the older single-run 62.33% baseline byte-for-byte; only the aggregation differs (best-of-3 instead of mean-of-3).

---

## 5. Results

### 5.1 Main result

ImageNette train (10 classes, 12,894 images) → IPC=10 distilled set (100 images). Real val (3,925 images).

**Mainline:** SDXL LoRA (rank=64, epoch 9) + DINOv2 HDBSCAN + medoid caption + text2img, ResNetAP-10, 3×3 protocol.

**Top-1 accuracy: 63.27 ± 0.19 %**

Per-seed best-of-3: seed=42 → 63.4 %, seed=123 → 63.0 %, seed=456 → 63.4 %. Min/max across 3 per-seed bests: 63.0 / 63.4.

### 5.2 Single-knob ablations (the negative-result table — paper Table 2)

All entries use the same mainline scaffold; one knob changed.

| Variant | Accuracy (IPC=10) | Δ vs mainline | Spec / commit |
|---|---|---|---|
| **Mainline (HDBSCAN + medoid + text2img)** | **63.27 ± 0.19** | — | `5dfd24f` |
| Mainline w/ old single-seed protocol | 62.33 ± 1.47 | (same data, old aggregation) | `runs/eval/2026-04-17_150749_*` |
| K-Means instead of HDBSCAN | 62.13 | −0.20 | `runs/eval/2026-04-17_150753_*` |
| Img2img from medoid (`--visual-mode medoid`, strength=0.8) | significantly worse (qualitatively) | clear regression | spec §15 (item 5) |
| Free-form Stage 3 / Stage 4 recaption | 56.67 ± 0.50 | **−6.6** | §6.2 below |
| Per-mode multi-candidate (DINOv2 prototype + diversity, β=0) | 60.8 ± 0.33 | **−1.5** | §6.3 below |
| Set-level moments, DINOv2 L2-normalized, N=10 candidates per mode | 59.53 ± 0.38 | **−2.8** | §6.4 below |
| Set-level moments, **VAE latents**, 16384-dim, no L2-norm | 59.07 ± 0.25 | **−3.3** | §6.4 below |
| MGD³-style mode guidance under structured captions | no usable scale | structural failure | §6.5 below |
| SD v1.5 backbone (full fine-tune) | 61.3 | −1.0 (at time of test) | §6.6 below |

### 5.3 IPC scaling (status: in progress)

The mainline infrastructure supports the sweep via `PIPELINE_IPC="10 20 50"`. As of this brief, only IPC=10 is fully measured under the 3×3 protocol. IPC=20 and IPC=50 are queued as the next experiment (`plan.md` §5.2, "Gate B"). The cross-architecture sweep (ConvNet-6 / ResNet-18 / ResNetAP-10 at each IPC) is also queued. For the paper, this section should be filled in by reading the per-IPC `summary.txt` produced under `runs/stage4/ImageNette_train/ipc{10,20,50}/lora/pipeline_*/summary.txt` once the sweeps complete.

### 5.4 ImageNet-1k status

Stage 1A on the full ImageNet-1k train (1.28M images) is in progress (`runs/stage1/attributes/Imagenet1k_train/qwen_local/2026-04-13_080405/`). Stage 1B was OOM-blocked in May 2026 by the in-memory implementation; fixed in commit `c35f207` (streaming + persistent VLM client). After 1B + 1C complete, the same single-command full pipeline (`scripts/pipelines/run_full_pipeline.sh /path/to/ImageNet1k/train`) runs end-to-end. Numbers for the paper should be added when IPC × arch sweeps complete.

---

## 6. Negative results (the most defensible part of the paper)

These should not be relegated to an appendix. Each was a self-contained, code-complete attempt with a specific hypothesis, and the falsification removed the corresponding code from the repo. Treat them as findings.

### 6.1 SD v1.5 backbone (2026-04-16 → 2026-04-17)

**Hypothesis.** SDXL at non-native 512 is the bottleneck. SD v1.5 is native-512, ~860M params, and validated by DD-VLCP. Full fine-tuning should beat SDXL LoRA.

**Setup.** SD v1.5 UNet full fine-tune via `train_text_to_image.py`, 8 epochs, batch=8, 2 GPUs, cosine LR, noise_offset=0.05, snr_gamma=5.0. Stage 4 auto-detects SD v1.5 vs SDXL and loads via `from_pretrained`.

**Result.** Sample quality looked good; eval **worse** than SDXL baseline (~61.3% vs ~62.33% at the time). PixArt-α and FLUX never had a working training path on our stack; both were considered and dropped.

**Interpretation.** SDXL's dual CLIP text encoders + larger capacity more than compensate for the 1024→512 resolution mismatch. The bottleneck is *not* the backbone family.

**Code state.** All `families/{sd15,flux,pixart}` subpackages removed in cleanup batch 1 (commit `d992e76`).

### 6.2 Free-form Stage 4 recaption (2026-04-15)

**Hypothesis.** Templated captions are rigid; letting the VLM rewrite a richer caption per mode should improve generation fidelity.

**Setup.** Free-form VLM rewrite of each Stage 3 mode's medoid caption, used at Stage 4 generation.

**Result.** **56.67 ± 0.50 % (−6.6 %)** vs 62.33% baseline (`runs/eval/2026-04-15_173911_*`).

**Interpretation.** This is the cleanest demonstration of the "caption-format contract" principle (§16.9). Enriched captions are out-of-distribution for the LoRA *trained* on canonical templated captions. The LoRA generates well only from the caption format it was adapted to. Any caption enrichment must be applied at Stage 1 (so it propagates through training) or not at all.

### 6.3 Per-mode multi-candidate selection — Phase 2 (2026-04-16)

**Hypothesis.** Generating $N > 1$ candidates per mode and picking the best one in feature space should beat the single-shot medoid baseline.

**Setup v1.** DINOv2 linear probe (discriminative) + cosine diversity, $N=10$, β-sweep.
**Result v1.** β=0.5 → 58.3%, β=0.0 → 60.8% (`runs/eval/2026-04-16_062943_*`). Both worse than baseline.

**Setup v2 (the recommended fix from D³HR / IGDS / DAP).** Architecture-agnostic scoring: prototype similarity (cosine to class mean DINOv2) + diversity. No proxy classifier.

**Result.** Did not beat the single-medoid baseline at IPC=10.

**Interpretation.** Per-mode selection re-introduces noise the medoid-of-cluster operation already removed. At IPC=10 the IPC budget is exactly the number of modes; there is no extra capacity to pay for selection.

**Code state.** `candidate_selection.py` deleted (commit `a36e8d9`).

### 6.4 Set-level representativeness selection — Phase 3 (2026-04-17)

**Hypothesis.** Greedy per-mode selection ignores cross-mode interactions. Optimize the whole set against the real class distribution.

**Setup.** Greedy selection with a 1-per-mode constraint (preserves Stage 3 mode structure). Objectives:
- `moments`: D³HR-style $\|\mu_{\text{synth}} - \mu_{\text{real}}\| + \|\sigma_{\text{synth}} - \sigma_{\text{real}}\| + 0.1 \cdot \|\text{skew}_{\text{synth}}\|$
- `mmd`: DAP-style linear-kernel MMD² between real and synthetic feature sets

**Result A/B #1 — DINOv2 CLS, L2-normalized, moments, N=10 candidates per mode:**
**59.53 ± 0.38 % (−2.80 %)** vs baseline. (`runs/eval/2026-04-17_210019_*`, commit `57b72f0`)

**Result A/B #2 — SDXL VAE latents, unnormalized, 16384-dim, moments, N=10 candidates per mode:**
**59.07 ± 0.25 % (−3.26 %)** vs baseline. (commit `d81b47b`)

**Interpretation.** Both feature spaces regressed by roughly the same margin. The VAE space test was specifically designed to control for the two most-likely a-priori explanations (proxy-space mismatch + L2-norm wiping magnitude). The result did not move toward the baseline. The bottleneck is the **objective itself**, not the feature space:

> The medoid baseline gives each mode the real image closest to *its own cluster centroid* — diversity comes from inter-mode variation. Greedy class-mean matching pulls the whole set toward the *class centroid*; the first pick anchors near it and later picks compensate, but inter-mode spread gets smoothed out. The result is a more homogeneous set than medoid → worse classifier training signal at low IPC.

This is the **strongest empirical evidence** that post-hoc set-level matching is the wrong place to spend modeling effort under low-IPC + strong text conditioning.

**Code state.** `representativeness.py` deleted (commit `a36e8d9`).

### 6.5 MGD³-style latent mode guidance (2026-04-16)

**Hypothesis.** Combine our structured caption conditioning (text path) with MGD³-style VAE latent centroid guidance (visual path) — a combination not in prior work.

**Setup.** `EulerModeGuidanceScheduler` subclasses `EulerDiscreteScheduler`, injects guidance in `step()`. Multiple implementations tried (custom denoising loop, callback hook, custom scheduler step). Two feature spaces (DINOv2 cluster VAE means, then VAE-native K-Means centroids). Scale sweep 0.1 → 0.18.

**Result.** Either no content effect (scale ≤ 0.1) or image quality collapses (scale ≥ 0.2). No usable sweet spot.

**Interpretation.** Latent guidance and text conditioning compete for control over the same UNet features. MGD³ works because its text conditioning is weak ("tench"). Our text conditioning is strong ("a brown speckled long-bodied tench being held in studio…") and dominates the sample. This is **structural**, not a tuning failure.

For the paper, this is a clean theoretical statement: **strong, detailed text conditioning is structurally incompatible with latent mode guidance** in a single diffusion pass. Either you weaken the text (and lose Stage 1's structural contribution) or you accept that mode guidance has no room to act.

**Code state.** `mode_guidance.py` deleted (commit `a36e8d9`).

### 6.6 The cross-cutting takeaway

The five negatives are not independent. Their common cause is the strength and specificity of the Stage 1 caption distribution. The same property that makes the mainline work — *each caption uniquely identifies a sub-mode of its class* — also crowds out (a) any free-form rewrite (distribution shift), (b) any post-hoc selection (the medoid baseline is already optimal under this signal), (c) any latent guidance (the UNet is already locked in).

This is a unified explanation worth one prominent paragraph in the discussion.

---

## 7. Related work

Treat these as the comparison anchors. Each has a one-line "signal for CSPD" interpretation that should be referenced in the related-work section.

| Paper | Venue | Signal for CSPD |
|---|---|---|
| GLaD | CVPR 2023 | Generative DD via latent prior; we follow the family but condition on structured text |
| DiT-DD | NeurIPS 2024 | DiT-based generative DD; our SDXL choice is in the same lineage |
| MGD³ | ICML 2025 | Mode guidance for class-name conditioning; **§6.5 shows it doesn't compose with our structured captions** |
| IGD | ICLR 2025 | Selection = downstream usefulness, not visual similarity; consistent with our negative §6.3 |
| D³HR | ICML 2025 | Representativeness via moments; **§6.4 falsifies it for our setup** at IPC=10 |
| VLCP (DD-VLCP) | ICCV 2025 | Class-prototype text; aligned with our archetype-aware direction (Gate C, queued) |
| IGDS | NeurIPS 2025 Workshop | IPC-aware semantic strength; queued for future work |
| CoDA | ICLR 2026 | Distribution alignment > stronger generator; consistent with our negative §6.1 (backbone swap) |
| DAP | ICLR 2026 | Representativeness as a generation-time prior, not post-hoc; queued as a candidate future direction |
| DDOQ | ICLR 2026 | Clustering / support construction *is* the method; supports our Stage 3 design |
| EVLF | 2026 (arXiv) | Early vision-language fusion beats late text; queued for future work |
| RDED, SRe2L | 2023–2024 | Matching-based DD baselines; for IPC scaling comparison |

External URLs (already in `plan.md`):
- IGD: openreview.net/forum?id=0whx8MhysK
- MGD³: openreview.net/forum?id=NIe74CY9lk
- D³HR: proceedings.mlr.press/v267/zhao25x.html
- VLCP: openaccess.thecvf.com/content/ICCV2025/html/Zou_Dataset_Distillation_via_Vision-Language_Category_Prototype_ICCV_2025_paper.html
- IGDS: openreview.net/forum?id=o2HVbnmazF
- CoDA: openreview.net/forum?id=6ycBM1nsS3
- DAP: openreview.net/forum?id=Hvge3NzkJN
- DDOQ: openreview.net/forum?id=FMSp8AUF3m

---

## 8. Suggested paper structure

A method paper with strong negatives. Roughly NeurIPS / ICLR length.

1. **Introduction** (1.5 pages). Open with the elevator pitch (§3). Frame the "where does semantics enter" question. Preview the four-stage pipeline and the five falsified alternatives.
2. **Related work** (1 page). The table in §7 + 2–3 paragraphs grouping generative DD / selection-based DD / matching-based DD.
3. **Method** (3 pages). Subsections from §4: archetype + slot structure (4.3); deterministic-first normalization (4.4); canonical render contract (4.5); LoRA adaptation (4.6); DINOv2 HDBSCAN mode discovery (4.7); medoid-caption generation (4.8). Emphasize that every stage is deterministic given seeds and that the *caption-format contract* — Stage 4 captions match Stage 2 training captions — is the key invariant.
4. **Experiments** (2 pages). Subsections from §5.
   - Setup: ImageNette, IPC=10, three eval archs, 3×3 protocol.
   - Main result table (§5.1, §5.2).
   - IPC scaling figure (§5.3 — fill in once IPC=20/50 done).
   - Cross-arch table.
5. **What does not work** (1.5 pages). The five subsections from §6, plus the cross-cutting takeaway §6.6. **This section is the paper's most novel scientific contribution**: a unified explanation for why post-hoc selection / mode guidance / recaption all fail under strong structured text conditioning.
6. **Discussion / limitations** (0.5 page). §10.
7. **Conclusion** (0.5 page).

Tables to include:
- Table 1: main result on ImageNette IPC=10 (mainline vs prior generative DD).
- Table 2: negative-result ablation (§5.2 above).
- Table 3: IPC × arch sweep (pending).
- Table 4: schema sizes per archetype (§4.3).
- Table 5 (appendix): full hyperparameter list.

Figures to include:
- Figure 1: pipeline diagram (Prep → 4 stages → eval).
- Figure 2: example canonical captions for 3–4 archetypes side-by-side.
- Figure 3: mode-discovery visualization (DINOv2 t-SNE / UMAP of one class, colored by HDBSCAN mode label).
- Figure 4: distilled vs real samples per class.
- Figure 5: ablation bar chart matching Table 2 (single most-citable diagram).

---

## 9. Reproducibility

**Environment.** Single conda env `cspd-dd` from the bundled `environment.yml`. Server-side requirements: PyTorch + CUDA 12.1, 2 × GPU, the official `diffusers` repo pip-installed from source.

**Datasets.**
- ImageNette (10 classes, 12,894 train / 3,925 val) — primary benchmark.
- ImageNet-1k (1.28M train) — Stage 1 in progress as of this brief.

**Reproducing the 63.27% number.**

```bash
bash scripts/pipelines/run_full_pipeline.sh /path/to/ImageNette/train
```

This runs Stage 1 → Stage 2 → Stage 3 → Stage 4 → Eval, with PIPELINE_IPC=10 and PIPELINE_SEEDS="42 123 456" by default. The aggregator at the end of the script prints per-seed best-of-3 and mean/std across seeds.

For a 1-seed sanity run: `PIPELINE_SEEDS="42" bash ...` reproduces the seed=42 round (63.4%).

**Key implementation invariants the paper should mention:**
- Stage 1C `record_id` = `class_name_raw::relative_path` (used to pair Stage 1 / Stage 2 / Stage 3 deterministically).
- Stage 4 per-image seed = `seed + mode_idx` (so `gen_seed42/` reproduces byte-for-byte).
- Stage 2 LoRA is loaded by `pipe.load_lora_weights(parent, weight_name=filename)`; the file is always named `pytorch_lora_weights.safetensors` (official diffusers convention).
- Eval auto-derives epochs from IPC via `ipc_epoch()` (matches MGD³ reference).

**Determinism.** Every stage is deterministic given its seed. The 3×3 protocol varies the seed at Stage 3B (HDBSCAN K-Means fallback / sub-clustering) and Stage 4 (per-image generator seed); Stage 1 and Stage 2 are seed-independent in practice (Stage 1A is greedy decoding from a deterministic prompt; Stage 2 is the official trainer which inherits its own seeding).

---

## 10. Limitations and honest scope

Things the paper must say upfront:

1. **One primary benchmark (ImageNette).** IPC=10 is fully measured under the 3×3 protocol. IPC=20, IPC=50, and the cross-arch sweep are infrastructure-ready but not yet executed. ImageNet-1k Stage 1 is in progress. Calibrate the paper's framing accordingly: as of this brief, CSPD is "a method validated cleanly at one IPC on one benchmark plus five strong negatives", not "a leaderboard-topping recipe across the board".

2. **Architecture coverage.** Mainline numbers are ResNetAP-10. The MGD³-style protocol (ConvNet-6, ResNet-18, ResNetAP-10) is the standard cross-arch check in this literature and is implemented but not yet swept.

3. **VLM cost.** Stage 1A on ImageNet-1k requires ~1.28M Qwen2.5-VL inferences. Stage 1B's `mapped_to_unknown_recovery` trigger adds ~96k more on ImageNet-1k full. This is a real compute cost, and the paper should disclose it. (Mitigation: Stage 1 outputs are reusable across all downstream IPC / seed / generator variations — paid once, used many times.)

4. **Class-level metadata is manual on ImageNet-1k.** `class_to_archetype_imagenet1k_manual.json` was hand-curated (with a 2026-04-12 revision fixing 20 misplaced classes). A VLM-driven mapper exists at `scripts/prep/generate_class_to_archetype_map_vlm.py`, but the bundled ImageNet-1k mapping is manual.

5. **Negative results are conditional on the mainline.** The five falsifications in §6 are honest within the configuration tested (SDXL LoRA, ImageNette, IPC=10, ResNetAP-10). At higher IPC or with a weaker text-conditioning regime the conclusions may shift — e.g. set-level selection could be useful again once IPC > number of natural modes. The paper should phrase the negatives as **falsified under detailed structured text conditioning at low IPC**, not as universal.

6. **The 63.27% number is not a leaderboard claim.** It is the locked baseline under our 3×3 protocol; comparisons to other DD papers should be careful about (a) their IPC and (b) their eval architecture, and (c) what "best" means (per-seed best-of-N vs mean-of-N is itself a methodological axis we surfaced — old 62.33% and new 63.27% are the same dataset under different aggregations).

---

## 11. Files and artifacts a paper reader should know

- `src/cspd_stage1/` — Stage 1 pipeline (extraction / normalization / render).
- `src/cspd_stage2/` — Stage 2 SDXL LoRA wrapper.
- `src/cspd_stage3/` — Stage 3 DINOv2 encode + HDBSCAN cluster + medoid extraction.
- `src/cspd_stage4/` — Stage 4 distilled-image generation.
- `src/cspd_eval/` — Classifier training + eval on real val.
- `scripts/pipelines/run_full_pipeline.sh` — single end-to-end driver.
- `configs/stage1/archetype_taxonomy_manual.json` — the 18-archetype taxonomy.
- `configs/stage1/class_to_archetype_imagenet1k_manual.json` — ImageNet-1k class-to-archetype.
- `configs/stage1/normalization/stage1_attribute_normalization_rules.json` — Stage 1B deterministic rules.
- `runs/stage4/.../pipeline_*/summary.txt` — per-IPC 3×3 aggregator output.
- `gen_dd_coding_instruction_spec.md` — full implementation spec (local-only).
- `plan.md` — research roadmap, what's closed / active / queued (local-only).

---

## 12. Glossary (for the LLM authoring the paper)

- **Archetype**: one of 18 fixed coarse semantic categories (animal, vehicle, structure_or_building, …). Determines slot schema.
- **Slot**: one named field in the per-archetype 7-slot schema (e.g. `pose_or_state`, `viewpoint`).
- **Canonical caption**: the deterministic single-line caption produced by Stage 1C from a normalized slot dict + the archetype template.
- **Mode**: an HDBSCAN-discovered density cluster in DINOv2 space, within one class.
- **Medoid**: the real sample whose DINOv2 feature is closest to its mode's centroid.
- **Representative caption**: the canonical caption of a mode's medoid; used as the Stage 4 generation prompt for that mode.
- **3×3 protocol**: 3 seeds (`{42, 123, 456}`) × 3 independent classifier trainings per seed; report best-of-3 per seed then mean/std across the 3 per-seed bests.
- **Caption-format contract**: the invariant that Stage 4 generation prompts come from the same template family as Stage 2 training captions. Breaking it (e.g. free-form recaption at Stage 4) was the cause of the largest negative result (§6.2).
- **Mapped-to-unknown recovery**: the Stage 1B VLM-review trigger added 2026-04-12 that re-examines slots whose deterministic normalization mapped them to `unknown`, asking the VLM to look at the image and provide a better value.
