# CSPD Research Roadmap Checklist

Status: research-facing planning document, checklist form
Date: `2026-04-19`
Supersedes: `2026-04-17` version (VAE-space set-level was still pending as the last open lever; that experiment has since been run, falsified, and removed)
Scope: an analysis-derived checklist of what is done, what is active, and what is next — with explicit reasons and source provenance for every item
Companion document: `gen_dd_coding_instruction_spec.md`
Repo snapshot assumed: `E:\Project\2026-03-25` (main branch)

---

## 0. How to read this file

`gen_dd_coding_instruction_spec.md` is the implementation-facing source of truth (what the code does).
This file is the research-facing checklist (what to do next and why).

Every item below is either:

- `[x]` done — closed by evidence
- `[ ]` active — next concrete step
- `[~]` deferred — not the right time; reason recorded

Every item has a reason and a source. When the repo state or evidence changes materially, update the relevant checkbox and the reason line, not only the prose.

---

## 1. Evidence policy

Internal evidence:

- local run artifacts under `runs/`
- current spec at `gen_dd_coding_instruction_spec.md`
- implementation state in `src/` and `scripts/`
- git log on the main branch

External evidence:

- primary paper pages only when possible (OpenReview / conference open-access / PMLR / arXiv official page)

Every recommendation below traces back to at least one of: a local experiment, a spec conclusion, a git commit, or an external paper.

---

## 2. Current empirical state of the repo

## 2.1 Infrastructure readiness

- [x] **Stage 1 extraction + render stable on ImageNette**
  - ImageNette extraction: `12894 / 12894` success
    - source: `runs/stage1/attributes/ImageNette_train/qwen_local/2026-04-12_232711/stage1_stats.json`
  - ImageNette render: `12894 / 12894` success
    - source: `runs/stage1/render/ImageNette_train/qwen_local/2026-04-13_111606/render_summary.json`
- [x] **Stage 1 scales to ImageNet-1k 5-shot**
  - render: `4999 / 5000` success
    - source: `runs/stage1/render/ImageNet1k_5shot/qwen_local/2026-04-12_225612/render_summary.json`
- [x] **End-to-end pipeline driver operational**
  - `scripts/pipelines/run_full_pipeline.sh` drives Stage 1 → Stage 2 → Stage 3 → Stage 4 → Eval
  - 3×3 measurement protocol is the default (3 seeds × `EVAL_REPEAT` independent classifier trainings, best-of-REPEAT per seed, mean/std across the three per-seed bests)
  - each stage is idempotent: Stage 1/2/3 skip when their canonical artifact exists; Stage 4 always produces a fresh timestamped run
  - env overrides: `PIPELINE_IPC`, `PIPELINE_SEEDS`, `EVAL_REPEAT`, `STAGE2_EPOCHS`, `LORA_WEIGHTS`
  - commit: `689b251` (folded `run_baseline_3x3.sh` into the full pipeline)

## 2.2 Current best known configuration

- [x] **Mainline locked** as of `2026-04-18`
  - Stage 2: SDXL LoRA, rank `64`, cosine LR, warmup `500`, noise_offset `0.05`, snr_gamma `5.0`, batch `8`, epoch `9`, 2 GPUs, 512 resolution
  - Stage 3: DINOv2 encode + HDBSCAN per-class clustering (K-Means fallback on small clusters) + medoid caption
  - Stage 4: `text2img`, `guidance=7.5`, `steps=50`, `visual_mode=none`
  - Eval: ResNetAP-10, 3×3 protocol (3 seeds × 3 repeats)
  - Accuracy: **`63.27 +/- 0.19`** (IPC=10)
    - per-seed: seed=42 → 63.4, seed=123 → 63.0, seed=456 → 63.4
    - source: commit `5dfd24f` + `runs/stage4/ImageNette_train/ipc10/lora/pipeline_*/summary.txt`
  - Replaces the earlier `62.33 +/- 1.47` number (same underlying eval data for seed=42; only the aggregation changed — old = mean-of-3, new = best-of-3)

## 2.3 Hypotheses closed since 2026-04-17

All five of these were open in the previous plan. All are now closed; their code has been deleted and the spec updated.

- [x] **Switch SDXL → SD v1.5** — falsified
  - sample quality fine, eval worse than SDXL
  - reference: spec section "SD v1.5 backbone experiment (2026-04-16 → 2026-04-17)"
  - code removed: commit `d992e76` (Stage 2 cleanup batch 1 — removed SD v1.5 / FLUX / PixArt family)
- [x] **Free-form / richer Stage 4 recaption** — falsified
  - result: `56.67 +/- 0.50` vs `62.33` baseline (~−6%)
  - source: `runs/eval/2026-04-15_173911_ipc10_resnet_ap/eval_resnet_ap.json`
- [x] **Per-mode multi-candidate (Phase 2) selection in DINO space** — falsified
  - result: `60.8 +/- 0.33` (~−1.5%)
  - source: `runs/eval/2026-04-16_062943_ipc10_resnet_ap/eval_resnet_ap.json`
  - code removed: commit `a36e8d9` (deleted `src/cspd_stage4/candidate_selection.py`)
- [x] **MGD³-style mode guidance** — falsified under structured captions
  - either no content effect or image-quality collapse; no usable sweet spot
  - code removed: commit `a36e8d9` (deleted `src/cspd_stage4/mode_guidance.py`)
- [x] **Set-level representativeness selection (Phase 3)** — falsified in both DINOv2 and VAE spaces
  - DINOv2-space: `59.53 +/- 0.38` (`−2.80`)
    - source: `runs/eval/2026-04-17_210019_ipc10_resnet_ap/eval_resnet_ap.json`, commit `57b72f0`
  - VAE-space (the "one more try" from the 2026-04-17 plan): `59.07 +/- 0.??` (`−3.26`)
    - source: commit `d81b47b` ("Record Phase 3 VAE-space A/B: 59.07% (-3.26%), close set-level line at IPC=10")
  - code removed: commit `a36e8d9` (deleted `src/cspd_stage4/representativeness.py`)

## 2.4 No high-priority hypothesis is currently active

As of `2026-04-19`, there is no single-experiment lever pending.
The next work has to come from either **benchmark breadth** (Section 5.2) or **method redirection** (Section 5.3).

---

## 3. Updated bottleneck diagnosis

## 3.1 Stage 1 is not the bottleneck

Confidence: high.
Reasoning: extraction and render are already near-perfect on ImageNette and near-complete on ImageNet-1k 5-shot. Recent regressions traced to Stage 2/3/4 method choices, not Stage 1.

## 3.2 Backbone family is not the bottleneck

Confidence: high.
Reasoning: SD v1.5 was a fair test of the "native-512 wins" hypothesis and lost to SDXL despite SDXL's resolution mismatch. Backbone swapping is no longer the top research axis.

## 3.3 Post-hoc selection is not the bottleneck (closed line)

Confidence: high.
Reasoning: three independent selection methods (multi-candidate / DINOv2 set-level / VAE set-level) all regressed materially. The issue was not "wrong space" or "wrong normalization" — the VAE test controlled for both. Post-hoc representativeness is not where the gains are.

## 3.4 Late / overly strong text conditioning remains a structural issue

Confidence: very high.
Reasoning: detailed Stage 1 captions help Stage 2 training, but the same text at Stage 4 generation time dominates the sample, crowding out mode guidance, reranking, and set-level repair. This is the cleanest unified explanation for all five closed negatives.
Implication: future method work must change **how and when** semantics enter generation, not just the selector downstream.

## 3.5 Primary bottleneck is now "utility alignment + benchmark breadth"

Confidence: high.
Reasoning: the mainline produces plausible images and a strong single-point number. The open questions are: (a) does the gain hold across IPC and architectures? (b) can a different semantic-conditioning regime break above 63.3%?

Implication: benchmark curve work has become a first-class research activity, not a pre-publication afterthought.

---

## 4. Literature signals (unchanged but re-prioritized)

The nine papers enumerated in the 2026-04-17 plan still apply; the takeaway is now sharper given the closed experiments.

| Paper | Signal for CSPD | Still actionable after 2026-04-18? |
|---|---|---|
| IGD, ICLR 2025 | selection = downstream usefulness, not visual similarity | weak (all our selection sweeps failed) |
| MGD³, ICML 2025 | keep mode discovery; mode guidance fails under strong text | confirms current Stage 3 + closes mode-guidance line |
| D3HR, ICML 2025 | model-native latent representativeness | closed — tested in VAE space, lost |
| VLCP, ICCV 2025 | text as prototype, not long prompt | **highly actionable** — Gate C direction |
| IGDS, NeurIPS 2025 Workshop | IPC-aware semantic strength | **highly actionable** — Gate C direction |
| CoDA, ICLR 2026 | distribution alignment > stronger generator | confirms backbone-swap dead end |
| DAP, ICLR 2026 | representativeness as generation-time prior, not post-hoc | **actionable** if Gate C reopens latent-space work |
| DDOQ, ICLR 2026 | clustering/support construction IS the method | confirms current HDBSCAN+medoid as core, not placeholder |
| EVLF, 2026 | early vision-language fusion beats late text | **highly actionable** — Gate C direction |

Common thread: keep mode discovery, treat support construction as method, move semantics earlier and lighter. Exactly the direction Gate C points to.

---

## 5. Active checklist — what to do next

## 5.1 Priority 0: protect the mainline

- [x] Mainline defined and locked (Section 2.2)
- [x] Reproducible via `scripts/pipelines/run_full_pipeline.sh`
- [x] 3×3 protocol canonicalized
- [ ] **Tag the mainline commit** and record it in the spec as the frozen reference point for all future A/Bs
  - Reason: prevent accidental baseline drift during Gate C exploration
  - Source: commit `6b43391` (docs sync, latest clean state)

## 5.2 Priority 1 (ACTIVE): strengthen the benchmark curve — Gate B

The 2026-04-17 plan named this Priority 2 / Gate B, behind the pending VAE-space test. With VAE-space closed, this is now the #1 active work.

- [ ] **Run IPC sweep on the mainline**
  - `PIPELINE_IPC="10 20 50" bash scripts/pipelines/run_full_pipeline.sh <train_root>`
  - required because: recent DD papers are judged on the low-IPC curve, not a single IPC=10 number
  - success condition: clean mean/std for each IPC; regression or flatness at IPC=20 / IPC=50 is itself diagnostic information
  - infrastructure: already in place (`PIPELINE_IPC` env var, parameterized per-IPC paths)
- [ ] **Run all three eval architectures**
  - ConvNet-6, ResNet-18, ResNetAP-10 — 3 repeats each
  - current `scripts/eval/run_eval_pipeline.sh` supports `all` as the arch selector
  - required because: cross-architecture transfer is the standard robustness check in DD literature (IGD, DAP, DDOQ all report it)
- [ ] **Produce a baseline benchmark table**
  - rows: IPC ∈ {10, 20, 50}
  - cols: ConvNet-6 / ResNet-18 / ResNetAP-10 (mean ± std, 3×3)
  - output: commit the table into the spec once stable
- [ ] **Verify mainline reproducibility on a fresh machine**
  - `environment.yml` + `scripts/pipelines/run_full_pipeline.sh` one-shot on a clean server
  - reason: the spec promises this; verifying it before publication writeup

Stop condition for Gate B: once the baseline table is committed and reproducible, move to Gate C.

## 5.3 Priority 2: method redirection — Gate C

Enter Gate C **only after** Gate B is done. Preferred entry point: earlier and lighter generation-time semantics. Concrete candidates (each is a separate branch, not a single experiment):

- [~] **Early-fusion semantic conditioning at Stage 2**
  - literature: EVLF, VLCP
  - sketch: inject class-prototype information during LoRA adaptation so Stage 4 can use lighter prompts at inference
  - gating: only if Gate B shows the mainline plateaus or regresses at higher IPC
- [~] **Prototype-level Stage 4 prompts**
  - literature: VLCP
  - sketch: compress Stage 4 generation prompts to class-critical prototype slots; keep rich Stage 1 captions only for Stage 2 training
  - gating: only after Gate B
- [~] **IPC-aware semantic strength**
  - literature: IGDS
  - sketch: IPC=10 → prototype-heavy / low-context; IPC=20 → moderate context; IPC=50 → more contextual variation
  - gating: requires Gate B's IPC curve to define the IPC-specific regimes
- [~] **Generation-time representativeness as a prior (not a post-hoc score)**
  - literature: DAP
  - note: this is distinct from the closed post-hoc selection line; the hypothesis is that representativeness must be baked into the sampler, not scored afterward
  - gating: lowest priority; open only if the above three don't move the needle

Stop condition for Gate C: any branch that clears the `63.27%` baseline by ≥ 1% on the 3×3 protocol at IPC=10 AND holds or improves at IPC=20/50 becomes a candidate new mainline.

## 5.4 Priority 3: ImageNet-1k scaling — deferred until ImageNette is stable

- [x] Stage 1 already verified on ImageNet-1k 5-shot
- [~] Stage 2/3/4 + eval at ImageNet-1k scale
  - reason for deferral: scaling an unstable method makes interpretation harder; need the Gate B baseline table first
  - gating: after Gate B completes cleanly on ImageNette

---

## 6. Stop list — what not to spend time on unless new evidence appears

- [x] More SD v1.5 (or other non-SDXL) backbone work — closed by the 2026-04-17 experiment
- [x] More DINOv2-space set-level / moments / MMD tuning — closed by the `59.53%` regression
- [x] More VAE-space set-level tuning — closed by the `59.07%` regression (this was the explicit "one more try" lever)
- [x] More mode-guidance scale sweeps under long structured captions — closed structurally, not just mis-tuned
- [x] More free-form recaption experiments at Stage 4 — closed by the `56.67%` regression
- [x] More small selector tweaks without a stronger utility story — closed by the three independent selection failures

---

## 7. Concrete roadmap and stop conditions

## 7.1 Gate A: set-level question — CLOSED ✓

- Both DINOv2-space and VAE-space variants tested; both regressed.
- Code removed in commit `a36e8d9`.
- Decision: set-level / post-hoc representativeness is not the right lever for this project.
- Record: spec sections on Phase 3 DINOv2 + Phase 3 VAE-space.

## 7.2 Gate B: baseline benchmark curve — ACTIVE

Run:
- IPC `10`, `20`, `50`
- eval archs: ConvNet-6, ResNet-18, ResNetAP-10
- 3×3 protocol (3 seeds × 3 repeats)

Output:
- clean benchmark table
- one reproducible command from the repo

Once output is in the spec, Gate B closes.

## 7.3 Gate C: Phase 4 method work — GATED ON GATE B

Preferred entry points, in rough order:
1. early-fusion semantic conditioning (Stage 2)
2. prototype-level Stage 4 prompts
3. IPC-aware semantic strength
4. representativeness-as-sampler-prior (not post-hoc)

Not preferred (these paths have been closed):
- another backbone swap
- another post-hoc selector in any feature space
- more late-prompt engineering

---

## 8. Bottom-line thesis of the current project state

The project has one coherent working line:

- **Stage 1 semantics + Stage 2 SDXL LoRA + Stage 3 HDBSCAN medoid + Stage 4 text2img → `63.27 ± 0.19` at IPC=10, 3×3.**

The bottleneck has shifted from "is it runnable?" (closed) and "which post-hoc selector?" (closed) to two things:

1. **Benchmark breadth** — the single-point 63.27% needs to become an IPC × arch curve before it carries weight in the literature.
2. **Method redirection** — if higher numbers are desired, the next lever is earlier/lighter semantics, not more selection.

The current recommendation is therefore:

1. finish Gate B (benchmark curve) — infrastructure is already in place
2. then enter Gate C (early-fusion / prototype-level / IPC-aware semantics) selectively
3. defer ImageNet-1k method claims until ImageNette is curve-stable

---

## 9. Source register

## 9.1 Internal sources (current)

Code (post-cleanup, as of `2026-04-19`):

- `src/cspd_stage1/` — Stage 1 canonical pipeline
- `src/cspd_stage2/` — SDXL-only Stage 2 (after cleanup batches 1/2/3)
- `src/cspd_stage3/` — DINOv2 + HDBSCAN + medoid (after 2026-04-18 cleanup)
- `src/cspd_stage4/` — text2img + optional img2img-from-medoid + optional refiner (after 2026-04-18 cleanup)
- `src/cspd_eval/` — ConvNet-6 / ResNet-18 / ResNetAP-10
- `scripts/pipelines/run_full_pipeline.sh` — end-to-end driver

Spec:

- `gen_dd_coding_instruction_spec.md` (authoritative technical spec, covers all stages post-cleanup)

Key commits (since 2026-04-17 plan):

- `5dfd24f` — new 3×3 baseline `63.27 ± 0.19`
- `d81b47b` — VAE-space set-level falsified
- `a36e8d9` — Stage 4 cleanup (set-level / multi-candidate / mode-guidance / VAE encoding removed)
- `2f439f1` — Stage 3 cleanup (caption-diversification / --cluster-method removed)
- `d992e76`, `f116470`, `6964de8` — Stage 2 cleanup batches 1/2/3 (non-SDXL families removed)
- `8a109a5`, `345432e`, `41fd074`, `689b251` — pipeline driver hardening
- `5fbf9fa` — eval output path mirrors Stage 4 hierarchy
- `6b43391` — README + code-docs sync to match post-cleanup state

Eval artifacts:

- baseline `63.27`: per-IPC summary under `runs/stage4/ImageNette_train/ipc10/lora/pipeline_*/summary.txt` (3×3 round, seed=42/123/456)
- old `62.33 ± 1.47`: `runs/eval/2026-04-17_150749_ipc10_resnet_ap/eval_resnet_ap.json`
- K-Means variant `62.13`: `runs/eval/2026-04-17_150753_ipc10_resnet_ap/eval_resnet_ap.json`
- recaption regression `56.67`: `runs/eval/2026-04-15_173911_ipc10_resnet_ap/eval_resnet_ap.json`
- multi-candidate regression `60.8`: `runs/eval/2026-04-16_062943_ipc10_resnet_ap/eval_resnet_ap.json`
- DINOv2 set-level regression `59.53`: `runs/eval/2026-04-17_210019_ipc10_resnet_ap/eval_resnet_ap.json`
- VAE set-level regression `59.07`: referenced in commit `d81b47b`

Stage 1 readiness artifacts:

- `runs/stage1/attributes/ImageNette_train/qwen_local/2026-04-12_232711/stage1_stats.json`
- `runs/stage1/render/ImageNette_train/qwen_local/2026-04-13_111606/render_summary.json`
- `runs/stage1/render/ImageNet1k_5shot/qwen_local/2026-04-12_225612/render_summary.json`

## 9.2 External paper sources

- IGD, ICLR 2025: https://openreview.net/forum?id=0whx8MhysK
- MGD³, ICML 2025: https://openreview.net/forum?id=NIe74CY9lk
- D3HR, ICML 2025: https://proceedings.mlr.press/v267/zhao25x.html
- VLCP, ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/html/Zou_Dataset_Distillation_via_Vision-Language_Category_Prototype_ICCV_2025_paper.html
- IGDS, NeurIPS 2025 Workshop: https://openreview.net/forum?id=o2HVbnmazF
- CoDA, ICLR 2026: https://openreview.net/forum?id=6ycBM1nsS3
- DAP, ICLR 2026: https://openreview.net/forum?id=Hvge3NzkJN
- DDOQ, ICLR 2026: https://openreview.net/forum?id=FMSp8AUF3m
- EVLF, 2026: https://arxiv.org/abs/2603.07476
