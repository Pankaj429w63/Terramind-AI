# TerraMind AI — Multimodal Fusion Strategy & Scientific Decision Report

**Date:** 2026-10-18  
**Scope:** Determine the most scientifically defensible next step for training the three multimodal fusion architectures (Early, Intermediate, Late) following the evaluation of Prompt `v3`.  
**Status:** **TRAINING WITHHELD (Audit Gates Active)** — No fusion models trained, official splits preserved, production EfficientNet-B0 weights untouched.

---

## 1. Executive Summary: Evidence-Based Finding on Prompt v3

The prompt evaluation conducted on an $n=89$ class-balanced sample ([metrics.json](file:///c:/Users/PANKAJ%20YADAV/OneDrive/Desktop/Terramind%20AI/outputs/research/prompt_v3_evaluation/metrics.json), [report.md](file:///c:/Users/PANKAJ%20YADAV/OneDrive/Desktop/Terramind%20AI/outputs/research/prompt_v3_evaluation/report.md)) evaluated the candidate symptom prompt:
$$\text{Prompt } v3: \text{"symptoms visible on this leaf include"}$$

While the implementation of normalized prefix stripping ([generation.py](file:///c:/Users/PANKAJ%20YADAV/OneDrive/Desktop/Terramind%20AI/ml/annotation/generation.py#L332-L365)) successfully resolved the tokenizer hyphen-spacing failure (reducing mean word length from 15.1 to 7.36 words) and reduced obvious non-plant misidentifications from 8.99% to 2.25%, the evaluation demonstrated two fatal scientific flaws in Prompt `v3` when paired with BLIP-base:

1. **Persistent Cross-Class Caption Collisions (35.96%):** 32 of the 89 images shared verbatim identical captions across completely different disease classes. For example, `"a few of the leaves that have been removed"` was produced for 8 distinct disease classes (`apple mosaic virus`, `blueberry rust`, `ginger leaf spot`, `maple tar spot`, `potato leaf`, `raspberry leaf`, `tobacco leaf`, `zucchini yellow mosaic virus`).
2. **Severe Rise in Unsupported Disease Claims (25.84%):** When prompted with `"symptoms visible on this leaf include"`, BLIP-base frequently completed the prompt tautologically with generic diagnostic assertions rather than morphology:
   - `"a few of the disease"` (7 images across 7 classes)
   - `"a few signs of disease"` (5 images across 5 classes)
   - `"a few of the signs of disease"` (2 images across 2 classes)
   - `"a number of signs of disease"` (2 images across 2 classes)

Under project quality gates ([quality.py](file:///c:/Users/PANKAJ%20YADAV/OneDrive/Desktop/Terramind%20AI/ml/annotation/quality.py#L31-L38)), asserting `"disease"` without visual morphological grounding triggers the `unsupported_disease_claim` filter, blocking 25.84% of rows from clean emission.

> **Definitive Decision:** Re-generating the 17,870-image corpus using Prompt `v3` with BLIP-base is **strictly prohibited**. It would replace prompt-echo padding with generic tautological disease assertions and 35.96% cross-class collisions, yielding a text modality that carries near-zero discriminative signal.

---

## 2. Inventory of Existing Project Data Sources & Code

A comprehensive inspection of the workspace was conducted to assess whether any existing text source can provide genuinely image-specific symptom descriptions:

| Asset / Source | Location | Current State | Usable for Fusion Training? | Rationale |
|---|---|---|:---:|---|
| **Completed V1 Store** | `outputs/annotation/model_generated/generation_records.csv` | 17,870 rows | **NO** | 100% prompt echo retention (`"a close - up view..."`), 77.4% duplicates, 72.2% cross-class collision rate across the full corpus. |
| **Paired Model CSV** | `data/paired_model_observations.csv` | 17,761 rows | **NO** | Formatted V1 text; flagged as `REVIEW_REQUIRED` by `scripts/audit_fusion_data.py`. Non-human machine text with high duplication. |
| **Synthetic Prompts** | `data/plantwild_prompts.json` | 4,600 rows (50/class) | **NO** | Class-conditioned GPT prompts generated from class names. Using these causes direct ground-truth label leakage into text. |
| **Annotator Workbook** | `outputs/annotation/annotator_workbook.csv` | 17,870 rows (blinded aliases) | **NOT YET** | 0 of 17,870 rows filled (`Filled observations: 0`). No human observations exist yet. |
| **Review Ledger** | `data/annotation/review_ledger_template.csv` | Template only | **NOT YET** | Empty. Independent review ledger required by `scripts/validate_annotations.py` does not exist. |
| **Blinded Review Tool** | `ml/annotation/review_tool.py` | Fully functional FastAPI app | **YES (Tooling ready)** | Blinded web interface running locally on `http://127.0.0.1:8040`. Ready to collect un-leaked human observations. |

**Scientific Conclusion:** There is **zero** pre-existing, genuinely image-specific, label-free text available in the repository. Per project guidelines, human observations must never be simulated or fabricated.

---

## 3. Comparative Assessment of Technical Options in a CPU-Only Environment

We evaluate four potential paths for producing paired text, taking into account hardware constraints (Windows, multi-core CPU, `torch.cuda.is_available() == False`, no dedicated GPU accelerator).

### Option 1: Full-Corpus Regeneration with Prompt V3 (BLIP-base, CPU)
- **Feasibility:** High (~15–20 hours of CPU compute).
- **Quality / External Validity:** **Failing.** 25.84% unsupported disease claims, 35.96% cross-class collisions.
- **Scientific Impact:** Training a text classifier or fusion model on this text would yield an uninformative model that learns to associate `"a few of the disease"` with 14 different diseases.
- **Verdict:** **REJECTED.**

### Option 2: Larger Open-Source Vision-Language Model on CPU (e.g., BLIP-large, Qwen2-VL-7B, LLaVA-1.5)
- **Feasibility:** **Extremely Poor on CPU.**
  - BLIP-base is 247M parameters (~1.6s/image on CPU).
  - BLIP-large is 470M parameters (~4–6s/image on CPU).
  - 7B Instruction-following VLMs (Qwen2-VL, LLaVA) require >14 GB RAM in float16 and take **45–90 seconds per single image** on CPU.
  - Generating 17,870 images with a 7B model on CPU would require **220 to 450 hours (9 to 19 days)** of continuous compute, with severe risk of memory swapping and thermal throttling.
- **Quality Risk:** Without domain-specific phytopathology fine-tuning, general-domain VLMs still exhibit caption template collapse (e.g., repeating `"a close up of a green plant leaf with dark spots"` across diverse species).
- **Verdict:** **UNFEASIBLE FOR FULL CORPUS ON CPU; PERMISSIBLE ONLY AS A BOUNDED 20-IMAGE PILOT PROBE.**

### Option 3: Commercial OpenAI-Compatible / HF Inference Remote VLM API
- **Feasibility:** Moderate (depends on budget, API key availability, and network egress).
- **Cost Estimate:** Remote multimodal endpoints cost ~$0.50 to $2.00 per 1,000 images. Generating 17,870 images would cost **$10 to $40 USD** and requires setting `VLM_BASE_URL`, `VLM_API_KEY`, and explicit cost confirmation (`--confirm-paid-usage`).
- **Data Governance Risk:** Uploading local agricultural research images to external commercial endpoints may violate data privacy or offline-environment constraints.
- **Verdict:** **RESERVED FOR FUTURE EXPLORATION; CURRENTLY BLOCKED BY ABSENCE OF API CREDENTIALS.**

### Option 4: Blinded Human Annotation Pilot via the Existing Review Tool (Recommended)
- **Feasibility:** **High.** The complete infrastructure ([review_tool.py](file:///c:/Users/PANKAJ%20YADAV/OneDrive/Desktop/Terramind%20AI/ml/annotation/review_tool.py), [validate_annotations.py](file:///c:/Users/PANKAJ%20YADAV/OneDrive/Desktop/Terramind%20AI/scripts/validate_annotations.py)) is already built, audited, and verified (44 tests passing).
- **Quality & External Validity:** **Highest.**
  - Perfectly aligns with [text_modality_decision.md](file:///c:/Users/PANKAJ%20YADAV/OneDrive/Desktop/Terramind%20AI/outputs/research/text_modality_decision.md) Option A (`user_observation`).
  - Matches the production inference interface (`POST /api/multimodal/analyze` takes user-typed `paired_text`).
  - Completely eliminates machine caption template collapse, prompt echo, and hallucinated rocks/soil.
  - Zero label leakage (enforced by blinded aliases and lexical checks).
- **Scale:** While annotating all 17,870 images requires a large contributor team, an initial **$n = 178$ class-balanced pilot (2 images per class across all 89 classes)** is completely manageable and provides genuine statistical grounding.
- **Verdict:** **RECOMMENDED PRIMARY PATH.**

---

## 4. Quantitative Acceptance Criteria Matrix

Before committing compute or annotator effort to any larger dataset, any candidate text-generation method (human or machine) must satisfy the following strict quantitative gates measured on a class-balanced sample of at least $n=89$ images:

| Gate | Metric | Current V1 (Store) | Current V3 | Acceptance Threshold | Failure Consequence |
|---|---|---:|---:|---:|---|
| **G1: Distinctiveness** | Unique Caption Rate | 74.16% | 73.03% | **$\ge 85.0\%$** | Template collapse; uninformative text |
| **G2: Separability** | Cross-Class Collision Rate | 34.83% | 35.96% | **$\le 5.0\%$** | Text cannot discriminate classes |
| **G3: Diagnostic Relevance** | Morphological Symptom Rate | 33.71% | 15.73% | **$\ge 70.0\%$** | Text lacks pathology features |
| **G4: Clinical Grounding** | Unsupported Disease Claims | 0.00% | 25.84% | **$< 2.0\%$** | Structural audit failure (hallucinated "disease") |
| **G5: Integrity** | Class-Label Leakage Rate | 0.00% | 0.00% | **$0.00\%$** | Target leakage into inputs |
| **G6: Visual Accuracy** | Image Misidentifications | 8.99% | 2.25% | **$< 1.0\%$** | Severe hallucination (rocks, soil, animals) |

---

## 5. Recommended Small, Reproducible Pilot Proposal

### Phase 1: Blinded Human Annotation Pilot ($n = 178$ Images)
1. **Sample Selection:**
   - Draw 2 images per class across all 89 classes from [fusion_eligibility_manifest.json](file:///c:/Users/PANKAJ%20YADAV/OneDrive/Desktop/Terramind%20AI/outputs/research/fusion_eligibility_manifest.json) using a deterministic seed (`seed=42`).
   - Stratified: 89 in `train`, 89 in `validation`.
2. **Execution via Blinded Review Tool:**
   - Reviewers access `python -m scripts.review_annotations --port 8040`.
   - Reviewers see only the image and opaque alias (`obs-xxxxxxxxxxxx`).
   - Instruction: *"Describe only visible leaf morphology: color, spot patterns, textures, margins, or wilting. Do not guess species or disease names."*
3. **Independent Adjudication:**
   - A second reviewer independently reviews entries into `data/annotation/review_ledger_template.csv`.
4. **Validation:**
   - Execute `python -m scripts.validate_annotations --workbook outputs/annotation/annotator_workbook.csv --alias-map outputs/annotation/sealed/alias_map.json --review-ledger data/annotation/review_ledger.csv`.

### Phase 2: Parallel 20-Image VLM Probe (Zero-Cost Verification)
- Run a bounded probe on 20 images using an instruction-tuned model (e.g., Florence-2-base or BLIP-2 if available, or HF Inference API if a token is supplied) using the strict Prompt `v2` instruction.
- Measure against the Acceptance Criteria Matrix (G1–G6).
- If the model scores $< 85\%$ uniqueness or $> 5\%$ collision, it is immediately discarded.

---

## 6. Safeguards & Mandate Compliance Checklist

- [x] **No Full Corpus Regeneration with V3:** Ruled out due to 25.84% unsupported disease claims and 35.96% cross-class collisions.
- [x] **Strict Modality Separation:** `model_generated_observation` and `user_observation` are strictly segregated; candidate machine text is never relabelled as human observation.
- [x] **Data Integrity Preserved:**
  - Official split files (`train_split.txt`, `val_split.txt`, `test_split.txt`) untouched.
  - [fusion_eligibility_manifest.json](file:///c:/Users/PANKAJ%20YADAV/OneDrive/Desktop/Terramind%20AI/outputs/research/fusion_eligibility_manifest.json) untouched.
  - Production EfficientNet-B0 checkpoint untouched.
  - Completed V1 dataset ([generation_records.csv](file:///c:/Users/PANKAJ%20YADAV/OneDrive/Desktop/Terramind%20AI/outputs/annotation/model_generated/generation_records.csv)) untouched.
- [x] **Fusion Training Withheld:** No fusion models trained prematurely on corrupted or collision-heavy text.
- [x] **Zero Fabrication:** No human review approvals or synthetic observations were invented.

---

## 7. Actionable Next Steps

1. **Keep Training Blocked:** Maintain the training guard in `ml/multimodal/train.py` and `scripts/audit_fusion_data.py` until approved paired data exists.
2. **Deploy Blinded Review Tool for Pilot:** Launch `python -m scripts.review_annotations` for human contributors to annotate the 178-image class-balanced pilot.
3. **Maintain Existing V1 Records as Engineering Baseline:** Retain the 17,870 V1 records as a historical baseline of unguided BLIP captioning, but do not promote them to clinical/production status.
