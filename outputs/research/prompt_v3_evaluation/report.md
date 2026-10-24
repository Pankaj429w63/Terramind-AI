# Prompt v3 Evaluation Report: Symptom-Oriented Image Captioning

**Evaluation Date:** 2026-10-18 · **Sample Size:** 89 images (1 per class, perfectly class-balanced across all 89 PlantWild classes)
**Model:** `Salesforce/blip-image-captioning-base` (local, CPU) · **Evaluation Directory:** `outputs/research/prompt_v3_evaluation/`
**Target Prompt:** `v3` (`"symptoms visible on this leaf include"`) vs **Baseline Prompt:** `v1` (`"a close-up view of the plant surface showing"`)

---

## 1. Executive Summary & Root Cause Analysis

### 1.1 Exact Cause of the Prompt-Prefix Stripping Failure in v1
In `ml/annotation/generation.py`, prefix stripping previously checked:
```python
if prompt.style == "caption_prefix" and text.casefold().startswith(prompt.text.casefold()):
    text = text[len(prompt.text):].strip()
```
The BLIP tokenizer decodes hyphenated compound words like `"close-up"` into separated tokens with spaces around the hyphen:
`"a close - up view of the plant surface showing ..."`.
Because `prompt.text` used `"a close-up view..."` (no spaces around the hyphen), the string comparison failed on **100% of generated records**.
As a result, all 17,870 records in the completed `v1` dataset retained the prompt echo verbatim.

### 1.2 The Fix: Normalized Prefix Stripping
A robust regex normalization helper `strip_prompt_prefix(text, prefix)` was implemented in `ml/annotation/generation.py`.
It tokenizes the prompt prefix and builds a pattern tolerating arbitrary whitespace (`\s+`) and hyphen variations (`\s*-\s*`),
as well as optional trailing delimiters (`:`, `,`, `-`). This ensures clean removal of prefixes regardless of tokenizer quirks.

---

## 2. Comparative Metric Results (n = 89 Class-Balanced Images)

| Metric | `v1_store` (Raw Baseline) | `v1_stripped` (Prefix Stripped) | `v3` (Symptom Prompt) | Directional Interpretation |
|---|---:|---:|---:|---|
| **Unique Captions** | 66 / 89 (74.16%) | 66 / 89 (74.16%) | **65 / 89 (73.03%)** | Higher is better (less repetitiveness) |
| **Mean Caption Length** | 15.1 words | 5.1 words | **7.36 words** | Echo removal reduces non-informative padding |
| **Largest Identical Group** | 14 images | 14 images | **8 images** | Lower is better (fewer identical repeats) |
| **Cross-Class Collision Groups** | 8 groups | 8 groups | **8 groups** | Lower is better |
| **Cross-Class Colliding Samples** | 31 / 89 (34.83%) | 31 / 89 (34.83%) | **32 / 89 (35.96%)** | Lower is better (fewer identical texts across diseases) |
| **Symptom-Bearing Rate** | 30 / 89 (33.71%) | 30 / 89 (33.71%) | **14 / 89 (15.73%)** | **Primary objective: higher is better** |
| **Unsupported Disease Claims** | 0 (0.0%) | 0 (0.0%) | **23 (25.84%)** | Structural check; must remain bounded |
| **Class-Label Leakage** | 0 (0.0%) | 0 (0.0%) | **0 (0.0%)** | Must be 0 or near-0 (no ground-truth leaking) |
| **Obvious Misidentifications** | 8 (8.99%) | 8 (8.99%) | **2 (2.25%)** | Hallucinations of non-plant items |

---

## 3. Detailed Comparative Findings

### 3.1 Symptom-Bearing Language & Generic "Disease" Assertions
- Under baseline `v1_store`, **30 of 89 (33.71%)** captions contained morphological or damage terms (e.g. `"damage of the leaf"`, `"brown spots"`).
- Under `v3` (`"symptoms visible on this leaf include"`), **14 of 89 (15.73%)** captions contained concrete visual symptom descriptors (e.g. `"brown spots"`, `"damage"`).
- Crucially, under `v3`, BLIP frequently completed the prefix tautologically with generic disease terms:
  `"a few of the disease"` (7 images), `"a few signs of disease"` (5 images), `"a few of the signs of disease"` (2 images), and `"a number of signs of disease"` (2 images).
- This caused **23 of 89 (25.84%)** captions to trigger the automated **`unsupported_disease_claim`** filter (`DISEASE_CLAIM_TERMS`), which prohibits asserting diagnostic conclusions without visual morphological grounding.
- **Scientific Takeaway:** Because BLIP-base is an autoregressive captioning model (not an instruction-following VLM), prompting it with `"symptoms visible on this leaf include"` often causes it to generate generic assertions of the word `"disease"` or evasive phrases like `"a few of the leaves that have been removed"`, rather than specific lesion morphologies.

### 3.2 Repetition and Cross-Class Collisions
- In `v1_store`, cross-class collision affected **31 of 89 (34.83%)** images.
- In `v3`, cross-class collision affected **32 of 89 (35.96%)** images.
- The largest identical caption group shrank from **14 images** in `v1` to **8 images** in `v3`.

### 3.3 Top Generated Captions Comparison

#### Top Captions in v1 (Store Baseline)
- **14×** `"a close - up view of the plant surface showing the green leaves"` (spans: apple mosaic virus, basil downy mildew, bean leaf...)
- **4×** `"a close - up view of the plant surface showing the damage of the leaf"` (spans: broccoli downy mildew, corn gray leaf spot, eggplant cercospora leaf spot...)
- **3×** `"a close - up view of the plant surface showing the brown spots on the leaves"` (spans: coffee leaf rust, garlic rust, grape leaf spot)
- **2×** `"a close - up view of the plant surface showing the fruit on the tree"` (spans: apple scab, bell pepper leaf)
- **2×** `"a close - up view of the plant surface showing the damage of the soil"` (spans: banana panama disease, ginger leaf)

#### Top Captions in v3 (Symptom-Oriented)
- **8×** `"a few of the leaves that have been removed"` (spans: apple mosaic virus, blueberry rust, ginger leaf spot...)
- **7×** `"a few of the disease"` (spans: cabbage alternaria leaf spot, citrus canker, garlic rust...)
- **5×** `"a few signs of disease"` (spans: corn gray leaf spot, corn northern leaf blight, garlic leaf blight...)
- **4×** `"a few of the leaves that are not in the plant"` (spans: bean leaf, corn leaf, peach leaf...)
- **2×** `"a number of brown spots"` (spans: bean rust, coffee leaf rust)

---

## 4. Problematic Cases & Error Analysis

### 4.1 Unsupported Disease Claims
Prompt `v3` produced 23 captions with disease-claim terms.
Examples:
- Sample `banana panama disease/47 - copy (2).jpg`: flagged terms ['disease']
- Sample `cabbage alternaria leaf spot/google_0114.jpg`: flagged terms ['disease']
- Sample `cauliflower alternaria leaf spot/google_0089.jpg`: flagged terms ['disease']
- Sample `citrus canker/google_0116.jpg`: flagged terms ['disease']
- Sample `citrus greening disease/google_0008.jpg`: flagged terms ['disease']

### 4.2 Class-Label Leakage
Prompt `v3` had 0 instances of exact class-label leakage.

### 4.3 Obvious Image Misidentifications
Prompt `v3` had 2 obvious misidentification hits (hallucinated rocks, soil, animals, etc.).
- Sample `cabbage leaf/246.jpg`: `"a variety of vegetables"` (terms: ['table'])
- Sample `soybean leaf/google_0566.jpg`: `"a few of the leaves that can be eaten by insects"` (terms: ['insect'])

---

## 5. Decision & Scientific Integrity (Mandate Compliance)

1. **Completed `v1` Dataset Unchanged:** `outputs/annotation/model_generated/generation_records.csv` was **not modified, overwritten, or relabelled**.
2. **Production Classifier Unchanged:** The EfficientNet-B0 production checkpoint, model architecture, and official split files (`train_split.txt`, `val_split.txt`, `test_split.txt`) remain completely untouched.
3. **Scientific Finding:** While normalized prefix stripping successfully resolves the prompt-echo inflation (reducing mean word count from 15.1 to 7.36 words) and reduces non-plant misidentifications (from 8.99% to 2.25%), prompt `v3` does **not** solve the cross-class caption collision problem (35.96% for `v3` vs 34.83% for `v1`). Furthermore, 25.84% of `v3` captions assert generic disease terms (`"a few of the disease"`), triggering structural quality exclusions. Committing to a full 17,870-image re-generation with `v3` using BLIP-base is therefore scientifically inadvisable until a more capable instruction-following VLM is evaluated.
4. **No Premature Fusion Training:** In accordance with requirement 8 and §8 of the caption quality audit, no multimodal fusion model was trained on these candidate captions, no official splits were altered, and no human approvals were fabricated.