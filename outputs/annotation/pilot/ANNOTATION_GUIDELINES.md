# TerraMind AI — Blinded Human Annotation Protocol & Guidance

**Pilot Scope:** 178 class-balanced images (2 images per class across 89 disease classes; 89 train, 89 validation, 0 held-out test).  
**Tool URL:** `http://127.0.0.1:8040`  
**Blinding Level:** Double-blind (opaque aliases `obs-xxxxxxxxxxxx`; no file names, crop labels, disease names, or split information exposed).

---

## 1. Core Principles

1. **Describe Visual Features Only:**
   - Report what you see directly on the leaf/plant surface: color, texture, lesion morphology, border definition, spatial distribution.
2. **Never Guess or Assert a Disease Diagnosis:**
   - **Prohibited:** Stating diagnosis names (e.g., "apple scab", "powdery mildew", "bacterial wilt", "blight", "virus").
   - **Reason:** Disease names leak the ground-truth target, violating multimodal fusion evaluation validity. Any description containing disease terms will be automatically rejected by the submission validator.
3. **Never Guess Plant Species Unless Obvious & Visible:**
   - Focus on symptom manifestation on the organ (leaf, stem, fruit surface) rather than speculating on cultivar or unverified context.
4. **Be Objective and Precise:**
   - Avoid vague adjectives like "sick", "bad", "unhealthy", "damaged".
   - Use precise visual terminology: "chlorotic halos", "dark necrotic lesions", "velvety margins", "interveinal yellowing", "concentric rings", "pustules on abaxial surface".

---

## 2. Standardized Visual Vocabulary Reference

| Feature Dimension | Recommended Descriptive Terms | Terms to Avoid |
| :--- | :--- | :--- |
| **Color / Pigmentation** | Chlorotic (pale yellow/light green), necrotic (brown/black), stippled, mottled, purplish-bronze, bleached white | "Diseased", "dying", "ugly" |
| **Lesion Morphology** | Circular spots, angular water-soaked lesions, irregular blotches, pustules, target-like concentric rings | "Scab spots", "rot marks", "blight areas" |
| **Lesion Margins / Borders** | Sharply defined, diffuse edges, surrounded by a prominent yellow/chlorotic halo, raised margins | "Infected border" |
| **Surface Texture** | Velvety, powdery/fluffy coating, sunken/cankered, blistered, crinkled, rugose, pustular | "Fungal growth", "moldy" |
| **Distribution / Pattern** | Marginal (along edges), apical (leaf tip), interveinal (between veins), scattered across adaxial surface, clustered | "Spreading virus" |

---

## 3. Step-by-Step Annotator Instructions

1. **Access Interface:** Open `http://127.0.0.1:8040` in your web browser.
2. **Set Observer ID:** Enter your unique assigned identifier (e.g., `expert_annotator_01`) and click **Save Observer ID**.
3. **Inspect Image:** Examine the displayed plant image behind its opaque alias (e.g., `obs-bc6f1eebeb2a`).
4. **Type Visual Description:** Write a concise 1–3 sentence objective description of visible leaf/tissue symptoms.
   - *Example:* *"Upper leaf surface displays numerous scattered dark-brown circular lesions (1-3mm diameter), each surrounded by a distinct yellow chlorotic halo. Tissue between lesions appears green and intact."*
5. **Save & Next:** Click **Save observation & next**. If valid, the observation is atomically written to `annotator_workbook.csv` and the next image loads.
6. **Skip if Ambiguous:** If the image is blurry, corrupted, or completely unidentifiable, click **Skip this image** to review it later.

---

## 4. Independent Review & Adjudication Protocol

Once annotations are completed:
1. **Separation of Roles:** An independent reviewer (must NOT be the annotator who wrote the observation) reviews each entry.
2. **Review Criteria:**
   - Factual accuracy: Does the text match the visual image?
   - Target leakage check: Is any disease name or target label leaked?
   - Provenance: Is the observer ID recorded and valid?
3. **Ledger Recording:** The reviewer marks `approved` or `rejected` with their reviewer ID and date in `outputs/annotation/pilot/review_ledger.csv`.
4. **Exclusion:** Any observation marked `rejected` or `pending` will be strictly excluded by `scripts/validate_annotations.py`.
