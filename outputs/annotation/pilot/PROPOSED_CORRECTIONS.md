# TerraMind AI — Proposed Pilot Annotation Corrections Dossier

**Target Dataset:** PlantWild Blinded Pilot (178 class-balanced samples)  
**Input Reference:** `outputs/annotation/pilot/annotator_workbook.csv`  
**Review Report:** `outputs/annotation/pilot/quality_review_report.csv`  
**Structured Corrections CSV:** `outputs/annotation/pilot/proposed_corrections.csv`  
**Status of Original Workbook:** **Preserved verbatim** (No automatic in-place modification).  
**Status of Review Ledger:** **Preserved verbatim** (All 178 entries remain `pending`).

---

## 1. Deep Dive on the 9 Flagged Cases

### Case 1: `garlic rust/google_0003.jpg` (Alias: `obs-ed667e8194fd`)
- **Observer ID:** `obs-ddc1912f882f` (Date: `2026-10-20`)
- **Existing Observation:**
  > *"The image shows two bright red, glossy fruits attached to long, thin green stems. A single broad green leaf with a prominent central vein and smaller branching veins extends to the right. The leaf has serrated edges and appears mostly intact, with no obvious spots or discoloration. The fruits and leaf are displayed against a plain white background."*
- **Visual Evidence:**
  The actual file is an outdoor photograph (`1200x900` JPEG) showing an arching, strap-like, bluish-green garlic leaf. Distributed linearly along the leaf blade are numerous small, raised, orange-to-reddish-brown pustules, many surrounded by subtle pale yellowish chlorotic halos. The background contains blurred green leaves and straw-like dried foliage. There are **no red fruits**, **no cherries**, and **no white background**.
- **Specific Concern:** Critical factual description error. The text is an accidental duplicate copied verbatim from `cherry leaf/183.jpg` (`obs-ecf09599c7f1`).
- **Proposed Objective Correction:**
  > *"The image shows an arching, strap-like bluish-green leaf with numerous small, raised orange to reddish-brown pustules scattered along the surface. Many pustules are surrounded by slight pale yellowish halos. The background shows out-of-focus green foliage."*
- **Observer Confirmation Required?** **YES.** Because this is a 100% replacement of the observation due to an annotator error, the original observer (`obs-ddc1912f882f`) or an authorized curator must verify and sign off on the replacement text.

---

### Case 2: `cherry leaf/183.jpg` (Alias: `obs-ecf09599c7f1`)
- **Observer ID:** `obs-ddc1912f882f` (Date: `2026-10-20`)
- **Existing Observation:**
  > *"The image shows two bright red, glossy fruits attached to long, thin green stems. A single broad green leaf with a prominent central vein and smaller branching veins extends to the right. The leaf has serrated edges and appears mostly intact, with no obvious spots or discoloration. The fruits and leaf are displayed against a plain white background."*
- **Visual Evidence:**
  The image (`320x320` JPEG) shows exactly two bright red heart-shaped cherries attached to long thin green stems, with a single serrated green leaf extending to the right, against an isolated plain white background.
- **Specific Concern:** Flagged only because its text was identical to Case 1 (`garlic rust`). The text is completely factual and visually accurate for this image.
- **Proposed Correction:**
  > Keep the existing text unchanged. Clear the duplicate flag once Case 1 is corrected.
- **Observer Confirmation Required?** **NO.** The existing text accurately reflects the image.

---

### Case 3: `banana leaf/google_0056.jpg` (Alias: `obs-8ff560e519df`)
- **Observer ID:** `reviewer_01` (Date: `2026-10-19`)
- **Existing Observation:**
  > *"The image shows a large, elongated green leaf with prominent parallel veins running along its length. The leaf is folded and curved, with a smooth surface and mostly uniform dark green coloring. A thin brownish line is visible along part of the edge, with a few small marks on the surface. No large spots, holes, or extensive discolor"*
- **Visual Evidence:**
  A curved, rolled dark green banana leaf (`512x288` JPEG) with prominent parallel lateral veins, smooth texture, and a narrow brown line along the curled margin.
- **Specific Concern:** Truncation. The sentence ends mid-word on `"discolor"` without terminal punctuation.
- **Proposed Objective Correction:**
  > *"The image shows a large, elongated green leaf with prominent parallel veins running along its length. The leaf is folded and curved, with a smooth surface and mostly uniform dark green coloring. A thin brownish line is visible along part of the edge, with a few small marks on the surface. No large spots, holes, or extensive discoloration."*
- **Observer Confirmation Required?** **NO.** Straightforward typographical completion of the unfinished word.

---

### Case 4: `basil leaf/284.jpg` (Alias: `obs-e94bddfb581f`)
- **Observer ID:** `obs-ddc1912f882f` (Date: `2026-10-20`)
- **Existing Observation:**
  > *"The image shows a green plant growing in a brown pot, with several upright stems and numerous pointed green leaves. Small clusters of light green buds or flowers are visible near the tops of the stems. Most leaves appear green and intact, with no obvious large brown lesions or widespread wilting. Several"*
- **Visual Evidence:**
  A potted basil plant (`350x350` JPEG) in a round brown plastic pot on weed-mat fabric, featuring upright stems, intact green pointed leaves, and budding terminal inflorescences. Other potted plants are visible in the background.
- **Specific Concern:** Truncation. Ends abruptly on the dangling word `"Several"`.
- **Proposed Objective Correction:**
  > *"The image shows a green plant growing in a brown pot, with several upright stems and numerous pointed green leaves. Small clusters of light green buds or flowers are visible near the tops of the stems. Most leaves appear green and intact, with no obvious large brown lesions or widespread wilting. Several other potted plants are visible in the background."*
- **Observer Confirmation Required?** **RESOLVED & CONFIRMED.** The human curator confirmed completion of the intended thought describing the background pots based on visual evidence.

---

### Case 5: `bean leaf/111.jpg` (Alias: `obs-5d1ef16e14c1`)
- **Observer ID:** `observer_01` (Date: `2026-10-18`)
- **Existing Observation:**
  > *"The image shows rows of green plants growing in cultivated soil. The plants have dense foliage with leaves of varying sizes. Some leaves appear lighter green than others, but no clear signs of severe damage or distinct disease symptoms are visible in this wide view."*
- **Visual Evidence:**
  Field plot (`475x284` JPEG) with rows of leafy green bean plants supported by bamboo/wood stakes in tilled soil.
- **Specific Concern:** Prohibited diagnostic wording. Section 1 of `ANNOTATION_GUIDELINES.md` states: *"Never Guess or Assert a Disease Diagnosis. Prohibited: Stating diagnosis names... or asserting disease."* The phrase `"distinct disease symptoms"` asserts a diagnostic category.
- **Proposed Objective Correction:**
  > *"The image shows rows of green plants growing in cultivated soil. The plants have dense foliage with leaves of varying sizes. Some leaves appear lighter green than others, but no clear signs of severe damage, necrotic lesions, or localized foliar symptoms are visible in this wide view."*
- **Observer Confirmation Required?** **NO.** Minor objective vocabulary adjustment strictly adhering to protocol guidelines.

---

### Case 6: `strawberry leaf scorch/google_0003.jpg` (Alias: `obs-4cb74f6bdb36`)
- **Observer ID:** `observer_01` (Date: `2026-10-18`)
- **Existing Observation:**
  > *"The leaves show numerous dark purple, reddish-brown, and black spots scattered across their surfaces. Several spots have merged into larger irregular patches, with yellowing and reddish discoloration around them. Some leaf edges are dry, curled, and brown, while other areas remain green. The damage varies in size and severity across the leaves."*
- **Visual Evidence:**
  A 4-panel composite (`1200x1200` JPEG) displaying strawberry leaves with small dark purple spots, coalesced dark necrotic blotches, and yellow/red chlorosis.
- **Specific Concern:** Subjective diagnostic claim. The word `"severity"` is flagged under `DISEASE_CLAIM_TERMS`. Guidelines require objective physical descriptions rather than clinical severity grading.
- **Proposed Objective Correction:**
  > *"The leaves show numerous dark purple, reddish-brown, and black spots scattered across their surfaces. Several spots have merged into larger irregular patches, with yellowing and reddish discoloration around them. Some leaf edges are dry, curled, and brown, while other areas remain green. The lesions vary in size, density, and spatial distribution across the leaves."*
- **Observer Confirmation Required?** **NO.** Direct adherence to visual terminology guidelines.

---

### Case 7: `coffee leaf/224.jpg` (Alias: `obs-3f7693b4faa5`)
- **Observer ID:** `observer_01` (Date: `2026-10-18`)
- **Existing Observation:**
  > *"The image shows clusters of small, round fruits in different stages of ripeness, ranging from green to red. The surrounding leaves are mostly green, but several have yellowish patches, brown spots, and irregular damaged areas. Some leaf edges appear dry or torn, while other leaves look relatively healthy."*
- **Visual Evidence:**
  Branch with green, yellow, and red coffee berries (`234x404` JPEG) surrounded by glossy leaves; some have torn margins, while others are intact.
- **Specific Concern:** Subjective vocabulary. Guidelines advise avoiding value-laden adjectives like `"healthy"` or `"sick"`.
- **Proposed Objective Correction:**
  > *"The image shows clusters of small, round fruits in different stages of ripeness, ranging from green to red. The surrounding leaves are mostly green, but several have yellowish patches, brown spots, and irregular damaged areas. Some leaf edges appear dry or torn, while other leaves remain intact and uniformly green."*
- **Observer Confirmation Required?** **NO.** Vocabulary standardization.

---

### Case 8: `cherry leaf/google_0335.jpg` (Alias: `obs-68092719c527`)
- **Observer ID:** `observer_01` (Date: `2026-10-18`)
- **Existing Observation:**
  > *"The plant has several healthy-looking green leaves with clearly visible veins and serrated edges. Small round fruits are visible, including red ripe fruits and green unripe fruits. A small white flower with yellow parts is present near the lower center. No obvious leaf spots or major damage are visible."*
- **Visual Evidence:**
  Hanging branch with serrated green leaves (`1200x1500` JPEG), small round red and green berries, and a single white five-petaled flower with yellow stamens.
- **Specific Concern:** Subjective vocabulary (`"healthy-looking"`).
- **Proposed Objective Correction:**
  > *"The plant has several intact, uniformly green leaves with clearly visible veins and serrated edges. Small round fruits are visible, including red ripe fruits and green unripe fruits. A small white flower with yellow parts is present near the lower center. No obvious leaf spots or major damage are visible."*
- **Observer Confirmation Required?** **NO.** Vocabulary standardization.

---

### Case 9: `celery leaf/google_0064.jpg` (Alias: `obs-6b72023b5de6`)
- **Observer ID:** `observer_01` (Date: `2026-10-18`)
- **Existing Observation:**
  > *"The image shows a bunch of fresh green leafy stems with finely divided, irregularly shaped leaves. The leaves are mostly bright green and appear dense and healthy. Some stem ends have slight brown discoloration, but no obvious spots, holes, or widespread leaf damage are visible."*
- **Visual Evidence:**
  Celery bunch with upright ribbed stalks and dense divided green foliage (`1932x2560` JPEG) on white cloth.
- **Specific Concern:** Subjective vocabulary (`"dense and healthy"`).
- **Proposed Objective Correction:**
  > *"The image shows a bunch of fresh green leafy stems with finely divided, irregularly shaped leaves. The leaves are mostly bright green, dense, and intact. Some stem ends have slight brown discoloration, but no obvious spots, holes, or widespread leaf damage are visible."*
- **Observer Confirmation Required?** **NO.** Vocabulary standardization.

---

## 2. Observer Confirmation Summary

- **Records requiring original observer confirmation (2):**
  1. `garlic rust/google_0003.jpg` (Observer `obs-ddc1912f882f`) — Complete re-annotation of leaf symptoms.
  2. `basil leaf/284.jpg` (Observer `obs-ddc1912f882f`) — Decision on dangling word `"Several"`.
- **Records ready for human reviewer sign-off without re-annotation (7):**
  - Cases 2, 3, 5, 6, 7, 8, 9.
