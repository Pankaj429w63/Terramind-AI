# TerraMind AI — Pilot Annotation Machine-Assisted Quality Review Report

**Date:** 2026-10-20  
**Scope:** 178 class-balanced pilot plant images (89 classes × 2 images per class; 89 train, 89 validation, 0 test)  
**Input Workbook:** `outputs/annotation/pilot/annotator_workbook.csv`  
**Quality Review Report CSV:** `outputs/annotation/pilot/quality_review_report.csv`  
**Review Ledger:** `outputs/annotation/pilot/review_ledger.csv`

---

## 1. Executive Summary

| Metric | Count | Notes |
| :--- | :--- | :--- |
| **Total Records Evaluated** | 178 | All 178 sealed pilot aliases verified |
| **Passed Automated Checks** | 169 | Structural, lexical, and file integrity verified |
| **Records Needing Human Review** | 9 | Identified with actionable recommendations below |
| **Missing Image Files** | 0 | All 178 images present in `data/plantwild/plantwild/images/` |
| **Corrupted Image Files** | 0 | All 178 images pass `PIL.Image.verify()` |
| **Duplicate Aliases in Workbook** | 0 | Strict 1-to-1 alias uniqueness maintained |
| **Label Leakage (Exact Class Names)** | 0 | Strict class-label leakage validator passed |
| **Independent Review Status** | 178 Pending | No approvals fabricated; awaiting human reviewer |

---

## 2. Categorization of Quality Issues

The 9 flagged records fall into 4 clear categories:

### Category A: Critical Visual Mismatch / Duplication (1 record)
- **`garlic rust/google_0003.jpg` (Alias: `obs-ed667e8194fd`, Observer: `obs-ddc1912f882f`)**
  - **Issue:** Text describes *"two bright red, glossy fruits attached to long, thin green stems... displayed against a plain white background."*
  - **Reality:** Image is an outdoor garlic plant foliage photograph (`1200x900` JPEG, mean RGB ~100/137/108), not two cherries on white background.
  - **Cause:** Text was accidentally duplicated from `cherry leaf/183.jpg` (`obs-ecf09599c7f1`).
  - **Action Required:** The human reviewer must reject this observation or have an annotator re-describe the actual garlic image.

### Category B: Truncated Observations (2 records)
- **`banana leaf/google_0056.jpg` (Alias: `obs-8ff560e519df`, Observer: `reviewer_01`)**
  - **Issue:** Text ends abruptly: `"...No large spots, holes, or extensive discolor"` (missing ending word and terminal punctuation).
  - **Action Required:** Complete truncated text (e.g., change `"discolor"` to `"discoloration."`).
- **`basil leaf/284.jpg` (Alias: `obs-e94bddfb581f`, Observer: `obs-ddc1912f882f`)**
  - **Issue:** Text ends abruptly with dangling word: `"...Most leaves appear green and intact, with no obvious large brown lesions or widespread wilting. Several"`
  - **Action Required:** Remove dangling word `"Several"` or complete the sentence.

### Category C: Prohibited Diagnostic / Disease Claims (2 records)
Per `ANNOTATION_GUIDELINES.md` §1 (Core Principles: *"Never Guess or Assert a Disease Diagnosis"*):
- **`bean leaf/111.jpg` (Alias: `obs-5d1ef16e14c1`, Observer: `observer_01`)**
  - **Issue:** Contains prohibited word `"disease"`: `"...no clear signs of severe damage or distinct disease symptoms are visible in this wide view."`
  - **Action Required:** Rephrase objectively based on visible features without using the word `"disease"`.
- **`strawberry leaf scorch/google_0003.jpg` (Alias: `obs-4cb74f6bdb36`, Observer: `observer_01`)**
  - **Issue:** Contains claim term `"severity"`: `"...The damage varies in size and severity across the leaves."`
  - **Action Required:** Replace subjective assertion of severity with objective morphological description (e.g., lesion diameter or leaf area affected).

### Category D: Subjective Non-Objective Vocabulary (3 records)
Per `ANNOTATION_GUIDELINES.md` §1 & §2 (Vocabulary: *"Avoid vague adjectives like 'sick', 'bad', 'unhealthy'"*):
- **`coffee leaf/224.jpg` (Alias: `obs-3f7693b4faa5`, Observer: `observer_01`)**
  - Text contains: `"...while others appear healthy-looking."`
  - **Action Required:** Replace with objective visual terminology (e.g. `"intact green leaves"`).
- **`cherry leaf/google_0335.jpg` (Alias: `obs-68092719c527`, Observer: `observer_01`)**
  - Text contains: `"The plant has several healthy-looking green leaves..."`
  - **Action Required:** Replace with objective visual terminology (e.g. `"intact green foliage"`).
- **`celery leaf/google_0064.jpg` (Alias: `obs-6b72023b5de6`, Observer: `observer_01`)**
  - Text contains: `"...The leaves appear healthy and intact..."`
  - **Action Required:** Replace with `"intact green leaves"`.

### Category E: Cross-Referenced Duplicate Source (1 record)
- **`cherry leaf/183.jpg` (Alias: `obs-ecf09599c7f1`, Observer: `obs-ddc1912f882f`)**
  - Visually corresponds to the cherry image, but shares identical text with `garlic rust/google_0003.jpg`. Confirm authenticity during review.

---

## 3. Human Review Action Ledger

All original text in `annotator_workbook.csv` is strictly preserved. In accordance with research integrity standards:
1. No row has been marked `approved` automatically.
2. The authorized independent reviewer must inspect these 9 flagged items and record their final adjudication in `outputs/annotation/pilot/review_ledger.csv`.
