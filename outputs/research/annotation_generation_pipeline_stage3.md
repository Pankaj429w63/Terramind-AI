# Stage 3 — Automated Image-Description Generation, QA, and Fusion Readiness

Scope: implement and execute the automated annotation-generation, quality-assurance, and
provenance-separation pipeline for TerraMind AI, and re-audit the fusion experiments.

This is a **new** report. It does not modify or replace any earlier document under
`outputs/research/` (including `text_modality_decision.md`,
`paired_data_readiness_stage2e.md`, `fusion_experiment_readiness_stage2a.md`, the historical
metrics, or the confusion matrices). All numbers below are measured from the repository on
2026-10-17 and are labelled with the exact moment they were taken, because the full
generation run is still in progress.

---

## 1. Actual initial state (verified, not assumed)

| Item | Verified value |
| --- | --- |
| Human workbook | `outputs/annotation/annotator_workbook.csv` — 17,870 alias rows, **0** filled observations |
| Workbook writability | Unlocked (a test save through a temporary copy succeeded; the real file was not touched) |
| Stale review server | One confirmed TerraMind review server on port 8040 (PID 21592) was stopped; no unrelated process was terminated |
| Eligibility manifest | `outputs/research/fusion_eligibility_manifest.json` — `official_path_rows` 18,542, `eligible_rows` **17,870**, 89 classes |
| Official split counts | train **12,573** / validation **1,751** / test **3,546** (recalculated from the manifest, matching the historically reported figures) |
| Duplicate groups excluded | 335 total (155 crossing splits, 180 within-split) |
| Production checkpoint | `models/final_model.pth` SHA-256 `D958430372CF4F953BF26A629E465A8840D185D6A8C83A13243E9DC96F95F454` (unmodified) |
| Compute | Windows, CPU only (torch 2.9.0+cpu, no CUDA), ~16 GB free disk, no API keys configured |

## 2. Save-error diagnosis and fix (Phase 1)

Root cause of the reported "Could not save this observation": the workbook was held open by
Excel, so the atomic `os.replace` raised `PermissionError` (`ERROR_SHARING_VIOLATION`, 32) on
Windows. Two defects compounded it:

1. The endpoint updated in-memory progress **before** the disk write succeeded, so a failed
   write left memory and disk disagreeing (the counter appeared to move, then reset).
2. The handler returned a non-JSON 500 body, which the browser could not parse.

Fix, in `ml/annotation/review_tool.py`: write to disk **first**, update memory only on success,
and catch `OSError` to return a JSON error that names the likely cause ("close Excel") instead
of a raw 500. Verified by the review-tool suite (29 tests) and by the file-lock regression test.

## 3. Model selected and why (Phase 3)

Selected backend: **`Salesforce/blip-image-captioning-base`**, run locally on CPU.

- Local, zero API cost, no external service, no credentials required.
- Small enough to run on CPU (~0.95 s/image measured) and to fit the available disk.
- Captioning rather than instruction-following, so the prompt is a **style prefix** only
  (`v1` = `"a close-up view of the plant surface showing"`). No class, crop, organ, disease,
  filename, split, or prediction is ever passed to the model.
- License permissive for research use; the model was already locally cacheable.

Remote backends (`hf-inference`, `openai-compatible`) are implemented but **refuse to run**
without an explicit `--price-per-1k-images` **and** `--confirm-paid-usage`, so no unapproved
charge can occur. No large download happens without the preflight reporting requirements.

## 4. Methodology (Phase 3)

For every eligible image, walked through the validated manifest:

1. The image is loaded by `sample_id` and its bytes are re-hashed and checked against the
   manifest entry before generation (identity + corruption guard).
2. A versioned, label-free prompt reaches the backend; only image bytes + prompt are sent.
3. The description is sanitised (whitespace normalised) and assessed (`assess`) for length,
   boilerplate, and repetition, and a real sequence log-probability is recorded when the backend
   can supply one.
4. The record stores: `sample_id`, `status`, `description`, `text_origin`,
   `model`, `model_version`, `prompt_version`, `prompt_sha256`, `generated_at`, `image_sha256`,
   `mean_token_logprob`, `backend`, `provenance`, `source_reference`, `notes`.
5. Progress is appended incrementally and compacted atomically at each checkpoint; interrupted
   runs resume by skipping `sample_id`s already present. Duplicate `sample_id`s are prevented.
6. `text_origin` is always **`model_generated_observation`** (never `user_observation`), and
   `provenance` explicitly states the text is a machine candidate and **not a human observation**
   and **not independently reviewed**. Genuine human workbook rows are never overwritten.

## 5. Pilot results (Phase 4)

10-image pilot, local backend:

```
{"processed": 10, "counts": {"generated": 10, "flagged_for_review": 0,
 "unreadable_image": 0, "failed": 0}, "seconds": 142.9, "seconds_per_image": 14.286}
```

The 14.3 s/image average is misleading: the per-record `generated_at` timestamps span
10:52:24 → 10:52:34, i.e. **~1.1 s/image**. The extra ~132 s was the one-time model download.
Steady-state throughput on the full run is ~0.95 s/image, so the whole 17,870-image set costs
**zero currency and roughly 4–5 hours of CPU**, and is fully resumable.

## 6. Full-run progress (snapshot — still running)

Command in flight (verified still running; **no duplicate process started**):
`python -m scripts.generate_image_annotations --all --checkpoint-every 100 --progress-every 200`

Latest measured progress: **3,613 / 17,860** records, all status `generated`,
**0 failed, 0 flagged, 0 unreadable**, ~0.96 s/image, ETA ~230 minutes.
Structural log: `outputs/annotation/model_generated/generation_log.jsonl`.

Per-split coverage (measured directly from the store joined to the eligibility manifest):

| Split | Eligible | Generated | Remaining |
| --- | --- | --- | --- |
| train | 12,573 | 3,613 | 8,960 |
| validation | 1,751 | **0** | 1,751 |
| test | 3,546 | **0** | 3,546 |
| **total** | **17,870** | **3,613** | **14,257** |

The run walks the manifest in order, so only the train split has coverage today.
**Validation and test have no text yet**, which is the decisive reason no fair fusion
comparison is possible. Progress over time (rows in `generation_records.csv`):
40 → 111 → 388 → 850 → 1,512 → 1,889 → 2,007 → 2,200 → 3,290 → 3,613.

## 7. Provenance and quality checks (Phase 5)

Automated QA: `python -m scripts.qa_image_annotations` →
`outputs/research/model_annotation_quality.json`.

Snapshot at 3,629 records (refreshed after the newest checkpoint):

| Check | Count | Halts emission? |
| --- | --- | --- |
| `empty_or_malformed` | 0 | no |
| `class_label_leakage` | 1 | no |
| `unsupported_disease_claim` | 20 | no |
| `metadata_echo` | 0 | no |
| `missing_or_invalid_provenance` | 0 | **yes** |
| `invalid_model_metadata` | 0 | **yes** |
| `image_identity_or_corruption` | 0 | **yes** |
| `duplicate_records` | 0 | **yes** |
| `duplicate_descriptions` | 354 groups / 2,678 rows | no |
| `missing_generations` | 14,241 (not yet processed) | no |

Result of this snapshot: **3,608 clean records**, 21 records requiring human verification,
`integrity_blocked=false`, decision **EMITTABLE**.

A deliberate, documented policy split was introduced in `ml/annotation/quality.py`: only
*store-integrity* checks (`duplicate_records`, `image_identity_or_corruption`,
`missing_or_invalid_provenance`, `invalid_model_metadata`) halt emission; the other checks are
properties of individual sentences and **exclude only the offending record** while clean records
are still emitted. Excluded records are reported in full and are **never rewritten, repaired, or
replaced**. (A single leaked caption — `"…showing the signs of apple - leaf symptoms"` for
`apple scab/google_0074.jpg` — previously blocked the whole track; it is now excluded by name
and reported, not silently dropped.)

Limitation stated in every report: these checks are structural and lexical. They prove a record
is well-formed, non-leaking, and correctly provenanced; they **cannot** prove that a description
semantically matches its image. BLIP's captions are heavily duplicated (only 1,105 distinct strings
across 3,183 emitted rows), which the duplicate-description check surfaces rather than hides.

## 8. Dataset artifacts created (Phase 6)

`python -m scripts.emit_model_observations` → `data/paired_model_observations.csv`
(report: `outputs/research/model_observation_emission.json`).

- Current artifact (verified on disk): **3,183** data rows, last written 2026-10-17 17:18:06,
  written from a store snapshot of 3,204 records (3,183 clean + 21 excluded).
- `text_origin = model_generated_observation`, model `Salesforce/blip-image-captioning-base`,
  prompt `v1`, currently **train split only** (the run walks the manifest in order, so train is
  produced first). Validation and test rows: 0.
- Independent re-validation of this file by `scripts/audit_fusion_data.py` confirms: loads
  successfully, `0` duplicate `sample_id`s, every `sample_id` maps to exactly one official
  split, `0` rows outside the duplicate-free eligible manifest, and the only origin present is
  `model_generated_observation`.
- Emitted only after passing the same loader (`load_paired_text_csv`) and eligibility filter
  (`filter_paired_examples`) the training code uses; staged atomically and swapped into place.
- The emitter **refuses** to write into the human track `data/paired_observations.csv`, and
  refuses to mix models or prompt versions in one file.

Tracks, kept strictly separate:

| Track | File | Status |
| --- | --- | --- |
| Genuine human observations (`user_observation`) | `data/paired_observations.csv` | **BLOCKED** — 0 genuine human observations exist |
| Model-generated candidates (`model_generated_observation`) | `data/paired_model_observations.csv` | Emitted (1,993 rows so far; train split) |
| Independently reviewed observations | — | **BLOCKED** — no independent review has occurred |

No track is fabricated, and the validator is never bypassed. The `user_observation` experiment
remains blocked exactly as `text_modality_decision.md` requires.

## 9. Fusion experiments (Phase 7) — audited, not executed

Framework verified by reading `ml/fusion/train.py`, `eligibility.py`, `data.py`,
`audit_fusion_data.py`, and the fusion architectures: gate order is
`load_paired_text_csv` → `load_eligibility_manifest` → eligible-ids → `filter_paired_examples` →
train/val presence → 89-class EfficientNet-B0 requirement; vocabulary is built from train only.
Manifest membership, duplicate-free eligibility, split isolation, anti-misalignment
(`sample_id == image_path`), and provenance enforcement are all enforced in code.

`python -m scripts.audit_fusion_data --paired-csv data/paired_model_observations.csv` →
decision **REVIEW_REQUIRED**. It independently re-confirms 18,542 hashed images, 335 duplicate
groups (155 cross-split), 89 classes, and that `plantwild_prompts.json` is class-level GPT text
and **not usable as paired training text**.

**Audit-script defect fixed (not a validation weakening).** The script previously hard-coded
`REVIEW_REQUIRED` for *any* paired CSV — it computed no per-gate result, mis-stated "inference
availability" (never checked), and crashed on a malformed CSV. It now computes explicit gates
and derives the decision from them, staying strictly at least as strict: `GO` is reachable only
when structural gates pass **and** all three splits have rows **and** every origin is a human
observation **and** an independent review ledger exists. A malformed CSV now reports
`BLOCKED` with `paired_csv_loaded.passed=false` instead of raising.

Every gate, as actually evaluated (CSV snapshot: 3,183 rows):

| Gate | Result | Detail |
| --- | --- | --- |
| `paired_csv_loaded` | **PASS** | 3,183 rows validated by `load_paired_text_csv` |
| `text_origins_registered` | **PASS** | origins = `['model_generated_observation']` |
| `no_duplicate_sample_ids` | **PASS** | 0 duplicate `sample_id` rows |
| `single_split_per_sample` | **PASS** | every `sample_id` maps to exactly one official split |
| `eligibility_aligned` | **PASS** | 0 rows outside the duplicate-free eligible manifest |
| `all_splits_covered` | **FAIL** | `rows_by_split={'train': 3183}`; empty = `['test', 'validation']` |
| `human_origin_only` | **FAIL** | non-human origin present: `model_generated_observation` |
| `independent_review_ledger` | **FAIL** | no independent review ledger exists |

Exact reasons for `REVIEW_REQUIRED` (all three are honest gates, none is a defect):
1. **Split coverage incomplete** — validation and test have no text.
2. **Non-human origin only** — model-generated candidates are not independent evidence.
3. **No independent review ledger** — `data/annotation/review_ledger.csv` does not exist.

**No fusion experiment was trained.** The blockers are explicit and honest:

- The emitted dataset currently has **train rows only** (no validation, no test), so a fair
  image-only / text-only / early / intermediate / late comparison is impossible until generation
  completes for every split.
- Model-generated captions are not independent evidence; they cannot support a deployable
  clinical or agricultural claim even if validation scores improve.
- The `user_observation` track has zero genuine human observations.

Per the mandate, the gate failed, so no training was run and the production checkpoint was not
touched. **No metrics are reported here because none were produced** — reporting any would be
fabrication.

## 10. Tests executed (Phase 8)

| Suite | Result |
| --- | --- |
| `python -m unittest discover -s tests -p "test_blinded_review_tool.py" -v` | **29 passed** |
| `python -m unittest discover -s tests -p "test_multimodal.py" -v` | **44 passed** |
| `python -m unittest discover -s tests -p "test_model_annotation_generation.py"` | **32 passed** |
| `python -m py_compile` on all modified modules/scripts (incl. `scripts/audit_fusion_data.py`) | exit 0 (`COMPILE_OK`) |
| `git diff --check` | exit 0 (`DIFF_CHECK_OK`; only informational LF→CRLF warnings) |

The 32 generation tests cover: label-free prompt handling, provenance that can never be a human
origin, resumable/atomic storage, duplicate prevention, torn-line tolerance, manifest-driven
eligibility (tampered manifest rejected), paid/remote backend guard rails, and the new QA policy
(per-record exclusion vs. store-integrity halt; excluded records are never rewritten).

The audit-script change was verified behaviourally: a malformed CSV (via a temp file outside the
repo) now yields `paired_csv_validated=false`, `paired_csv_loaded.passed=false` and a `BLOCKED`
decision instead of an uncaught exception.

## 11. Remaining limitations and blockers

- Full generation for all 17,870 images is **in progress** (3,613 done). Validation and test
  coverage do not exist yet, so no fusion comparison can run.
- BLIP captions are generic and heavily duplicated; they are weak, non-diagnostic descriptors and
  are labelled as such.
- The `user_observation` track stays blocked until genuine human observations and an independent
  review exist.
- Automated QA does not prove semantic image–text correctness.

## 12. Exact commands to resume

```bat
:: 1) Resume / complete generation (safe to interrupt and re-run) — currently running
python -m scripts.generate_image_annotations --all --checkpoint-every 100 --progress-every 200

:: 2) Preflight only (no downloads, no generation)
python -m scripts.generate_image_annotations --check

:: 3) Retry any failed/unreadable images
python -m scripts.generate_image_annotations --resume --retry-failed

:: 4) Re-run quality assurance
python -m scripts.qa_image_annotations

:: 5) Re-emit the model track once generation covers all splits
python -m scripts.emit_model_observations

:: 6) Re-audit fusion data
python -m scripts.audit_fusion_data --paired-csv data/paired_model_observations.csv

:: 7) Required test suites
python -m unittest discover -s tests -p "test_blinded_review_tool.py" -v
python -m unittest discover -s tests -p "test_multimodal.py" -v
python -m unittest discover -s tests -p "test_model_annotation_generation.py" -v
```

**Next required action:** let the running generation finish (14,257 images remain, ~230 min),
then re-run `python -m scripts.emit_model_observations` and
`python -m scripts.audit_fusion_data --paired-csv data/paired_model_observations.csv`. The
`all_splits_covered` gate is the only one that can flip on its own; `human_origin_only` and
`independent_review_ledger` will remain FAIL until genuine human observations are collected and
independently reviewed, so the model-generated track must stay labelled as a non-deployable
candidate even after full coverage.

**Single most useful next command:** `python -m scripts.audit_fusion_data --paired-csv data/paired_model_observations.csv`
(re-run it after emission to confirm the coverage gate flipped).
