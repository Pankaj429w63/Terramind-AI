# TerraMind AI — Stage 2D: User-Style Symptom Observation Collection & Readiness

**Date:** 2026-10-17
**Status:** **GO (collection tooling) / BLOCKED (dataset)** — no genuine observations exist, so the output dataset was deliberately left ungenerated and no model was trained.
**Scope:** finalize the `user_observation` collection path for the three-way fusion experiment. Official splits, the eligibility manifest, the production checkpoint, and historical metrics were not modified.

## 1. What was inspected

- `outputs/research/fusion_eligibility_manifest.json` — eligible population and per-split/per-class counts.
- `outputs/annotation/annotator_workbook.csv` (17,870 alias-only rows) and `outputs/annotation/sealed/alias_map.json` (sealed).
- `data/annotation/annotator_workbook_template.csv`, `observations_template.csv`, `review_ledger_template.csv`.
- `ml/fusion/annotation.py`, `ml/fusion/data.py`, `scripts/build_annotation_workbook.py`, `scripts/validate_annotations.py`.
- `outputs/research/annotation_specification.md`, `text_modality_decision.md`, `symptom_collection_guide.md`.

## 2. Exact required schema and commands (from the repository — unchanged)

**Annotator workbook:** `alias, observation, observer_id, observed_at`
**Training CSV (six columns, unchanged):** `sample_id, image_path, text, text_origin, provenance, source_reference`
**Review ledger:** `sample_id, review_status, reviewer, reviewed_at, notes` with `review_status ∈ {pending, approved, rejected, disputed}`

These are the constants `WORKBOOK_COLUMNS`, `OBSERVATION_COLUMNS`, and `REVIEW_COLUMNS` in `ml/fusion/annotation.py`, and the `required` set in `load_paired_text_csv`. No field was invented or renamed.

```powershell
# 1. Blinded workbook + sealed alias map (already generated; idempotent)
python -m scripts.build_annotation_workbook

# 2. Progress check (no emission)
python -m scripts.validate_annotations --workbook outputs/annotation/annotator_workbook.csv `
    --alias-map outputs/annotation/sealed/alias_map.json

# 3. Emit ONLY when every row is approved
python -m scripts.validate_annotations --workbook <filled.csv> `
    --alias-map outputs/annotation/sealed/alias_map.json --review-ledger <ledger.csv> `
    --text-origin user_observation --instrument-version "<instrument version>" `
    --provenance "<method + participant pool + dates + instrument version>" `
    --source-reference <stable-uri> --emit-observations data/paired_observations.csv

# 4. Structural audit (must not report BLOCKED)
python scripts/audit_fusion_data.py --paired-csv data/paired_observations.csv
```

## 3. Blinding verification (task 4)

- The delivered workbook header is `alias,observation,observer_id,observed_at`; the generated file has **17,870 rows and 0 path/label tokens** (grep for `\.jpg|/|apple|tomato|scab|rust|blight` → **0 matches**).
- `load_annotator_workbook` raises `Annotation workbook must not expose labels, paths, or splits` if any of `image_path`, `path`, `relative_path`, `sample_id`, `class_id`, `class_name`, `label`, `target`, `target_label`, `split`, `official_split`, `disease` appears (covered by `test_workbook_rejects_label_or_path_exposure`).
- Annotators only ever receive aliases; the alias → path/split mapping stays in `outputs/annotation/sealed/alias_map.json`, which is never handed to annotators.
- Class names are also blocked *inside* the text: `validate_text_not_label_derived` rejects any PlantWild class phrase, so a leak in the observation itself is caught at collation.

## 4. Genuine-observation, provenance, instrument, and review safeguards (task 5)

- **Genuine:** collation copies the annotator's text **verbatim**; it only skips empty rows, never generates or rewrites text, and requires a non-empty `observer_id` for any row carrying an observation.
- **Provenance:** `validate_provenance` requires ≥12 non-placeholder characters and a URL/stable-URI `source_reference`; both are re-checked by the training loader.
- **Instrument version:** now enforced. `require_instrument_version` (new) rejects a missing version and any provenance that does not record it; the validator requires `--instrument-version` when emitting and records it in the report (covered by `test_instrument_version_must_be_recorded_in_provenance`).
- **Reviewer decisions:** the ledger captures `reviewer`, `reviewed_at`, and `notes`; `require_reviewed` blocks emission for any row that is missing a ledger entry or is not `approved` (covered by `test_review_gate_blocks_unapproved_or_unobserved`; duplicate IDs and unknown statuses rejected by `test_review_ledger_rejects_duplicates_and_unknown_status`).

No unpublished-data path was populated: templates remain header-only, and no row was marked approved by tooling.

## 5. Actual observations (task 7) — none

- `data/paired_observations.csv` does not exist (repo-wide search: no match).
- The generated workbook contains **0 filled rows**; there is no review ledger.
- Validator run (real artifacts) reported: `workbook_rows: 17870, aliases_defined: 17870, eligible_samples: 17870, filled_rows: 0, emitted: false, decision: INCOMPLETE`.
- Therefore the output dataset was **left ungenerated** and nothing was emitted. No invented, generated, or copied descriptions were inserted, and no unreviewed row was marked approved.

Consequence for task 8: there are no rows to accept or reject. The loader and validator paths are exercised by the test suite against a real on-disk dataset fixture (`test_emitted_observations_pass_the_training_loader`); the loader fails fast on the first invalid row and its message names the file, line, and reason, which is the rejection report the curator sees when data arrives.

## 6. Coverage (task 9) — target vs. actual

Eligible coverage is unchanged; **actual observation coverage is 0 everywhere**.

| Split | Eligible samples | Classes | Min/class | Median/class | Observed (annotated) |
|---|---:|---:|---:|---:|---:|
| train | 12,573 | 89 | 32 | 139 | **0** |
| validation | 1,751 | 89 | 4 | 19 | **0** |
| test | 3,546 | 89 | 8 | 40 | **0** |
| **total** | **17,870** | 89 | — | — | **0** |

Every class has ≥4 eligible images in every split, so full class coverage is reachable without touching official splits. Labels and split membership were never exposed to annotators (Section 3).

## 7. Verification (actual results)

| Check | Command | Result |
|---|---|---|
| Focused annotation + fusion-data tests | `python -m unittest discover -s tests -p test_multimodal.py -v` | **26 tests, OK** |
| Validator (real artifacts) | `python -m scripts.validate_annotations --workbook outputs/annotation/annotator_workbook.csv --alias-map outputs/annotation/sealed/alias_map.json` | **exit 0**, `INCOMPLETE`, nothing emitted |
| Workbook blinding | grep `\.jpg|/|apple|tomato|scab|rust|blight` over the workbook | **0 matches** |
| Compilation | `python -m py_compile` on annotation/data/text_only/eligibility + scripts + tests | **exit 0** |
| Whitespace check | `git diff --check` | **exit 0** |

## 8. GO / BLOCKED

- **GO** for collection: guide, blinded template, sealed mapping, review gate, instrument-version enforcement, and validation commands are in place and tested.
- **BLOCKED** for the dataset and for training: zero genuine observations exist.

**Decision: BLOCKED.**

**Minimum next action:** run a real collection session — contributors fill the blinded workbook per `symptom_collection_guide.md`, the curator compiles the review ledger with independent approval, then emit and validate:

```powershell
python -m scripts.validate_annotations --workbook <filled.csv> --alias-map outputs/annotation/sealed/alias_map.json `
    --review-ledger <ledger.csv> --text-origin user_observation --instrument-version "<vX>" `
    --provenance "<method + participant pool + dates + instrument vX>" `
    --source-reference <stable-uri> --emit-observations data/paired_observations.csv
python scripts/audit_fusion_data.py --paired-csv data/paired_observations.csv
```

Training becomes permissible only when that emission succeeds and the audit does not report `BLOCKED`.

### Changed files (this stage)
- `ml/fusion/annotation.py` — added `require_instrument_version`
- `scripts/validate_annotations.py` — `--instrument-version` required for emission, recorded in the report, default `--text-origin user_observation`
- `tests/test_multimodal.py` — added `test_instrument_version_must_be_recorded_in_provenance` (26 tests total)
- `outputs/research/symptom_collection_guide.md` — new practical collection guide
- `outputs/research/symptom_collection_readiness_stage2d.md` — this report
- `outputs/research/annotation_specification.md`, `outputs/research/text_modality_decision.md` — validation command updated to include `--instrument-version`/`--text-origin`

No changes to official splits, the eligibility manifest, the production checkpoint, or historical metrics.
