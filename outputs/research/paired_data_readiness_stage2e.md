# TerraMind AI — Stage 2E: Paired-Data Readiness Verification

**Date:** 2026-10-17
**Decision:** **BLOCKED** — no genuine observations exist; the approved-only validator cannot emit and the paired-data loader has no file to read. No models were trained; official splits, the eligibility manifest, the production checkpoint, and historical metrics were not modified.

## 1. Actual annotation files inspected

| Path | State |
|---|---|
| `outputs/annotation/annotator_workbook.csv` | **17,870 rows, 0 filled** (all observations blank) |
| `outputs/annotation/sealed/alias_map.json` | present (sealed; 17,870 aliases) |
| Review ledger (filled) | **does not exist** — only `data/annotation/review_ledger_template.csv` (header only) |
| Filled workbook (any) | **does not exist** — only `data/annotation/annotator_workbook_template.csv` (header only) |
| `data/paired_observations.csv` | **does not exist** (repo-wide CSV search: no match) |

No observation was created, altered, or fabricated. The blinded workflow produced a workbook but no collection session has run.

## 2. Commands run and exact results

```powershell
# Approved-only validator (progress mode)
python -m scripts.validate_annotations --workbook outputs/annotation/annotator_workbook.csv `
    --alias-map outputs/annotation/sealed/alias_map.json
```
```
{ "workbook_rows": 17870, "aliases_defined": 17870, "eligible_samples": 17870,
  "emitted": false, "filled_rows": 0, "decision": "INCOMPLETE",
  "blocker": "Provide observations for every eligible alias and an approved review ledger, then re-run with --emit-observations." }
```

```powershell
python scripts/audit_fusion_data.py
```
```
{ "decision": "BLOCKED",
  "blocker": "No image-specific, independently sourced text-pair CSV is present.",
  "official_split_counts": { "train": 13045, "validation": 1820, "test": 3677 },
  "official_class_count": 89, "hashed_images": 18542,
  "exact_duplicate_image_groups": 335, "exact_duplicate_groups_crossing_splits": 155,
  "paired_csv_exists": false, "paired_csv_validated": false }
```

```powershell
# Paired-data loader against the expected emitted path
python -c "from pathlib import Path; from ml.fusion.data import load_paired_text_csv; load_paired_text_csv(Path('data/paired_observations.csv'), Path('data/plantwild/plantwild'), {}, {})"
```
```
FileNotFoundError: Paired text CSV not found: data\paired_observations.csv
```

Emission was not attempted with `--emit-observations`: there are 0 filled rows and no review ledger, so the approved-only gate cannot be satisfied. Running it would not produce data.

## 3. Gate-by-gate verification

| Gate | Status | Evidence |
|---|---|---|
| **Genuine observations present** | **FAIL** | 0 filled workbook rows; no filled workbook file |
| **Approved review status** | **NOT EVALUABLE** | no review ledger exists; `require_reviewed` blocks missing/non-`approved` rows (test `test_review_gate_blocks_unapproved_or_unobserved`) |
| **Provenance** | **NOT EVALUABLE** | no rows; `validate_provenance` requires ≥12 non-placeholder chars + URI `source_reference` |
| **Instrument version** | **NOT EVALUABLE** | enforced at emission by `require_instrument_version` (test `test_instrument_version_must_be_recorded_in_provenance`) |
| **Label leakage** | **NOT EVALUABLE** | no text; `validate_text_not_label_derived` rejects class phrases (tests `test_collation_copies_text_verbatim_and_rejects_leakage`) |
| **Sample identity (`sample_id` == normalized `image_path`)** | **NOT EVALUABLE** | enforced by `load_paired_text_csv`; proven by `test_emitted_observations_pass_the_training_loader` |
| **Duplicate-free eligibility** | **PASS (manifest level)** | `validate_eligibility_manifest` re-derived from untouched official files → **17,870 eligible**; 335 duplicate groups (155 crossing splits) excluded |
| **Class alignment** | **PASS (manifest level)** | 89 official classes; every class has eligible images in each split |
| **Train/validation/test coverage** | **FAIL (actual)** | eligible target 12,573 / 1,751 / 3,546 = 17,870; **observed annotations = 0** |

## 4. Coverage

| Split | Eligible samples | Classes | Min/class | Median/class | Observed |
|---|---:|---:|---:|---:|---:|
| train | 12,573 | 89 | 32 | 139 | **0** |
| validation | 1,751 | 89 | 4 | 19 | **0** |
| test | 3,546 | 89 | 8 | 40 | **0** |
| **total** | **17,870** | 89 | — | — | **0** |

Class and split were never exposed to annotators (workbook is alias-only; sealed map not distributed).

## 5. Exact failing gate

**Gate 1 — no genuine observations.** The workbook contains zero filled rows, so there is no review ledger and no emitted `data/paired_observations.csv`. Consequently the validator reports `INCOMPLETE` (emitted: false), the audit reports `decision: BLOCKED`, and the loader raises `FileNotFoundError: Paired text CSV not found: data\paired_observations.csv`. Every downstream gate (approval, provenance, leakage, sample identity, coverage) is unevaluable until Gate 1 passes.

## 6. Verification (actual results)

| Check | Command | Result |
|---|---|---|
| Approved-only validator | `python -m scripts.validate_annotations ...` | exit 0, `INCOMPLETE`, `emitted: false` |
| Data audit | `python scripts/audit_fusion_data.py` | exit 0, **`decision: BLOCKED`** |
| Paired-data loader | `load_paired_text_csv('data/paired_observations.csv', ...)` | **FileNotFoundError** (expected) |
| Eligibility re-validation | `validate_eligibility_manifest(...)` | **OK**, 17,870 eligible |
| Focused tests | `python -m unittest discover -s tests -p test_multimodal.py -v` | **26 tests, OK** |
| Compilation | `python -m py_compile` on fusion/data + scripts | **exit 0** |
| Whitespace check | `git diff --check` | **exit 0** |

Regenerated (identical content, not new facts): `outputs/research/fusion_data_audit.json`, `outputs/research/annotation_validation.json`.

## 7. Decision

**BLOCKED.** The pipeline is ready and verified, but there is no collected data, so no GO is possible and no report can claim training readiness.

**Exact remaining blocker:** run a real blinded collection session so `outputs/annotation/annotator_workbook.csv` gains filled rows with genuine `observer_id`s, compile an independent review ledger, then emit through the approved-only gate:

```powershell
python -m scripts.validate_annotations --workbook <filled.csv> `
    --alias-map outputs/annotation/sealed/alias_map.json --review-ledger <ledger.csv> `
    --text-origin user_observation --instrument-version "<vX>" `
    --provenance "<method + participant pool + dates + instrument vX>" `
    --source-reference <stable-uri> --emit-observations data/paired_observations.csv
python scripts/audit_fusion_data.py --paired-csv data/paired_observations.csv
```

Training becomes permissible only when emission succeeds and the audit no longer reports `BLOCKED`.

### Changed files (this stage)
- `outputs/research/paired_data_readiness_stage2e.md` — new (this report)
- Regenerated: `outputs/research/fusion_data_audit.json`, `outputs/research/annotation_validation.json`

No code, official splits, eligibility manifest, production checkpoint, or historical metrics were modified.

---

## 8. Recheck — "newly collected observations" (2026-10-17)

Re-run after a claim that observations and a completed review ledger now exist. **No such data exists on disk.**

| Expected artifact | Found |
|---|---|
| Filled annotator workbook | **absent** — only `data/annotation/annotator_workbook_template.csv` (header only) |
| Completed review ledger | **absent** — only `data/annotation/review_ledger_template.csv` (header only) |
| `data/paired_observations.csv` | **absent** (`paired_observations*` search: no file) |

Re-run results (identical to Section 2):

```powershell
python -m scripts.validate_annotations --workbook outputs/annotation/annotator_workbook.csv --alias-map outputs/annotation/sealed/alias_map.json
# → {"workbook_rows": 17870, "aliases_defined": 17870, "eligible_samples": 17870,
#     "emitted": false, "filled_rows": 0, "decision": "INCOMPLETE"}

python scripts/audit_fusion_data.py --paired-csv data/paired_observations.csv
# → {"decision": "BLOCKED", "blocker": "No image-specific, independently sourced text-pair CSV is present.",
#     "paired_csv_exists": false, "paired_csv_validated": false, ...}

python -c "... load_paired_text_csv(Path('data/paired_observations.csv'), ...)"
# → FileNotFoundError: Paired text CSV not found: data\paired_observations.csv
```

No `--instrument-version` could be supplied to an emission run because there is nothing to emit: 0 filled rows and no review ledger.

**Gate status: unchanged.** Gate 1 (genuine observations) still fails; approval, provenance, instrument version, leakage, sample identity, and observed coverage remain **not evaluable**. Duplicate-free eligibility and class alignment still **PASS** at the manifest level (17,870 eligible; 89 classes; 335 duplicate groups, 155 crossing splits).

| Recheck verification | Result |
|---|---|
| `python -m unittest discover -s tests -p test_multimodal.py -v` | **26 tests, OK** |
| `python -m py_compile` (fusion/data + scripts) | **exit 0** |
| `git diff --check` | **exit 0** (LF→CRLF warnings only) |

**Decision: still BLOCKED.** No data was fabricated and no metrics were invented. The exact reason remains: `data/paired_observations.csv` does not exist because the blinded workbook has zero filled rows and no review ledger has been compiled.
