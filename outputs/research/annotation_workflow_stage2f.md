# TerraMind AI — Stage 2F: Fix Annotation Validation and Windows Workflow End-to-End

Status: **BLOCKED — no genuine observations and no completed independent review ledger exist.**
No paired-data CSV was emitted. No model was trained. Official splits, the eligibility
manifest, the production EfficientNet-B0 checkpoint, and historical metrics were not touched.

---

## 1. Actual initial repository state (verified from disk)

Artifacts inspected:

- `scripts/validate_annotations.py`, `ml/fusion/annotation.py`, `ml/fusion/data.py`,
  `scripts/audit_fusion_data.py`, `scripts/build_annotation_workbook.py`
- `outputs/research/{symptom_collection_guide,annotation_specification,text_modality_decision}.md`
- `data/annotation/{annotator_workbook_template,observations_template,review_ledger_template}.csv`
- `outputs/annotation/annotator_workbook.csv`, `outputs/annotation/sealed/alias_map.json`
- `outputs/research/annotation_validation.json` (previous run)

Measured state:

| Item | Value |
| --- | --- |
| Workbook rows | **17,870** |
| Non-empty observations | **0** |
| Distinct observer IDs | **0** |
| Observation timestamps | none present (no observations to validate) |
| Completed review ledger (`data/annotation/review_ledger.csv`) | **does not exist** (only the header-only template exists) |
| Approved / rejected / disputed / pending | **0 / 0 / 0 / 0** (no ledger) |
| `data/paired_observations.csv` | **does not exist** |
| Partial output left by a failed validation | **none** — no `*.partial`, no `*.staged`, no `data/*.csv` |
| Sealed alias map | present, **never printed**; aliases are `obs-<sha256[:12]>`, opaque |

The previous `outputs/research/annotation_validation.json` recorded `workbook_rows=17870`,
`filled_rows=0`, `decision=INCOMPLETE`. It is a report from a successful read-only pass, not a
partial emission.

Conclusion: the two errors in the user's command were raised **before** any observation could
validate, and there is **no genuine, review-approved data** in the repository. Per the strict
stop conditions, collection must happen before anything can be emitted or trained.

---

## 2. Root causes of the two validation errors

The user ran the validator with **non-genuine placeholder metadata**
(`ACTUAL_POOL`, `ACTUAL_DATES`, `ACTUAL_SOURCE_REFERENCE`).

1. `ValueError: provenance must record the elicitation instrument version`
   Triggered because the provenance string never actually contained the instrument version, and
   the old check reported this with a message that did not explain *why* or *how to fix it*.
   The placeholder tokens were also silently accepted as if they were real collection metadata.

2. `ValueError: source_reference must be a URL or stable URI`
   Triggered because `ACTUAL_SOURCE_REFERENCE` is not a URI. The old message was vague: it did
   not distinguish a placeholder, a missing scheme, a local drive path, or a `file:`/`data:`
   scheme, so the user could not tell what was actually wrong.

Both errors are **correct in spirit** (the values are not valid evidence) but were **not
actionable** and did **not reject placeholders explicitly**. The fix makes them precise and
placeholders are now rejected outright. Validation was strengthened, not weakened.

---

## 3. Files changed

- `ml/fusion/data.py` — rewrote provenance/source-reference validation:
  - `PLACEHOLDER_VALUES` + `PLACEHOLDER_PATTERN` reject `ACTUAL_*`, `YOUR_*`, `YOUR_ACTUAL*`,
    `<...>`, `*_HERE`, `replace-me`, `changeme`, `unknown`, `n/a`, `none`, `tbd`, …
  - `validate_source_reference` requires a scheme; rejects local drive paths (`C:\…`),
    `file:`/`data:` schemes, malformed hosts, and URIs with no record identifier.
  - `validate_provenance` requires ≥ 12 chars, no placeholders, an ISO `YYYY-MM-DD` collection
    date, and a valid source reference.
- `ml/fusion/annotation.py`:
  - Added `validate_observed_at` (ISO `YYYY-MM-DD` timestamps).
  - Added `observation_issues` (per-row, actionable reasons that never echo text, labels, or
    image paths).
  - Tightened `require_instrument_version` to whole-token matching (so `v1` does not match `v10`).
  - Made `write_observations_csv` **atomic** (`NamedTemporaryFile` + `os.replace`; removes the
    temp file on any failure).
  - `load_annotator_workbook` now validates the timestamp of every filled row.
- `scripts/validate_annotations.py` — rewrote the CLI:
  - Decision states: `INCOMPLETE` → `BLOCKED` → `READY` → `EMITTED`.
  - Reports workbook/alias/eligible counts, per-row rejected reasons (aliases only), review
    counts by status, instrument version, and per-split validated rows.
  - Emission is **staged then atomically replaced**, and the staged file is re-loaded through the
    real training loader (`load_paired_text_csv`) before the destination is touched.
- `tests/test_multimodal.py` — updated 3 fixtures to include a collection date, aligned one
  assertion to the new message, and added `AnnotationValidationStage2FTests` (18 regression tests).

Unrelated working-tree changes (backend, frontend, rag, other tests, etc.) were **left untouched**.

---

## 4. Validation behavior: before vs after

| Gate | Before | After |
| --- | --- | --- |
| Provenance contains a date | not required | required (`YYYY-MM-DD`) |
| Placeholder provenance (`ACTUAL_POOL`, …) | accepted | **rejected** with a precise message |
| Instrument version | generic message, loose match | whole-token match; must appear in provenance |
| `source_reference` | generic "must be a URL" | placeholders, missing scheme, local drive paths (`C:\…`), `file:`/`data:`, malformed hosts, and id-less URIs all rejected with distinct messages |
| Per-row invalid observations | not itemized | itemized by alias with actionable reasons, no leakage |
| Timestamps | unvalidated | ISO `YYYY-MM-DD` enforced |
| Emission | could write directly | temp file + `os.replace`; pre-existing output preserved on failure |
| Outcome reporting | success/failure only | `INCOMPLETE` / `BLOCKED` / `READY` / `EMITTED` |

Verified new messages (real run):

```
REJECT: source_reference is still a placeholder ('ACTUAL_SOURCE_REFERENCE'); provide the real URL or stable URI of the annotation records
REJECT: provenance is still a placeholder ('ACTUAL_POOL; dates: ACTUAL_DATES; instrument v1'); replace ACTUAL_*/YOUR_*/<...> values with the real collection method, observer pool, collection dates, and instrument version
PASS
```

---

## 5. Exact single-line Windows CMD commands

All commands are single-line and copy-pasteable into **cmd.exe**. None contain PowerShell
backtick continuations, CMD caret continuations, or angle-bracket placeholders.

**1. Check workbook completeness**

```
python -c "import csv;rows=list(csv.DictReader(open('outputs/annotation/annotator_workbook.csv',newline='',encoding='utf-8-sig')));filled=sum(1 for r in rows if (r.get('observation') or '').strip());print('rows',len(rows),'filled',filled,'missing',len(rows)-filled)"
```

**2. Check the review ledger** (run only once `data/annotation/review_ledger.csv` exists — see §6)

```
python -c "import csv,collections;rows=list(csv.DictReader(open('data/annotation/review_ledger.csv',newline='',encoding='utf-8-sig')));print(dict(collections.Counter((r.get('review_status') or '').strip().lower() for r in rows)))"
```

**3. Run annotation validation (read-only; no emission)**

```
python -m scripts.validate_annotations --workbook "outputs/annotation/annotator_workbook.csv" --alias-map "outputs/annotation/sealed/alias_map.json" --output "outputs/research/annotation_validation.json"
```

**4. Run the paired-data audit**

```
python scripts/audit_fusion_data.py
```

**5. Run the focused tests**

```
python -m unittest discover -s tests -p test_multimodal.py -v
```

**Emission command (do NOT run until every alias is observed and the ledger is complete).**
Replace each quoted value with your genuine collection metadata; placeholders are rejected.

```
python -m scripts.validate_annotations --workbook "outputs/annotation/annotator_workbook.csv" --alias-map "outputs/annotation/sealed/alias_map.json" --review-ledger "data/annotation/review_ledger.csv" --text-origin user_observation --instrument-version "v1" --provenance "REPLACE with the real collection method; observer pool; collected YYYY-MM-DD; instrument v1" --source-reference "REPLACE with the real citation URL or stable URI" --emit-observations "data/paired_observations.csv"
```

Running the emission command as written is safely rejected (the provenance has no ISO date and
the source reference has no scheme), so it cannot silently produce fake evidence.

---

## 6. Workbook, ledger, and templates

Four distinct artifacts:

- **Blinded annotator workbook** (`outputs/annotation/annotator_workbook.csv`) — columns
  `alias,observation,observer_id,observed_at`. Contains **only** opaque aliases. Annotators never
  see class names, image paths, labels, or split membership.
- **Sealed alias map** (`outputs/annotation/sealed/alias_map.json`) — the confidential
  `alias → sample_id` mapping. Treat it as confidential: used **only** by the trusted
  collation/validation stage, never shared and never placed in the workbook.
- **Independent review ledger** (`data/annotation/review_ledger.csv`) — columns
  `sample_id,review_status,reviewer,reviewed_at,notes`. Adjudicates each observation; only
  `approved` rows may be emitted.
- **Emitted six-column training dataset** (`data/paired_observations.csv`) — columns
  `sample_id,image_path,text,text_origin,provenance,source_reference`. Produced only by the
  validator, only when every gate passes.

No completed ledger exists. To create a working copy from the header-only template **without
overwriting the template**:

```
copy data\annotation\review_ledger_template.csv data\annotation\review_ledger.csv
```

Headers already present: `sample_id,review_status,reviewer,reviewed_at,notes`. Populate one row
per reviewed sample. `sample_id` is the **image-relative path** (for example `apple scab/1.jpg`),
which is why the ledger is a trusted-stage file. `review_status` must be one of
`pending`, `approved`, `rejected`, `disputed`. Give the real reviewer identity and review date,
and leave `notes` empty until you have something genuine to record. Reviewer identities and
timestamps must never be invented.

Ledger IDs are matched to aliases through the sealed mapping during collation, so a ledger entry
is only meaningful for a sample that was actually observed under its sealed alias.

---

## 7. Atomicity and safety

- All gates are checked **before** any output is written.
- Emission writes to a temporary file next to the destination and moves it into place with
  `os.replace`; the staged file is re-validated through the real loader first.
- A failure can never leave a partial or misleading `data/paired_observations.csv`, and a
  pre-existing output is preserved unless a fully validated replacement is ready.
- Invalid observations are reported by alias with actionable reasons; they are never silently
  dropped and success is never reported for an incomplete set.
- Decisions are explicit: `INCOMPLETE` (missing observations), `BLOCKED` (rejected rows / failed
  gates), `READY` (valid, emission not requested), `EMITTED` (validated output written).

---

## 8. Real dataset requirements enforced by the workflow

Genuine image-specific observations; `text_origin=user_observation`; independent review approval;
valid provenance and source reference; recorded instrument version; `sample_id == image_path`;
eligibility-manifest membership; duplicate-free eligibility; official-split isolation; class
alignment from trusted official manifests; rejection of target-derived text, class-name prompts,
duplicate IDs, mismatched images, and unsupported samples.

Coverage: the eligible set is 17,870 samples (12,573 train / 1,751 validation / 3,546 test). With
zero observations, coverage is 0 in every split.

---

## 9. Verification results (actual)

```
python -m unittest discover -s tests -p test_multimodal.py -v
Ran 44 tests in 0.567s
OK
```

`AnnotationValidationStage2FTests` (new, all passing): valid provenance w/ instrument version;
missing/mismatched instrument version; placeholder provenance and source; real-detail
requirements; invalid URLs and unsupported schemes; placeholder collation; empty/unfilled
workbooks; malformed timestamps; actionable `observation_issues`; missing review ledger;
non-approved (pending/rejected/disputed) and missing review records; duplicate sample IDs; sample
identity mismatch; label leakage; missing/invalid source reference; failed emission does not
create or replace output; successful emission only when all gates pass; documented Windows
commands are single-line and placeholder-free.

Compilation:

```
python -m py_compile ml/fusion/data.py ml/fusion/annotation.py scripts/validate_annotations.py tests/test_multimodal.py
py_compile exit=0
```

Whitespace: `git diff --check` → exit 0 (only Git LF→CRLF advisories; no whitespace errors).

Read-only validator (real run):

```
{"workbook_rows": 17870, "aliases_defined": 17870, "eligible_samples": 17870,
 "filled_rows": 0, "missing_rows": 17870, "rejected_count": 0, "rejected_rows": [],
 "emitted": false, "decision": "INCOMPLETE",
 "blocker": "17870 of 17870 eligible images still have no observation"}
```

Paired-data audit (real run): **decision `BLOCKED`** —
`No image-specific, independently sourced text-pair CSV is present.` Official splits 13,045 /
1,820 / 3,677, 89 classes, `paired_csv_exists=false`, `usable_as_paired_training_text=false`.

---

## 10. Emission / audit status

- Observations emitted: **none** (0 filled aliases; no review ledger).
- `data/paired_observations.csv`: **does not exist**.
- Paired-data audit: **BLOCKED**.

---

## 11. Remaining blockers

1. No genuine, image-specific observations have been collected (0 of 17,870 aliases filled).
2. No completed independent review ledger exists, so nothing can be approved or emitted.
3. The only text asset on disk (`plantwild_prompts.json`) is class-level GPT prompts and is **not**
   valid image-specific text; it must not be used for paired training.

## 12. Final status

**BLOCKED**

The validation layer is fixed, hardened, atomic, and covered by regression tests; the workflow is
fully usable from Windows CMD. Training remains blocked until real blinded observations are
collected and independently reviewed.
