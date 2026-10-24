# TerraMind AI — Annotation Workflow: Next Stage Audit & Improvements

**Date:** 2026-10-17
**Stage:** 2G (post Stage 2F annotation-validation fix + blinded review tool)
**Scope:** audit the working blinded review tool and ship the next safe, practical step for collecting genuine observations.
**Non-goals:** no model training, no annotation generation, no auto-approval, no paired-data emission, no changes to official splits/manifest/checkpoint/metrics.

---

## 1. Confirmed starting state

| Check | Command | Result |
|---|---|---|
| Aliases resolve | `python -m scripts.review_annotations --check` | `READY`, 17,870 resolvable, **0 unresolved** |
| Workbook rows | `outputs/annotation/annotator_workbook.csv` | 17,870 rows, **0 filled** |
| Tool port | `python -m scripts.review_annotations --port 8040` | opens `http://127.0.0.1:8040` |
| Fusion models | — | untrained (blocked on genuine paired observations + independent review) |

The sealed mapping resolved safely, so per the task's stop-condition there was no need to halt; work proceeded.

---

## 2. Audit findings

### 2.1 Save semantics (`record()` in `ml/annotation/review_tool.py`)
- Locates the **single** workbook row whose alias matches the request; refuses with a clear error if the alias is absent.
- Refuses to write when the row already contains an observation: `"this image already has an observation; refusing to overwrite it"`.
- Runs `validate_text_not_label_derived` before writing (label/species leakage rejected).
- Writes via `_atomic_write_workbook` (`tempfile.NamedTemporaryFile` + `os.replace`) so a crash cannot truncate the file.
- Row order and all other rows are preserved verbatim.

**Verdict:** saving updates only the matching alias, preserves all other observations, and never silently overwrites prior work. ✅

### 2.2 Observer id + timestamp (`record()`)
- `observer_id` is supplied by the human once via `POST /api/observer`; placeholders such as `unknown` are rejected by `_is_placeholder`. Nothing is invented.
- `observed_at = datetime.date.today().isoformat()` — the **real** save date — then passed through `validate_observed_at`.
- No provenance or source-reference is written here (those are emission-time fields populated by the human at collation).

**Verdict:** identity and timestamp are recorded correctly and are never fabricated. ✅

### 2.3 Next / errors / restart  → **defect found and fixed**
- `Next` previously called `POST /api/skip` and immediately advanced, **silently discarding any unsaved text** in the box.
- **Fix:** the Next handler now guards on unsaved text:
  ```javascript
  if (el("obs").value.trim() && !window.confirm("This image has unsaved text. Skip it and discard that text?")) return;
  ```
- Errors (bad observer id, label-derived text, duplicate save, unknown alias) are already returned as non-2xx and rendered into the page status line.
- Restart: the workbook is the single source of truth, so saved observations survive; in-memory skips reset (correct — skips are session-only and are never persisted).

**Verdict:** unsaved text is now protected; errors surface; restart preserves saved work. ✅

### 2.4 Progress indicator  → **added**
- New `ReviewContext.progress()` returns **counts only**:
  `{"completed", "remaining", "total"}` — no aliases, sample ids, paths, labels, or split membership.
- New `GET /api/progress` route exposes exactly that dict.
- Page shows `Completed: N · Remaining: M`, refreshed at the top of every `loadNext()`.

**Verdict:** safe progress is displayed and leaks nothing (asserted by tests). ✅

### 2.5 No auto-generation / auto-approval / paired-data creation
- The tool only **collects**: it writes the workbook only. It never writes the review ledger, never writes `data/paired_observations.csv`, never flips a review status.
- Verified by `test_collection_tool_never_writes_review_fields_or_ledger`. ✅

### 2.6 Independent review  → **safety gap found and fixed**
- `annotation_specification.md` §8 already prescribes blind double annotation + independent adjudication, and `data/annotation/review_ledger_template.csv` defines `sample_id,review_status,reviewer,reviewed_at,notes`.
- **Gap:** nothing prevented an annotator from being recorded as their own independent reviewer at emission.
- **Fix:** added `require_independent_review(workbook_rows, alias_map, review_ledger)` to `ml/fusion/annotation.py`. It rejects emission when an `approved` row's `reviewer` equals its own `observer_id` (case-insensitive), and rejects an `approved` row with no named reviewer. Wired into `scripts/validate_annotations.py` as `report["independent_review_gate"]`.
- The collection tool is explicitly **not** a reviewer interface: independent review remains a separate human step against the ledger template.

**Verdict:** self-approval is now structurally impossible at the emission gate. ✅

### 2.7 No separate review UI built
A dedicated independent-review browser UI was **not** built. The safe minimum — documented procedure (§8) + ledger template + a code-enforced independence gate — is in place, and the collection tool stays strictly collection-only. Building a second blinded UI is the natural next step if/when human reviewers are stood up.

---

## 3. Files changed

All under the repo root `c:\Users\PANKAJ YADAV\OneDrive\Desktop\Terramind AI`. These modules are new (untracked) from earlier stages, so they appear as untracked paths, not `git diff` hunks.

| File | Change |
|---|---|
| `ml/annotation/review_tool.py` | Added `ReviewContext.progress()`; added `GET /api/progress`; added `#progress` UI element + CSS + `refreshProgress()`; added unsaved-text confirm guard on Next; expanded module docstring (collect-only, independent review separate). |
| `ml/fusion/annotation.py` | Added `require_independent_review(...)` emission gate. |
| `scripts/validate_annotations.py` | Imported and invoked `require_independent_review`, recorded as `independent_review_gate` in the report. |
| `tests/test_blinded_review_tool.py` | Added imports, `_make_repo` helper (with `by_sample`), 8 new tool tests + `IndependentReviewGateTests` (5 tests). Test count 13 → 23. |
| `outputs/research/annotation_workflow_next_stage.md` | This report (new). |

**Untouched by design:** official PlantWild split files, `fusion_eligibility_manifest.json`, production EfficientNet-B0 checkpoint, historical metrics, and all unrelated working-tree changes.

---

## 4. Regression coverage added (`tests/test_blinded_review_tool.py`)

- `test_save_updates_only_the_matching_alias_and_preserves_the_rest`
- `test_duplicate_save_is_refused_and_original_text_kept`
- `test_progress_counts_reflect_completed_and_remaining`
- `test_progress_endpoint_leaks_nothing_and_matches_context`
- `test_restart_preserves_saved_observations_and_resets_skips`
- `test_skip_never_writes_or_alters_the_workbook`
- `test_page_shows_progress_and_guards_unsaved_text`
- `test_collection_tool_never_writes_review_fields_or_ledger`
- `IndependentReviewGateTests`: `test_self_approval_is_rejected`, `test_reviewer_matching_is_case_insensitive`, `test_distinct_reviewer_is_accepted`, `test_approved_row_without_a_named_reviewer_is_rejected`, `test_unapproved_rows_are_left_to_the_other_gates`

---

## 5. Actual test results (run on 2026-10-17)

| Command | Result |
|---|---|
| `python -m unittest discover -s tests -p "test_blinded_review_tool.py" -v` | **Ran 23 tests — OK** |
| `python -m unittest discover -s tests -p "test_multimodal.py" -v` | **Ran 44 tests — OK** |
| `git diff --check` | exit 0; only `LF will be replaced by CRLF` notices, **no whitespace errors** |
| `python -m scripts.review_annotations --check` | `READY`, 17,870 resolvable, 0 unresolved, 0 filled |
| real workbook | 17,870 rows, **0 filled** (unchanged) |

No test is reported as passing unless it was actually executed above.

---

## 6. Blockers / status

- **Status: INCOMPLETE (collection).** The tool is safe and functional; the only remaining bottleneck is **genuine human observations**, then an **independent human review** in the ledger.
- Fusion training stays **BLOCKED** until: (a) observations are collected, (b) the review ledger marks rows `approved`, and (c) `scripts/validate_annotations.py` passes all gates (including the new independence gate) to emit `data/paired_observations.csv`.
- No safety or data-integrity issue was left unresolved; both found issues (silent Next discard, self-approval) were fixed and covered by tests.

---

## 7. Exact next CMD command

From the repo root `c:\Users\PANKAJ YADAV\OneDrive\Desktop\Terramind AI`:

```
python -m scripts.review_annotations --port 8040
```

Then open `http://127.0.0.1:8040`, enter your real observer id, and record genuine visual observations one image at a time. After collection, run the independent review against `data/annotation/review_ledger_template.csv`, then validate/emit:

```
python -m scripts.validate_annotations
```
