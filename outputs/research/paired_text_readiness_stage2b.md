# TerraMind AI — Stage 2B: Paired-Text Data Readiness

**Date:** 2026-10-17
**Decision:** **BLOCKED — no training; paired-text data does not yet exist**
**Scope:** annotation protocol, blinded template, review workflow, and inference-availability analysis. No models were trained, no splits/manifest/checkpoints/historical results were changed.

## 1. What was inspected

- `outputs/research/fusion_eligibility_manifest.json` — 17,870 eligible samples; per-split and per-class counts intact.
- `ml/fusion/eligibility.py`, `ml/fusion/data.py`, `ml/fusion/text_only.py` — eligibility filtering, six-column paired loader with leakage guards, and the independent text-only baseline.
- `scripts/audit_fusion_data.py` — read-only readiness audit (`BLOCKED` → `REVIEW_REQUIRED` once a valid CSV exists).

Confirmed: there is still no image-specific paired text. `plantwild_prompts.json` is class-conditioned GPT text and `url_record.json` is image-URL provenance; neither is per-image symptom text.

## 2. Annotation specification and blinding

`outputs/research/annotation_specification.md` defines the collection protocol. Key point: PlantWild paths embed the class name (`apple scab/1.jpg`), so annotators are given **opaque aliases only**. The alias → image map is written to a **sealed** file (`outputs/annotation/sealed/alias_map.json`) that is never shared with annotators. Workbooks that expose `image_path`, `sample_id`, class, label, or split columns are rejected by the loader.

## 3. Artifacts created

| Artifact | Purpose |
|---|---|
| `ml/fusion/annotation.py` | Alias generation, blinded workbook build/load, review-ledger load, verbatim collation, approved-only emission gate |
| `scripts/build_annotation_workbook.py` | Emits the blinded workbook + sealed alias map from the eligibility manifest |
| `scripts/validate_annotations.py` | Validates annotations, enforces review, and (optionally) emits + re-validates the training CSV |
| `data/annotation/annotator_workbook_template.csv` | Safe header-only annotator template (`alias, observation, observer_id, observed_at`) |
| `data/annotation/observations_template.csv` | Final six-column training schema (header only) |
| `data/annotation/review_ledger_template.csv` | Review ledger template (`sample_id, review_status, reviewer, reviewed_at, notes`) |
| `outputs/annotation/annotator_workbook.csv` | Generated blinded workbook — 17,870 alias-only rows, no path/label |
| `outputs/annotation/sealed/alias_map.json` | Sealed alias map (must not be shared with annotators) |
| `outputs/research/annotation_validation.json` | Latest validation report |

## 4. Review and disagreement process

Held-out blind double annotation (≥5% stratified) with two independent observers; symptom-feature agreement rubric; a third independent adjudicator resolves disagreements using the image only, allowed to select a submitted observation or a faithful synthesis of **visible** features — never new invented content. Unresolved rows stay `disputed` and are excluded. Only `approved` rows can be emitted into the training CSV.

## 5. Inference-time availability (key finding)

Curator-written image-specific observations exist only offline and are **not available at inference**. The honest options are: (a) reframe the modality as a **user-provided symptom description** captured at inference (`text_origin = user_observation`), with matching text collected during evaluation; or (b) keep `paired_human_observation` as a research-only offline study without claiming deployable multimodal inference. The production image-only model remains unchanged either way. This decision must be made before annotation scales.

## 6. Leakage checks (unchanged, re-verified)

Class/split come only from official split files; target/split columns are rejected; `sample_id` must equal the normalized `image_path`; class-name-derived text is rejected (`validate_text_not_label_derived`); unapproved `text_origin` values are rejected; provenance and `source_reference` are mandatory and validated; duplicate sample IDs and cross-split byte-identical images are rejected; emission is blocked unless every observation is review-approved.

## 7. Verification (actual results)

| Check | Command | Result |
|---|---|---|
| Focused tests | `python -m unittest discover -s tests -p test_multimodal.py -v` | **25 tests, OK** (7 new annotation-workflow tests) |
| Workbook build | `python -m scripts.build_annotation_workbook` | **exit 0** — 17,870 alias-only rows; sealed map written |
| Annotation validation | `python -m scripts.validate_annotations --workbook ... --alias-map ...` | **exit 0** — `INCOMPLETE` (0 filled rows), no emission |
| Compilation | `python -m py_compile` on changed modules | **exit 0** |
| Whitespace check | `git diff --check` | **exit 0** |

New tests cover: alias opacity/determinism, blinded-workbook enforcement (rejects path/label columns), verbatim collation with label-leakage rejection, review gate blocking pending/unobserved rows, review-ledger duplicate/status validation, emitted CSV passing the real training loader, and template column safety.

## 8. Decision and remaining blocker

**BLOCKED.** The protocol, safe template, sealed aliasing, validation workflow, review gate, and disagreement handling are implemented and tested, and the eligibility manifest is untouched (17,870 eligible: 12,573 / 1,751 / 3,546; every class has ≥4 eligible images per split). The blocker is unchanged and external to the code: **no genuine image-specific observations exist**, so there is nothing valid to train on.

**Exact remaining blocker:** collect and review image-specific observations for the eligible images (per `annotation_specification.md`), decide the inference-time text modality (Section 5), then emit and validate `data/paired_observations.csv`. Only then may `ml.fusion.train`, `ml.fusion.text_only`, and `ml.fusion.evaluate` run.

### Changed files
- `ml/fusion/annotation.py` — new
- `scripts/build_annotation_workbook.py` — new
- `scripts/validate_annotations.py` — new
- `data/annotation/annotator_workbook_template.csv`, `data/annotation/observations_template.csv`, `data/annotation/review_ledger_template.csv` — new
- `tests/test_multimodal.py` — added `AnnotationWorkflowTests`
- `outputs/research/annotation_specification.md` — new
- `outputs/research/paired_text_readiness_stage2b.md` — this report
- Generated: `outputs/annotation/annotator_workbook.csv`, `outputs/annotation/sealed/alias_map.json`, `outputs/research/annotation_validation.json`

Official splits, the eligibility manifest, production checkpoints, and historical results were not modified.
