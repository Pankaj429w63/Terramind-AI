# TerraMind AI — Stage 2C: Text Modality Decision and Data-Collection Preparation

**Date:** 2026-10-17
**Status:** **GO (protocol) / BLOCKED (data)** — the modality and collection path are finalized and verified; training stays blocked until real observations exist.
**Scope:** resolve the text-modality choice, document the chosen protocol, and specify the exact point at which training becomes permissible. No models were trained; official splits, the eligibility manifest, the production checkpoint, and historical results are unchanged.

## 1. The decision to make

Stage 2A/2B produced a duplicate-free eligible population (17,870 samples) and a blinded annotation workflow, but left one open question: **which text modality** the experiment should use. The choice determines whether the resulting model is deployable and whether the professor's three-fusion-method comparison has external validity.

## 2. Option A — User-provided symptom descriptions

Participants describe the plant/symptoms in their own words alongside the image. `text_origin = user_observation`.

- **Scenario:** in production, the submitter writes a short free-text description when uploading a photo; offline, contributors are briefed as "simulated users" and describe eligible PlantWild images through the same instrument.
- **Collection:** web/paper form showing one image per opaque alias with no path, filename, class name, or prompt; a fixed elicitation question reused verbatim at inference.

## 3. Option B — Blinded curator observations

Trained curators/annotators write morphology-focused observations offline for eligible images. `text_origin = paired_human_observation`.

- **Scenario:** a controlled research annotation of the PlantWild corpus; text is never available to a real end user.
- **Collection:** same blinded-alias mechanism, higher per-item training requirement.

## 4. Tradeoff comparison

| Criterion | Option A: user-provided symptoms | Option B: blinded curator observations |
|---|---|---|
| **Data availability at inference** | **Yes** — the user supplies it | **No** — curator text exists only offline |
| **Availability for offline PlantWild run** | Requires a contributor ("simulated user") pool; needs volume | Requires a trained annotator pool; expert time is the bottleneck |
| **Annotation cost** | Lower skill per item; higher volume/QC cost | Higher skill per item; costly to scale to 17,870 |
| **Label leakage risk** | Low under blinded alias; non-experts may still name plant/disease from prior knowledge — mitigated by class-name rejection + review | Low; trained observers describe morphology, less likely to name labels |
| **Inference compatibility** | **High** — matches the existing `paired_text` form field on `/api/multimodal/analyze` | **None** — a farmer cannot provide a curator's hidden caption |
| **Scientific validity** | High for the deployed system: train/val/test/inference text come from one instrument and one distribution | High as an offline benchmark of morphological text, but external validity is limited to curator-style input |
| **Fit for the three-fusion comparison** | Strong — image-only vs text-only vs early/intermediate/late under realistic (noisy) text; supports a deployability claim | Strong for an architecture-only study; cannot support a production multimodal claim |
| **Deployability** | Yes | No (research-only) |

## 5. Recommendation and justification

**Choose Option A (user-provided symptom descriptions).**

1. **Only Option A is deployable.** The production inference path already requires the text: `POST /api/multimodal/analyze` takes `paired_text: str = Form(...)` alongside the image. Option B has no matching inference-time source.
2. **Distribution match = honest evaluation.** Training, validation, test, and inference text all come from the same elicitation instrument, so the fusion comparison reflects real operating conditions rather than an offline-only setting.
3. **It still answers the professor's question.** Image-only, text-only, and all three fusion methods are compared on the identical eligible samples and official splits; the modality is simply the realistic one.
4. **Option B remains valid but must be scoped.** If the professor deliberately defines the study as offline/benchmark-only, Option B may be used instead — recorded as a separate, clearly labeled research variant.

**No mixing.** Option A and Option B text must never be combined in one experiment. A model trained on curator text cannot be evaluated or deployed against user text, and vice-versa. Any deliberate mixture would require its own documented protocol showing the two sources are statistically interchangeable for every split; none exists, so mixing is prohibited.

## 6. Schema, provenance, and safeguards (unchanged)

The six-column training schema is kept as-is; no incompatibility was demonstrated.

`sample_id, image_path, text, text_origin, provenance, source_reference`

For Option A the emitted rows carry `text_origin = user_observation`. Provenance and review approval are validated **before** any data emission:

- `provenance` must name the collection method, participant pool, dates, and **instrument version** (≥12 chars, not a placeholder).
- `source_reference` must be a URL or stable URI to the response records.
- Emission is blocked unless every observation has a review-ledger entry whose `review_status` is `approved`.
- Collation copies observation text verbatim; empty rows are skipped, never auto-filled.

## 7. Inference input requirements (Option A)

- A required free-text field on the upload form: one short description of what the submitter sees, 1–3 sentences.
- The field must be the **same instrument** (same question, same guidance, same length expectation) used during collection; its version is recorded in `provenance`.
- Input is rejected if it contains a PlantWild class name (target leakage) — the same `validate_text_not_label_derived` rule used at training time.
- The user is never shown predicted labels or prompts before writing.

## 8. Data quality checks

- Blinded workbook only (`alias, observation, observer_id, observed_at`); any path/label/split column is rejected.
- Review ledger keyed by `sample_id` with `review_status ∈ {pending, approved, rejected, disputed}`; duplicates and unknown statuses rejected.
- Held-out blind double annotation (≥5% stratified) with a symptom-feature agreement rubric; disputes adjudicated by a third party using the image only.
- Emitted CSV must pass `load_paired_text_csv` on the real dataset and the eligibility filter.

## 9. Exact point at which training becomes permissible

Training may start only when **all** of the following hold:

1. `data/paired_observations.csv` exists with exactly the six columns and passes `load_paired_text_csv` on the real dataset.
2. Every row's `text_origin = user_observation` (Option A).
3. `python -m scripts.validate_annotations ... --emit-observations` succeeds, i.e. every row is `approved` in the review ledger.
4. All emitted samples are in the eligibility manifest (`eligible_by_split`); the filtered set is non-empty for train, validation, and test.
5. Each evaluated class is represented in the eligible train and validation sets (every class has ≥4 eligible images per split, so this is achievable).
6. The elicitation instrument version is recorded, and the same instrument is specified for inference.

Until then, `ml.fusion.train`, `ml.fusion.text_only`, and `ml.fusion.evaluate` must not run. `scripts/audit_fusion_data.py --paired-csv` returns `BLOCKED` now and `REVIEW_REQUIRED` (not `GO`) after structural checks pass.

## 10. Exact data requirements and validation commands

**Requirements:** approved user-style observations for the eligible samples — full target 17,870 (12,573 / 1,751 / 3,546); minimum viable is at least one approved observation per class per split.

```powershell
# 1. Blinded workbook + sealed alias map (already generated; idempotent)
python -m scripts.build_annotation_workbook

# 2. Contributors fill observation text; curator compiles the review ledger

# 3. Validate and emit ONLY when every row is approved
python -m scripts.validate_annotations --workbook <filled.csv> `
    --alias-map outputs/annotation/sealed/alias_map.json --review-ledger <ledger.csv> `
    --text-origin user_observation --instrument-version "<instrument version>" `
    --provenance "<method + participant pool + dates + instrument version>" `
    --source-reference <stable-uri> --emit-observations data/paired_observations.csv

# 4. Structural audit (must not report BLOCKED)
python scripts/audit_fusion_data.py --paired-csv data/paired_observations.csv
```

## 11. Verification (actual results)

| Check | Command | Result |
|---|---|---|
| Focused annotation + fusion-data tests | `python -m unittest discover -s tests -p test_multimodal.py -v` | **25 tests, OK** |
| Compilation | `python -m py_compile` on annotation/data/text_only/eligibility + scripts + tests | **exit 0** |
| Whitespace check | `git diff --check` | **exit 0** |
| Workbook blinding | grep `\.jpg|/|apple|tomato|scab|rust|blight` over `outputs/annotation/annotator_workbook.csv` | **0 matches** (17,870 alias-only rows) |

Safeguard evidence: the workbook header is `alias,observation,observer_id,observed_at` with no path/label/split; the sealed map lives at `outputs/annotation/sealed/alias_map.json`; `load_annotator_workbook` raises `must not expose` on any leaking column (covered by `test_workbook_rejects_label_or_path_exposure`); `require_reviewed` blocks pending/disputed rows (covered by `test_review_gate_blocks_unapproved_or_unobserved`); review decisions are auditable via `reviewer`, `reviewed_at`, and `notes`.

## 12. GO / BLOCKED

- **GO** for the modality decision and the collection protocol: Option A is selected, the schema is unchanged, safeguards are implemented and tested, and the training gate is unambiguous.
- **BLOCKED** for training: zero real observations exist. Nothing may be trained until Section 9 is satisfied.

**Remaining blocker:** collect and review genuine user-style symptom descriptions for the eligible PlantWild images, emit `data/paired_observations.csv` through the approved-only gate, and record the elicitation instrument version. No synthetic text may be substituted.

### Changed files (this stage)
- `outputs/research/text_modality_decision.md` — new (this report)
- `outputs/research/annotation_specification.md` — Section 9 finalized to the chosen modality

No code, official splits, eligibility manifest, production checkpoint, or historical results were modified in Stage 2C.
