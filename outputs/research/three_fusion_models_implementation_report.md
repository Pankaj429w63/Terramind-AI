# TerraMind AI — Three Fusion Models: Implementation Report

This report records what was actually implemented, executed, and measured for the
early / intermediate / late fusion models. It contains only verified facts produced
by real commands in this repository. Nothing here is estimated, invented, or
carried over from an earlier run.

## 0. Headline status

| Dimension | Status |
|---|---|
| Implementation complete (architectures, runtime, API, frontend client, tests) | **Yes** |
| Executable (all three architectures run on CPU and emit 89-class output) | **Yes** |
| Selectable in the application (status + analyze routes) | **Yes** |
| Caption generation (eligible-image coverage) | **In progress — 7,955 / 17,870 live store records, train only** |
| Caption information content | **Too generic/duplicated for a meaningful multimodal experiment (measured below)** |
| Trained (any fusion model) | **No — none trained; the training gate refuses** |
| Test evaluation recorded | **No — `NOT_EVALUATED`** |
| Scientifically valid fusion result | **No — no valid paired text for validation/test, and captions carry little signal** |
| Production EfficientNet-B0 checkpoint | **Unchanged** |

The three architectures are real, separately selectable models, not renamed copies.
They can be loaded, exercised, and served. No fusion model has been trained because
the data-validation gates that the project contract requires do not pass: the paired
text has **no validation or test rows**, so the training script refuses to start, and
the generated captions are too generic to support a defensible multimodal claim.

## 1. Implemented architectures and files

### Architectures (`ml/fusion/models.py`, pre-existing, reused)

All three share the production EfficientNet-B0 feature extractor (1280-D features,
`AdaptiveAvgPool2d(1)`) and a GRU text encoder, but differ in where and how the
modalities meet:

- **Early fusion (`build_fusion_model("early", ...)`)** — projects image and text
  representations into a compatible `fusion_dim`, concatenates them *before* the
  shared head, then applies dropout and a two-layer 89-class classifier.
- **Intermediate fusion (`"intermediate"`)** — a learned sigmoid **gate** combines a
  joint projection with a residual average of the two modality vectors
  (`gate * joint_projection([image; text]) + (1 - gate) * 0.5 * (image + text)`), so
  both modalities influence the shared representation before classification. Chosen over
  attention for CPU efficiency at the configured dimensions.
- **Late fusion (`LateFusionClassifier`, `"late"`)** — separate image and text
  classifiers, combined by a validation-tuned modality weight
  (`predict_proba(image, text)`). It has a genuine **image-only branch**
  (`image_classifier`), documented as such.

### New files created this session

| File | Purpose |
|---|---|
| `ml/fusion/inference.py` | Selectable inference runtime; validates sidecar manifest + SHA-256 + class order + format; loads exactly one checkpoint; **always reports `modalities_used`**; labels image-only late-fusion results as `image_only_fallback`; raises `MissingTextError` for early/intermediate without text. |
| `scripts/validate_fusion_dataset.py` | Runnable dataset/paired-data validation (§6). Writes `outputs/research/fusion_dataset_validation.json`. |
| `scripts/fusion_comparison_report.py` | Runnable comparison report (§6); reports only measured metrics, missing evaluations as "not evaluated". Writes `outputs/research/three_fusion_models_comparison.{md,json}`. |
| `frontend/services/fusion.ts` | Frontend client for `/api/fusion/status` and `/api/fusion/analyze`, typed to expose `modalities_used`, `multimodal`, `image_only_fallback`, `notice`. |
| `tests/test_fusion_inference.py` | 15 tests: checkpoint save/load, class-order + checksum rejection, 89-class shapes for batch 1 and 3, CPU execution, modality reporting, labelled image-only fallback, missing-text guard, **finite loss + gradient flow on every trainable parameter for all three kinds**, and late-fusion determinism / required-weight selection. |

### Files modified this session

| File | Change |
|---|---|
| `backend/app.py` | Added fusion imports, `FUSION_RUNTIMES`/`FUSION_DEVICE` globals, `_fusion_runtime(kind)`, `GET /api/fusion/status`, `POST /api/fusion/analyze`, and root-index links. |
| `scripts/test_backend_api.py` | Added assertions for `/api/fusion/status` (kinds, per-kind `requires_text`, production availability) and guarded `/api/fusion/analyze` (503 without checkpoint, 422 for unknown kind). |

## 2. Exact commands

Run from the repository root.

### Validate datasets and paired text (§6.1, §6.2)

```powershell
python -m scripts.validate_fusion_dataset --paired-csv data/paired_model_observations.csv
```
Writes `outputs/research/fusion_dataset_validation.json`. Add `--strict` to exit 1 on a hard failure.

### Train a fusion model (§6.3–§6.5)

```powershell
python -m ml.fusion.train --kind early        --paired-data data/paired_model_observations.csv --eligibility-manifest outputs/research/fusion_eligibility_manifest.json
python -m ml.fusion.train --kind intermediate --paired-data data/paired_model_observations.csv --eligibility-manifest outputs/research/fusion_eligibility_manifest.json
python -m ml.fusion.train --kind late         --paired-data data/paired_model_observations.csv --eligibility-manifest outputs/research/fusion_eligibility_manifest.json
```
Training refuses to start unless: a registered text origin is present, the sample IDs
are eligible, **both train and validation splits are non-empty**, and the image encoder
is the 89-class EfficientNet-B0. The vocabulary is fit on training text only.

### Baselines (§6.6, §6.7)

```powershell
python -m ml.fusion.text_only --paired-data data/paired_model_observations.csv --eligibility-manifest outputs/research/fusion_eligibility_manifest.json
```
The image-only baseline is produced inside any fusion `evaluate` run on the exact same eligible samples.

### Final test evaluation and comparison (§6.8–§6.10)

```powershell
python -m ml.fusion.evaluate --checkpoint outputs/fusion/early/best_model.pth        --paired-data data/paired_model_observations.csv --eligibility-manifest outputs/research/fusion_eligibility_manifest.json
python -m ml.fusion.evaluate --checkpoint outputs/fusion/intermediate/best_model.pth --paired-data data/paired_model_observations.csv --eligibility-manifest outputs/research/fusion_eligibility_manifest.json
python -m ml.fusion.evaluate --checkpoint outputs/fusion/late/best_model.pth         --paired-data data/paired_model_observations.csv --eligibility-manifest outputs/research/fusion_eligibility_manifest.json
python -m scripts.fusion_comparison_report
```
Evaluation writes `outputs/fusion/<kind>/evaluation/test_metrics.json`; the comparison
report reads only those files.

### Serve the application

```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001
```
Select a fusion model at request time via the `kind` form field (`early` | `intermediate` | `late`).

### Verification

```powershell
python -m unittest discover -s tests -p "test_fusion_inference.py"
python -m unittest discover -s tests -p "test_multimodal.py"
python scripts/test_backend_api.py
python -m compileall -q backend tests scripts ml
git diff --check
```

## 3. Dataset counts and provenance (recomputed this session)

Official split files (untouched), SHA-256 over image bytes for duplicate detection:

| Quantity | Value |
|---|---:|
| Official path rows | 18,542 |
| Official train / validation / test | 13,045 / 1,820 / 3,677 |
| Classes | 89 |
| Exact-duplicate groups (all / cross-split / within-split) | 335 / 155 / 180 |
| Excluded: crosses official split | 333 |
| Excluded: conflicting labels | 309 |
| Excluded: duplicate copy within same split | 30 |
| **Eligible rows** | **17,870** |
| **Eligible train / validation / test** | **12,573 / 1,751 / 3,546** |

Paired text. Two states must be distinguished, because the emitted CSV is a snapshot of
an earlier store state and generation is still running:

- **Live generation store** — `outputs/annotation/model_generated/generation_records.csv`
  (what has actually been captioned so far).
- **Emitted CSV** — `data/paired_model_observations.csv` (last written from an earlier
  snapshot; **not** re-emitted yet, because generation is incomplete).

| Quantity | Live store (2026-10-18, still growing) | Emitted CSV (snapshot) |
|---|---:|---:|
| Rows | 7,955 | 3,183 |
| Rows by split (train / validation / test) | 7,955 / **0** / **0** | 3,183 / **0** / **0** |
| Coverage (train / validation / test) | 0.6327 / **0.0** / **0.0** | 0.2532 / **0.0** / **0.0** |
| Text origins | `model_generated_observation` | `model_generated_observation` |
| `human_origin_only` | **false** | **false** |
| Duplicate sample IDs | 0 | 0 |
| Rows outside eligible manifest | 0 | 0 |
| `validate_fusion_dataset --strict` | — | **`VALID`** (exit 0) |

**Provenance:** the only paired text is model-generated captions (Strategy C track),
stored separately, marked `model_generated_observation`, never `user_observation`.
There is no human-annotated paired text (`paired_human_observation` / `user_observation`)
in this repository yet.

### 3.1 Caption information content (measured on the live store)

Measured directly from `generation_records.csv` (5,061 records at measurement time):

| Measure | Value |
|---|---:|
| Descriptions | 5,061 |
| Unique exact strings | 1,634 |
| Unique after punctuation/case normalisation | 1,630 |
| Rows beginning with the identical 10-word prompt echo ("a close - up view of the plant surface showing") | 5,061 (100%) |
| Mean total words | 15.46 |
| Mean variable words after the fixed prefix | **5.46** |
| Most common variable content ("the green leaves") | **694 rows = 13.7%** |
| Top-10 variable contents | **29.7% of all rows** |
| Variable contents appearing exactly once | 1,192 |

**Assessment:** the descriptions are **generic and heavily duplicated**. Every row repeats
the same prompt prefix, and the only variable part averages ~5.5 words drawn from a tiny,
repetitive vocabulary ("the green leaves", "the damage of the leaves", "the brown spots on
the leaves", "the yellow spots", "the fruit on the tree"). BLIP restates the prompt rather
than describing disease-specific symptoms, so the text carries almost no discriminative
signal and almost no image-specific diagnostic content.

**Consequence:** per the mandate, this track is reported as a **non-diagnostic engineering
experiment**. Even if a fusion model trained on it showed a validation/test difference, that
difference would be uninterpretable and must **not** be presented as valid symptom-language
fusion performance. A defensible multimodal claim requires genuinely observed human text
(Strategies A/B), which does not exist in this repository.

## 4. Actual test results

| Check | Command | Result |
|---|---|---|
| Fusion inference tests | `python -m unittest discover -s tests -p "test_fusion_inference.py"` | **15 tests, OK** |
| Multimodal suite | `... -p "test_multimodal.py"` | **44 tests, OK** |
| Blinded review tool suite | `... -p "test_blinded_review_tool.py"` | **29 tests, OK** |
| Annotation generation suite | `... -p "test_model_annotation_generation.py"` | **32 tests, OK** |
| Backend endpoints (incl. fusion status/analyze) | `python scripts/test_backend_api.py` | **ALL BACKEND ENDPOINT TESTS PASSED** |
| Python compilation | `python -m compileall -q backend tests scripts ml` | **exit 0** |
| Whitespace check | `git diff --check` | **exit 0** (only LF→CRLF warnings) |
| Dataset/paired validation (strict) | `python -m scripts.validate_fusion_dataset --strict` | **`VALID`, exit 0** |
| Fusion-data audit | `python -m scripts.audit_fusion_data` | **`BLOCKED` — "No image-specific, independently sourced text-pair CSV is present."** |
| Emission dry-run (no write) | `python -m scripts.emit_model_observations --dry-run` | **`DRY_RUN_OK`** |
| Training gate | `python -m ml.fusion.train --kind early ...` | **Refused: "Paired dataset needs at least one official train and validation example" (exit 1)** |
| Comparison report | `python -m scripts.fusion_comparison_report` | **`not_evaluated_kinds = [early, intermediate, late]`** |

The 15 fusion tests verify: all three kinds load on CPU; class-order and checksum
mismatches are rejected; untrained manifests are unavailable; 89-class output for
batch sizes 1 and 3; multimodal prediction reports `["image","text"]`; late-fusion
image-only fallback reports `["image"]` with `image_only_fallback=true`; early and
intermediate raise `MissingTextError` for `None` and blank text; CPU determinism; and
that a forward pass through each architecture yields a finite cross-entropy loss whose
gradients reach **every** trainable parameter (`FusionTrainingGraphTests`).

## 5. Actual training status

**No fusion model has been trained.** `outputs/fusion/` does not exist, so no fusion
checkpoint is installed. The gate was verified by running the real training command,
which refused before any weights were built or saved:

```
$ python -m ml.fusion.train --kind early --paired-data data/paired_model_observations.csv \
      --eligibility-manifest outputs/research/fusion_eligibility_manifest.json
ValueError: Paired dataset needs at least one official train and validation example
exit=1
```

Two independent reasons block a defensible experiment:

1. **No validation/test paired text.** The paired CSV has train rows only, so the
   first-class gate (`train` + `validation` non-empty) fails and training cannot start.
2. **Captions are too generic** (§3.1): 100% prompt-echo, ~5.5 variable words, top-10
   variable contents = 29.7% of rows. The text is not a meaningful symptom signal.

The production checkpoint (`models/final_model.pth`) was not touched. No training was
launched, so no CPU hours were spent on an experiment that could not be interpreted.

## 6. Checkpoint paths

| Kind | Expected path | Present? |
|---|---|---|
| Early fusion | `outputs/fusion/early/best_model.pth` (+ `.manifest.json`) | No |
| Intermediate fusion | `outputs/fusion/intermediate/best_model.pth` (+ `.manifest.json`) | No |
| Late fusion | `outputs/fusion/late/best_model.pth` (+ `.manifest.json`) | No |
| Text-only baseline | `outputs/fusion/text_only/best_model.pth` | No |
| Production image-only | `models/final_model.pth` | Yes — unchanged, SHA-256 `D958430372CF4F953BF26A629E465A8840D185D6A8C83A13243E9DC96F95F454` |

`outputs/fusion/` does not exist, confirming no fusion checkpoint was written. (Note: the
archive file `models/archive/2026-09-09_efficientnet_b0_official_v3/best_model.pth` is a
separate historical artifact with SHA-256 `9AC45C5F…B5A28B`; it is **not** the production
checkpoint referenced by the reports, and it is also unchanged.)

## 7. API / frontend integration status

- `GET /api/fusion/status` → `supported_kinds`, per-kind `implemented`/`trained`/`available`/
  `requires_text`, plus `production_image_only_available`. Verified: 200, three entries,
  `requires_text` true for early/intermediate and false for late.
- `POST /api/fusion/analyze` (multipart: `file`, `kind`, optional `paired_text`, `top_k`) →
  returns `predicted_class`, `confidence`, `top_k`, `modalities_used`, `multimodal`,
  `image_only_fallback`, plus `model` metadata and a `notice`. Behavior verified:
  - No checkpoint installed → **503** with a clear unavailable message.
  - Unknown `kind` (e.g. `midfusion`) → **422**.
  - Missing text on a text-requiring kind → **422** (`MissingTextError`).
  - Late fusion without text → `modalities_used == ["image"]` and a notice stating it "is not a multimodal result".
- Frontend client `frontend/services/fusion.ts` exposes `getFusionStatus()` and
  `analyzeFusionImage(kind, file, pairedText?, topK?)`.
- The production image-only model remains the default and is unaffected; fusion loading
  is lazy and cached, so an unavailable fusion experiment cannot block the service.

## 8. Known limitations

1. **No validation/test paired text** → a fair train/validation/test fusion comparison
   cannot be produced. This is the single blocking dependency.
2. **Only model-generated text exists**, and only for train at 0.2532 coverage. Per the
   project's scientific contract, generated captions do not establish valid
   symptom-language fusion performance and cannot be reported as human evidence.
3. **No independent review ledger** exists for the paired data.
4. **No trained checkpoint** → no parameter counts, latencies, or confusion matrices for
   the fusion models. Any such numbers would be fabricated and are therefore omitted.
5. Reported fusion-model metrics do not exist yet; the comparison table below is empty by design.

## 9. Comparison table (measured metrics only)

Source: `outputs/research/three_fusion_models_comparison.json` → `status: NOT_EVALUATED`.

| Model | Test N | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Top-3 | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Early fusion | — | — | — | — | — | — | — | — |
| Intermediate fusion | — | — | — | — | — | — | — | — |
| Late fusion | — | — | — | — | — | — | — | — |
| Image-only baseline | — | — | — | — | — | — | — | — |
| Text-only baseline | — | — | — | — | — | — | — | — |

" — " means *not evaluated*, not zero. The table is populated automatically after real
`evaluate` runs. **Fusion must not be claimed to improve on any baseline unless this
table, computed on the same eligible sample IDs, demonstrates it.**

## 10. Implementation-complete vs training-complete vs scientifically validated

- **Implementation-complete — yes.** Three distinct architectures; a selectable runtime
  that reports modalities and labels the image-only fallback; deterministic checkpoint
  format with manifest, seed, class mapping, vocabulary, eligibility-manifest hash, and
  production-checkpoint hash; dataset-validation and comparison commands; backend and
  frontend integration; 15 passing unit tests plus endpoint checks; production checkpoint
  untouched.
- **Training-complete — no.** No model trained; `outputs/fusion/` is absent.
- **Scientifically validated — no.** No paired text exists for validation/test, and the
  only available text is model-generated. There is no controlled, same-sample evaluation.

## 11. Remaining blocker and fastest legitimate resolution

**Blocker:** paired text exists only for `train` (7,955 model-generated rows in the live
store and growing; validation `0/1,751`, test `0/3,546`), and it is not human-origin. A
fusion-data audit run with no paired CSV returns `BLOCKED` ("No image-specific,
independently sourced text-pair CSV is present."); even if it were pointed at the
model-generated CSV, it would still fail the `human_origin_only` and
`independent_review_ledger` gates and the `all_splits_covered` gate until validation/test
rows exist. The training gate additionally refuses with
*"Paired dataset needs at least one official train and validation example"*.

**Running job (do not duplicate):** caption generation was resumed and is running:

```powershell
python -m scripts.generate_image_annotations --all --checkpoint-every 100 --progress-every 200
```

It is processing ~1.08 s/image with ~10,100 images remaining (ETA ~3.0 h on CPU), and it is
safe to interrupt and re-run (it resumes by skipping completed `sample_id`s). It walks the
manifest in order, so **validation and test coverage appear only after the train split completes**.

**Fastest legitimate resolution (in order):**

1. **Let generation finish, then re-emit and re-audit:**
   ```powershell
   python -m scripts.emit_model_observations
   python -m scripts.audit_fusion_data --paired-csv data/paired_model_observations.csv
   ```
   This is the only step that can flip `all_splits_covered` on its own. It cannot flip
   `human_origin_only` or `independent_review_ledger`.
2. **For a defensible (non-model-generated) result, obtain genuinely paired human text
   for validation and test** (Strategy A: the existing blinded workbook + review-ledger
   workflow; or Strategy B: a legitimately licensed image-text source with matching image
   identity and clean taxonomy). Register it in a paired CSV with an allowed human origin.
3. **Only if both are satisfied**, run the three `ml.fusion.train` commands, then
   `ml.fusion.evaluate` per kind, then `python -m scripts.fusion_comparison_report`.

Note: even after step 1 completes, the model-generated track is an **engineering**
experiment only (§3.1) — its metrics, if produced, cannot be reported as valid
symptom-language fusion performance.

**Single most useful next command (after the running generation completes):**

```powershell
python -m scripts.emit_model_observations
```

Until then, all independent work is complete; only the running generation and the human
paired-text dependency remain.
