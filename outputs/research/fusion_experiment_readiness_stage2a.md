# TerraMind AI — Stage 2A: Fusion Experiment Readiness

**Date:** 2026-10-17
**Decision:** **BLOCKED — fusion models remain untrained**
**Scope:** duplicate-free eligibility manifest, shared-baseline filtering, independent text-only control, registry/interface audit, and paired-data leakage verification. No training or evaluation was run. The production EfficientNet-B0 checkpoint, official split files, and historical metrics were not modified.

## 1. Duplicate-free eligibility manifest

`scripts/build_fusion_eligibility.py` builds a deterministic manifest from SHA-256 image content hashes and writes `outputs/research/fusion_eligibility_manifest.json` (format `terramind-duplicate-free-eligibility-v1`). Official split assignments are read from `train_split.txt` / `val_split.txt` / `test_split.txt` and are never rewritten; the manifest only records which officially-assigned samples remain eligible for comparison.

**Policy**

- Official class and split for every path come only from the official split files (`official_assignments_are_immutable: true`).
- A byte-identical (same SHA-256) group present in **more than one** official split is excluded in full — every member of the group is dropped.
- Within a single split, copies with **conflicting** official classes are all excluded.
- Within a single split, copies sharing the same official class retain only the lexicographically first relative path; the rest are dropped as redundant copies.
- No path is ever reassigned to a different split.

**Counts (regenerated on this checkout)**

| Quantity | Value |
|---|---:|
| Official split rows | 18,542 |
| Eligible rows | 17,870 |
| Eligible train / validation / test | 12,573 / 1,751 / 3,546 |
| Duplicate groups (total) | 335 |
| — crossing splits | 155 |
| — within split | 180 |
| Excluded: content crosses official splits | 333 |
| Excluded: conflicting labels within split | 309 |
| Excluded: redundant same-split copy | 30 |
| Total excluded | 672 |

The manifest also stores `official_split_sha256` (per split file), the full class mapping, per-split eligible ID lists (`eligible_by_split`), and the exclude lists (`excluded_by_split`, `excluded`). `validate_eligibility_manifest` re-derives the manifest from the untouched official files and rejects any entry whose class, split, or hash differs, any duplicate/obsolete ID, and any mismatch in the recorded split-file hashes or class mapping.

## 2. Shared eligible samples across all baselines

`ml/fusion/train.py` and `ml/fusion/evaluate.py` now require `--eligibility-manifest`. Each script loads the manifest, re-derives the eligible ID set from the official files, and filters the paired examples with `filter_paired_examples`, so image-only, text-only, and early/intermediate/late fusion all see the identical eligible sample IDs. Training records `eligible_sample_count` and `eligibility_manifest_sha256` in the checkpoint and refuses to resume against a different manifest. Evaluation reports `same_eligible_test_examples: true` and the manifest hash. Train/evaluate exit with a clear `parser.error` if the manifest file is absent.

## 3. Independent text-only baseline

`ml/fusion/text_only.py` implements `TextOnlyClassifier` (reuses the existing `TextEncoder` plus a dropout/linear head), `TextOnlyDataset`, and `train_text_only` / `load_text_only_model`. It fits vocabulary on training text only, uses the same official splits and the same eligible subset, selects on validation macro-F1, and writes a manifest sidecar (`terramind-fusion-text-only-v1`) that records class order and a checkpoint hash. `load_text_only_model` rejects untrained, altered, or class-order-mismatched checkpoints. The evaluator accepts an optional `--text-only-checkpoint` and inserts a `text_only_baseline` comparison row. The module's `main()` refuses to run while the paired CSV or eligibility manifest is missing, so **no text-only model was trained or evaluated**.

## 4. Audit of commands, registry, class mapping, checkpoint loading, inference

- `ml/fusion/registry.py`: rewritten to report fusion kinds and the text-only control separately so the control is not mislabeled as a fusion model; sidecar-only (`.manifest.json` + checksum), never deserializes experimental weights.
- `backend/app.py` `/api/models/status`: `fusion_models` restricted to `{early, intermediate, late}`; the text-only control is now exposed as `text_only_baseline`; `experimental_weights_loaded` stays `false`.
- `frontend/services/system.ts`: added optional `text_only_baseline` to `ModelRegistryStatus`.
- Class mapping, checkpoint loading, and the inference interface were unchanged; no defect required a fix beyond the text-only reporting split above.

## 5. Paired-data schema and leakage safeguards

The loader `ml/fusion/data.py` enforces: the exact column set `sample_id, image_path, text, text_origin, provenance, source_reference` (target/split columns rejected), unique `sample_id`, `sample_id == normalized image_path`, safe in-dataset image paths, approved `text_origin` only, non-empty provenance/source reference, official-derived class/split, `validate_text_not_label_derived` (exact class-name phrases rejected), and cross-split byte-identical image rejection among supplied rows. This is unchanged and still authoritative; the eligibility manifest adds the project-wide duplicate view on top of it.

## 6. Task 7 — can genuine image-specific text be obtained?

Local inspection (read-only) confirms no paired manifest exists: under `data/plantwild/` there is no `paired_observations.csv` or any image-level caption file. `readme.txt` documents the only text artifacts as `plantwild_prompts.json` (GPT-generated, 50 per **class**) and `url_record.json` (per-image download URLs, i.e. provenance, not symptom text). Neither is image-specific text.

External candidates verified in the Stage 1 report (`fusion_experiment_readiness.md`) — PlantWild card (CC BY-NC-ND 4.0; class-level prompts only), PlantDoc (visual annotations, different image corpus), PlantVillageVQA (different corpus, generated Q/A) — none provide text aligned to the PlantWild sample IDs. No dataset was downloaded, and no compatibility is claimed without evidence of image identity and licensing for these exact images.

**Minimum practical next step:** obtain independently written, image-specific notes/captions for the **eligible** PlantWild images across the existing splits (one observation per eligible `image_path`), recorded with `sample_id`, `text`, `text_origin`, `provenance`, and `source_reference`, and with assurance that the same kind of text can be collected identically at inference. Then run `scripts/audit_fusion_data.py --paired-csv ...` and the validator, and have a human verify provenance and image-text correspondence before any training. Do not synthesize observations or reuse disease labels as captions.

## 7. Leakage checks

- Class and split are derived only from official files; CSV target/split columns are rejected.
- Cross-split byte-identical images are excluded project-wide from comparison.
- Class-name-derived text, unapproved text origins, duplicate sample IDs, image-text misalignment, and unsupported/omitted eligible samples are all rejected by validation.
- Eligible manifest is re-derived and hash-checked at train and evaluate time; resume against a different manifest is refused.

## 8. Verification (actual results)

| Check | Command | Result |
|---|---|---|
| Focused tests | `python -m unittest discover -s tests -p test_multimodal.py -v` | **18 tests, OK** (includes 4 eligibility + 2 text-only tests) |
| Compilation | `python -m py_compile ...` on all changed modules | **exit 0** |
| Whitespace check | `git diff --check` | **exit 0** |
| Manifest build | `python -m scripts.build_fusion_eligibility` | **exit 0**, counts above |

Test coverage added: manifest reproducibility (two builds identical), duplicate-group exclusion (cross-split and same-split cases), class alignment against official labels, split isolation (no eligible sample changes split), manifest round-trip and tamper rejection, eligible-pair filtering, and text-only forward/input contract and class-order validation.

## 9. Decision and remaining blocker

**BLOCKED.** All scaffolding required for a valid future comparison now exists and is verified: a deterministic duplicate-free eligibility manifest, shared eligible-sample filtering across every baseline, an independent text-only control, and a registry that reports it correctly. The single remaining blocker is unchanged: **there is no valid image-specific paired text dataset** for PlantWild. Fusion and text-only models must remain untrained until such text is collected and verified.

### Changed files

- `ml/fusion/eligibility.py` — added per-split views, `comparison_rule`, `load_eligibility_manifest`, `eligible_sample_ids`
- `ml/fusion/train.py` — required `--eligibility-manifest`, eligible-sample filter, manifest hash in checkpoint/resume
- `ml/fusion/evaluate.py` — eligible-sample filter, optional `--text-only-checkpoint`, `text_only_baseline` row, generalized per-class/confusion output
- `ml/fusion/text_only.py` — **new** independent text-only baseline (guarded against missing data)
- `ml/fusion/registry.py` — report text-only control separately
- `backend/app.py` — split `fusion_models` vs `text_only_baseline`
- `frontend/services/system.ts` — optional `text_only_baseline` field
- `tests/test_multimodal.py` — eligibility/text-only tests; stale fixtures aligned to the `sample_id == image_path` contract
- `outputs/research/fusion_eligibility_manifest.json` — generated manifest
- `outputs/research/fusion_experiment_readiness_stage2a.md` — this report

## References

1. Wei et al., [Benchmarking In-the-Wild Multimodal Disease Recognition and A Versatile Baseline](https://arxiv.org/abs/2408.03120), 2024; [PlantWild dataset card](https://huggingface.co/datasets/uqtwei2/PlantWild).
2. [PlantWild project website](https://tqwei05.github.io/PlantWild/).
3. Stage 1 report: `outputs/research/fusion_experiment_readiness.md`; audit: `outputs/research/fusion_data_audit.json`.
