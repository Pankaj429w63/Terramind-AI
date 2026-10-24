# TerraMind AI — Fusion Data Audit and Experiment Readiness

**Audit date:** 2026-10-17  
**Decision:** **BLOCKED — do not train fusion models**  
**Scope:** local dataset and code inspection, source/license review, split/duplicate validation, and checks of the existing fusion interfaces. No model training was started and no historical model results were changed.

## Executive finding

The local PlantWild v1 checkout has 18,542 images and official train/validation/test files for 89 classes. It has no image-specific text annotations or paired image-text CSV. `plantwild_prompts.json` contains 4,600 GPT-generated descriptions keyed by class (50 per class), not by image. These descriptions contain disease/class information and would reveal the target. `url_record.json` has one image-source URL for each image; those URLs are provenance for images, not symptom text. Neither file is valid paired text for the proposed supervised fusion experiment.

All three fusion architectures are implemented and can accept aligned image/text tensors, but there is no valid paired dataset with which to train or evaluate them. They remain untrained and unavailable for inference. The image-only production model and its checkpoint were not modified.

## Data found and provenance

| Local asset | Verified content | Fusion suitability |
|---|---|---|
| `classes.txt` | 89 class IDs and names | Target mapping only; never text input |
| `train_split.txt`, `val_split.txt`, `test_split.txt` | Official split rows; 13,045 / 1,820 / 3,677 | Retained as the only target/split authority |
| `plantwild_prompts.json` | 92 keys, 50 prompts/key, 4,600 rows; 89 keys match official labels; 4,394 unique exact case-folded strings and 206 repeated rows | Class-conditioned, label-derived text; not paired to individual images and rejected for this experiment |
| `url_record.json` | 18,542 image-path-to-URL records | Image-source provenance, not text descriptions |
| Paired annotation manifest | No candidate CSV present under the PlantWild data directory | Required text modality absent |

The local PlantWild folder has no accompanying license file. The upstream [PlantWild Hugging Face dataset card](https://huggingface.co/datasets/uqtwei2/PlantWild) identifies the dataset as CC BY-NC-ND 4.0 and describes 18,542 images across 89 classes with disease descriptions. The local `readme.txt` says the descriptions are GPT-generated, 50 per class, and supplied as textual encoder input; it also says image-source URLs are recorded separately. The [PlantWild paper](https://arxiv.org/abs/2408.03120) describes generating class-level prompts with GPT-3.5. Respect attribution, non-commercial, and no-derivatives terms stated by the source when using or sharing the dataset; the local copy does not provide separate rights metadata for each image URL.

Two public candidates were checked, but neither yields valid aligned text for the current PlantWild split:

- [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) provides visual disease annotations for a different set of web-sourced images, not text annotations for these PlantWild sample IDs. Its repository includes a CC BY 4.0 license file, but that alone does not establish rights in every underlying third-party image. It cannot create PlantWild image-text pairs.
- [PlantVillageVQA](https://huggingface.co/datasets/SyedNazmusSakib/PlantVillageVQA) is listed as CC BY 4.0 and its card shows an 871 MB archive. Its [paper](https://arxiv.org/abs/2508.17117) describes QA constructed from PlantVillage image metadata, then linguistically re-engineered. It is a different image corpus with generated Q/A, not the PlantWild samples in the official splits. It was not downloaded and cannot be joined by class label without fabricating correspondence or changing the experiment.

No external dataset was downloaded. Dataset facts above are from local files and the linked source documentation; no external data access is claimed.

## Split, duplicate, and leakage checks

The read-only audit hashed and opened all 18,542 referenced images:

| Check | Result |
|---|---:|
| Train / validation / test rows | 13,045 / 1,820 / 3,677 |
| Official classes represented per split | 89 / 89 / 89 |
| Unique paths / path overlap across split files | 18,542 / 0 |
| Missing images / image decode failures | 0 / 0 |
| Unique SHA-256 hashes | 18,175 |
| Exact duplicate image groups / extra duplicate rows | 335 / 367 |
| Exact duplicate groups crossing official splits | 155 |
| Prompt rows / repeated exact prompt rows | 4,600 / 206 |

Thus official **paths** are mutually exclusive, but byte-identical copies cross split boundaries. The paired-data loader detects duplicate bytes that occur among supplied paired rows; it does not currently create a project-wide duplicate-free eligible split manifest. Before training/evaluation, the same duplicate-free eligibility filter must be applied to image-only, text-only, and all fusion baselines so that paired test examples cannot leak through duplicate copies. The current audit lists example cross-split groups in `fusion_data_audit.json`.

Leakage controls retained in `ml/fusion/data.py` derive target class and split only from official split files, reject target/split columns in the input manifest, reject exact class-name phrases and unapproved text origins, enforce safe in-dataset image paths, and hash-check paired images across splits. These checks cannot prove how a human observation was authored; provenance requires independent curator verification. Exact duplicate text is counted by the audit when a paired CSV is supplied, but duplicate wording itself is not automatically rejected because independently observed generic statements can legitimately coincide.

## Paired-data schema and validation

The paired input CSV now requires these columns:

| Column | Meaning / validation |
|---|---|
| `sample_id` | Stable, non-empty, unique ID for the annotated sample |
| `image_path` | Relative path under the PlantWild `images/` directory; must occur in one official split |
| `text` | Non-empty, image-specific observation/caption available at real inference time; must not be target-derived |
| `text_origin` | One of `paired_human_observation`, `paired_caption`, `user_observation` |
| `provenance` | Collection or annotation method and provenance record |
| `source_reference` | Citation/URL or stable identifier for the source annotation record |

The resulting joined experiment record also carries `class_id` and `split`, derived from the official split files by `image_path`; they are not input CSV columns and are never model text. A text CSV must provide train, validation, and test examples. Vocabulary is fit on training text only. The curator must attest that text exists independently of the disease label and can be collected in the same way for evaluation and inference.

The minimum practical data task is to obtain independently written, image-specific notes/captions for PlantWild images across the existing splits, with sample ID, image path, annotation provenance/source reference, and confirmation that the same kind of text can be requested at inference. Then run the validator and have a human verify provenance and correspondence. If this dataset cannot be annotated, the alternative is to explicitly redesign the research question and evaluation dataset; do not join a different corpus to PlantWild by matching disease names.

## Existing fusion implementation and experiment protocol

Existing code reused:

- `EarlyFusionClassifier`: projected image and text representations concatenated before its classifier head.
- `IntermediateFusionClassifier`: gated neural fusion over projected image and text features.
- `LateFusionClassifier`: separate modality heads, trained jointly, with probability weights selected on validation macro-F1.
- `ml/fusion/train.py`: seeded training, training-only vocabulary, validation macro-F1 checkpoint selection, separate `outputs/fusion/` checkpoints, optional resume, weighted loss/sampler configuration, and read-only EfficientNet-B0 initialization.
- `ml/fusion/evaluate.py`: official-test evaluation, validation-selected late-fusion weight, class-order/checkpoint compatibility checks, duplicate checks over paired rows, and accuracy, macro/weighted metrics, top-3, per-class metrics, confusion matrix, parameter count, checkpoint size, and measured latency.
- `ml/fusion/registry.py`: sidecar-based availability reporting without deserializing every experimental checkpoint at application startup.

The three classifiers accept `(images, token_ids)` and output 89 ordered class logits; batches must have aligned images and texts. The image encoder is initialized from the production EfficientNet-B0 checkpoint without writing to it. Existing defaults are seed 42, up to 20 epochs, patience 5, batch size 16, AdamW at 1e-4, weight decay 1e-4, dropout 0.3, label smoothing 0.05, frozen image features, and weighted loss. The training script saves only under `outputs/fusion/`.

Before a valid comparison, use a single deterministic, duplicate-free eligible sample manifest and report the *same exact examples* for all systems. Keep the official split boundaries; fit vocabulary and model weights on train only; select checkpoints and late-fusion weights on validation macro-F1; evaluate the untouched eligible test examples once. Report accuracy, macro precision/recall/F1, weighted-F1, top-3, per-class support/precision/recall/F1, confusion matrix, parameters, artifact size, elapsed training, and measured inference latency. Required controls are image-only EfficientNet-B0, text-only classifier, and each of early/intermediate/late fusion on the same eligible subset. **The current evaluator emits fusion and paired-subset image-only results but does not yet emit an independently trained text-only control.** Add that control before claiming a complete comparison; it is not a reason to use synthetic text.

## GO / BLOCKED decision and commands

**BLOCKED.** There is no valid paired CSV. Training all fusion methods now would use class-level descriptions or fabricate per-image text, violating the leakage and correspondence requirements. The existing architecture and training commands are ready to be used after the data and text-only-control prerequisites are satisfied. No training or test evaluation was run.

Run the local, read-only data audit (writes only the named research JSON):

```powershell
python scripts/audit_fusion_data.py
```

When the paired manifest exists, validate it against the official files before any model training:

```powershell
python scripts/audit_fusion_data.py --paired-csv data/paired_observations.csv
```

After the audit returns `REVIEW_REQUIRED` and an independent human provenance review is complete, train each separate experimental model:

```powershell
python -m ml.fusion.train --kind early --paired-data data/paired_observations.csv
python -m ml.fusion.train --kind intermediate --paired-data data/paired_observations.csv
python -m ml.fusion.train --kind late --paired-data data/paired_observations.csv
```

Evaluate once per trained checkpoint after the duplicate-free eligible subset and text-only control are implemented:

```powershell
python -m ml.fusion.evaluate --checkpoint outputs/fusion/early/best_model.pth --paired-data data/paired_observations.csv
python -m ml.fusion.evaluate --checkpoint outputs/fusion/intermediate/best_model.pth --paired-data data/paired_observations.csv
python -m ml.fusion.evaluate --checkpoint outputs/fusion/late/best_model.pth --paired-data data/paired_observations.csv
```

The environment reports 12 logical CPUs, no CUDA device, 15.4 GiB total RAM, and about 5.42 GiB available at audit time. No runtime was extrapolated from this snapshot. CPU training of three 89-class models with image loading is likely to be slow; start only after the data checks pass, use the existing small batch/configuration, and monitor memory. No production checkpoint promotion is in scope.

## Verification

- `python scripts/audit_fusion_data.py` — completed; 18,542 images hashed/opened, with one benign Pillow corrupt-EXIF warning and zero decode failures; result is `BLOCKED`.
- `python -m unittest discover -s tests -p test_multimodal.py -v` — fusion/data-safeguard suite; rerun after the schema change. It exercises forward dimensions, paired alignment, official label/split joining, target leakage rejection, sample-ID/source schema, duplicate-image split protection, and registry availability.
- No training command was executed.

## References

1. Wei et al., [Benchmarking In-the-Wild Multimodal Disease Recognition and A Versatile Baseline](https://arxiv.org/abs/2408.03120), 2024; [PlantWild dataset card](https://huggingface.co/datasets/uqtwei2/PlantWild).
2. [PlantWild project website](https://tqwei05.github.io/PlantWild/).
3. Singh et al., [PlantDoc: A Dataset for Visual Plant Disease Detection](https://arxiv.org/abs/1911.10317), 2020; [repository and license](https://github.com/pratikkayal/PlantDoc-Dataset).
4. Sakib et al., [PlantVillageVQA: A Visual Question Answering Dataset for Benchmarking Vision-Language Models in Plant Science](https://arxiv.org/abs/2508.17117), 2025; [dataset card](https://huggingface.co/datasets/SyedNazmusSakib/PlantVillageVQA).
