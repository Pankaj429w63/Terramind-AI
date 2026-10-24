# TerraMind supervised fusion classifiers

`models.py` contains three separate classifiers: early concatenation before the classifier head, intermediate gated fusion, and late probability fusion with its image/text weight selected on validation macro-F1. Each has the same `(images, token_ids) -> class logits` interface; late fusion exposes `predict_proba` and refuses inference until a validation weight is set.

## Data contract and leakage controls

The repository does not contain clean paired PlantWild text. `plantwild_prompts.json` contains class-conditioned descriptions, and `ml/multimodal/data.py` pairs exact disease class names with images. Neither can be used for supervised disease fusion. `load_paired_text_csv` requires user-supplied rows with `sample_id,image_path,text,text_origin,provenance,source_reference`. `sample_id` is unique; `source_reference` identifies the capture/annotation record. Target class and split are derived from official split files and must not be included as CSV columns or model text. The loader rejects unknown/unapproved origins, exact class-name phrases, paths outside the official split, duplicate rows/IDs, and duplicate image bytes crossing official splits. The vocabulary is fit on train text only. This provenance contract requires the supplied curator to attest that text is independently available per image; automated checks cannot establish that a symptom narrative was not written from the label.

PlantDoc supplies visual disease annotations but not paired text for the PlantWild images ([paper](https://arxiv.org/abs/1911.10317)). PlantVillageVQA provides PlantVillage image/QA pairs, but documents programmatic QA synthesis from structured labels and uses another image corpus ([dataset card](https://huggingface.co/datasets/SyedNazmusSakib/PlantVillageVQA)); it is not imported or aligned to PlantWild in this project. No external dataset is downloaded.

## Train and evaluate

Create a CSV from genuinely paired image-specific observations/captions with the columns above. `text_origin` must be one of `paired_human_observation`, `paired_caption`, or `user_observation`; `provenance` records the collection/annotation process, and `source_reference` identifies the source record. `sample_id` must be stable and unique. Do not include target or split columns, or populate the text with class names, PlantWild class prompts, or label-derived text. Official class and split assignments are joined by `image_path` from the existing split files.

```powershell
python -m ml.fusion.train --kind early --paired-data data/paired_observations.csv
python -m ml.fusion.train --kind intermediate --paired-data data/paired_observations.csv
python -m ml.fusion.train --kind late --paired-data data/paired_observations.csv
python -m ml.fusion.evaluate --checkpoint outputs/fusion/early/best_model.pth --paired-data data/paired_observations.csv
python -m ml.fusion.evaluate --checkpoint outputs/fusion/intermediate/best_model.pth --paired-data data/paired_observations.csv
python -m ml.fusion.evaluate --checkpoint outputs/fusion/late/best_model.pth --paired-data data/paired_observations.csv
```

Before any training, run `python scripts/audit_fusion_data.py --paired-csv data/paired_observations.csv`. The audit writes `outputs/research/fusion_data_audit.json`; it can validate structure, but an independent curator must still verify that the text was not created from the target and will be available at inference.

Training uses the production EfficientNet-B0 as read-only initialization, configured augmentations/weighted loss, fixed seed, train-only text vocabulary, validation macro-F1 selection, optional resume, and separate checkpoints under `outputs/fusion/`. Test evaluation is a separate command and refuses a second run by default. It reports fusion and image-only baseline metrics on the same eligible test examples, with per-class metrics, confusion matrices, top-3, parameter count, checkpoint size, and measured latency. No test examples are used to tune checkpoints or late-fusion weights.

Training is blocked in the current checkout until the paired CSV exists. The registry reads only sidecar manifests/checksums; it does not deserialize experimental weights during backend startup.

To check current hardware without starting training: `python -c "import torch,os; print('cpu_count',os.cpu_count(),'cuda_available',torch.cuda.is_available())"`. The available environment has no CUDA device; no training is launched by these implementation steps.
