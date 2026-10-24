from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import models
import yaml

from ml.models.class_labels import load_class_labels
from ml.models.model_loader import DEFAULT_MODEL_PATH, load_checkpoint
from ml.serving.preprocessing import build_inference_transform
from .data import PAD_ID, PairedImageTextDataset, build_vocabulary, collate_paired, load_paired_text_csv
from .eligibility import eligible_sample_ids, filter_paired_examples, load_eligibility_manifest
from .models import FusionConfig, LateFusionClassifier, build_fusion_model
from .text_only import load_text_only_model
from .train import DATASET_ROOT

ROOT = Path(__file__).resolve().parents[2]


def _load_manifest(checkpoint: Path) -> tuple[Path, dict]:
    path = checkpoint.with_suffix(".manifest.json")
    if not path.is_file():
        raise FileNotFoundError(f"Fusion manifest is required for registry/evaluation safety: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("trained"):
        raise ValueError("Fusion checkpoint manifest does not mark completed training")
    expected = manifest.get("sha256")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if not expected or digest != expected:
        raise ValueError("Fusion checkpoint SHA-256 does not match its manifest")
    return path, manifest


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict:
    checkpoint = args.checkpoint
    manifest_path, manifest = _load_manifest(checkpoint)
    if manifest.get("test_evaluation_completed") and not args.allow_repeat:
        raise RuntimeError("Final test evaluation is already recorded for this checkpoint; refusing a second test-set evaluation")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    labels = load_class_labels(args.dataset_root / "classes.txt")
    manifest_labels = {int(key): value for key, value in manifest.get("class_labels", {}).items()}
    if payload.get("class_labels") != labels or manifest_labels != labels:
        raise ValueError("Fusion checkpoint class ordering differs from the official class mapping")
    if payload.get("fusion_kind") != manifest.get("fusion_kind"):
        raise ValueError("Fusion kind differs between checkpoint and manifest")
    kind = payload["fusion_kind"]
    output_dir = checkpoint.parent / "evaluation"
    result_path = output_dir / "test_metrics.json"
    if result_path.exists() and not args.allow_repeat:
        raise FileExistsError(f"Test evaluation artifact already exists: {result_path}")
    split_files = {"train": args.train_split, "validation": args.validation_split, "test": args.test_split}
    pairs = load_paired_text_csv(args.paired_data, args.dataset_root, split_files, labels)
    # Evaluate only on the same duplicate-free eligible samples used for every other model.
    eligibility = load_eligibility_manifest(args.eligibility_manifest)
    eligible = eligible_sample_ids(eligibility, args.dataset_root, split_files, labels)
    pairs = filter_paired_examples(pairs, eligible)
    test_pairs = [item for item in pairs if item.split == "test"]
    train_pairs = [item for item in pairs if item.split == "train"]
    if not test_pairs or not train_pairs:
        raise ValueError("Paired data needs eligible official train and test records")
    vocabulary = payload["vocabulary"]
    train_words = build_vocabulary([item.text for item in train_pairs])
    if vocabulary != train_words:
        raise ValueError("Checkpoint vocabulary does not match training-only paired text vocabulary")
    cfg = FusionConfig(**payload["model_config"])
    image_encoder = models.efficientnet_b0(weights=None).features
    model = build_fusion_model(kind, image_encoder, cfg)
    model.load_state_dict(payload["model_state"], strict=True)
    if isinstance(model, LateFusionClassifier):
        model.validation_image_weight = payload.get("validation_image_weight")
        if model.validation_image_weight is None:
            raise ValueError("Late-fusion checkpoint lacks validation-selected modality weight")

    production, baseline_info = load_checkpoint(args.production_checkpoint, args.dataset_root / "classes.txt", "cpu")
    if baseline_info["backbone"] != "EfficientNet-B0" or baseline_info["labels"] != labels:
        raise ValueError("Image-only baseline checkpoint class order does not match fusion checkpoint")
    if payload.get("production_checkpoint_sha256"):
        baseline_sha = hashlib.sha256(args.production_checkpoint.read_bytes()).hexdigest()
        if baseline_sha != payload["production_checkpoint_sha256"]:
            raise ValueError("Production image encoder checkpoint changed since fusion training")

    device = torch.device(args.device)
    model.to(device).eval()
    production.to(device).eval()
    text_model = None
    if args.text_only_checkpoint:
        text_model, text_payload = load_text_only_model(args.text_only_checkpoint, labels)
        if text_payload.get("vocabulary") != vocabulary:
            raise ValueError("Text-only baseline was trained on a different eligible training vocabulary")
        text_model.to(device).eval()
    transform = build_inference_transform("EfficientNet-B0", 224, baseline_info["mean"], baseline_info["std"])
    dataset = PairedImageTextDataset(test_pairs, vocabulary, transform, cfg.max_text_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_paired, pin_memory=torch.cuda.is_available())
    y_true: list[int] = []
    fusion_probs: list[torch.Tensor] = []
    image_probs: list[torch.Tensor] = []
    text_probs: list[torch.Tensor] = []
    latencies: list[tuple[float, float, float]] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device)
            tokens = batch["token_ids"].to(device)
            targets = batch["class_ids"].tolist()
            start = time.perf_counter()
            if isinstance(model, LateFusionClassifier):
                predictions = model.predict_proba(images, tokens)
            else:
                predictions = model(images, tokens).softmax(1)
            fusion_time = time.perf_counter() - start
            start = time.perf_counter()
            base_predictions = production(images).softmax(1)
            baseline_time = time.perf_counter() - start
            text_time = 0.0
            if text_model is not None:
                start = time.perf_counter()
                text_predictions = text_model(tokens).softmax(1)
                text_time = time.perf_counter() - start
                text_probs.append(text_predictions.cpu())
            latencies.append((fusion_time / len(targets), baseline_time / len(targets), text_time / len(targets)))
            y_true.extend(targets)
            fusion_probs.append(predictions.cpu())
            image_probs.append(base_predictions.cpu())
    targets_np = np.asarray(y_true)
    probs = torch.cat(fusion_probs).numpy()
    base_probs = torch.cat(image_probs).numpy()

    def metrics(probabilities: np.ndarray, model_name: str, params: int, model_size: int, latency: float) -> dict:
        pred = probabilities.argmax(1)
        top_k = np.argsort(probabilities, axis=1)[:, -min(3, len(labels)):]
        per_class = classification_report(targets_np, pred, labels=list(range(len(labels))),
                                          target_names=[labels[i] for i in range(len(labels))],
                                          output_dict=True, zero_division=0)
        matrix = confusion_matrix(targets_np, pred, labels=list(range(len(labels))))
        return {
            "model": model_name,
            "test_examples": len(targets_np),
            "test_accuracy": float(accuracy_score(targets_np, pred)),
            "macro_precision": float(precision_score(targets_np, pred, average="macro", labels=list(labels), zero_division=0)),
            "macro_recall": float(recall_score(targets_np, pred, average="macro", labels=list(labels), zero_division=0)),
            "macro_f1": float(f1_score(targets_np, pred, average="macro", labels=list(labels), zero_division=0)),
            "weighted_f1": float(f1_score(targets_np, pred, average="weighted", labels=list(labels), zero_division=0)),
            "top3_accuracy": float(np.mean([target in row for target, row in zip(targets_np, top_k)])),
            "parameter_count": params, "model_size_bytes": model_size,
            "inference_latency_ms_per_image": latency,
            "per_class": per_class,
            "confusion_matrix": matrix.tolist(),
        }

    fusion_parameters = sum(parameter.numel() for parameter in model.parameters())
    baseline_parameters = sum(parameter.numel() for parameter in production.parameters())
    fusion_result = metrics(probs, f"{kind.title()} fusion", fusion_parameters,
                            checkpoint.stat().st_size, 1000 * float(np.mean([r[0] for r in latencies])))
    baseline_result = metrics(base_probs, "Image-only EfficientNet-B0 baseline", baseline_parameters,
                              args.production_checkpoint.stat().st_size,
                              1000 * float(np.mean([r[1] for r in latencies])))
    result = {
        "evaluation_split": "official test; no test-based selection",
        "same_eligible_test_examples": True,
        "test_examples": len(test_pairs),
        "eligibility_manifest_sha256": hashlib.sha256(args.eligibility_manifest.read_bytes()).hexdigest(),
        "split_counts": payload["split_counts"],
        "fusion": fusion_result,
        "image_only_baseline": baseline_result,
        "late_fusion_validation_image_weight": payload.get("validation_image_weight"),
        "seed": payload["seed"],
        "training_duration_seconds": payload.get("training_duration_seconds"),
        "text_origins": payload["text_origins"],
        "modality_provenance": "paired-data CSV; labels only from official split; vocabulary fit on train text",
    }
    comparison_keys = ["image_only_baseline", "fusion"]
    if text_model is not None:
        text_parameters = sum(parameter.numel() for parameter in text_model.parameters())
        result["text_only_baseline"] = metrics(
            torch.cat(text_probs).numpy(), "Text-only baseline", text_parameters,
            args.text_only_checkpoint.stat().st_size, 1000 * float(np.mean([r[2] for r in latencies])),
        )
        comparison_keys.insert(1, "text_only_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    per_class_rows = []
    for key in comparison_keys:
        for label_id, label in enumerate(labels.values()):
            metric = result[key]["per_class"][label]
            per_class_rows.append({"model": result[key]["model"], "class_id": label_id, "class_name": label,
                                   "precision": metric["precision"], "recall": metric["recall"],
                                   "f1": metric.get("f1", metric.get("f1-score", 0.0)), "support": int(metric["support"])})
    _write_csv(output_dir / "per_class_metrics.csv", list(per_class_rows[0]), per_class_rows)
    for key in comparison_keys:
        display = "image_only_baseline" if key == "image_only_baseline" else key
        _write_csv(output_dir / f"{display}_confusion_matrix.csv", ["true_class_id", *[str(i) for i in range(len(labels))]],
                   [{"true_class_id": i, **{str(j): value for j, value in enumerate(row)}}
                    for i, row in enumerate(result[key]["confusion_matrix"])])
    manifest["test_evaluation_completed"] = True
    manifest["test_metrics_path"] = str(result_path.resolve())
    manifest["test_evaluation_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {"fusion": {k: v for k, v in fusion_result.items() if k not in {"per_class", "confusion_matrix"}},
               "baseline": {k: v for k, v in baseline_result.items() if k not in {"per_class", "confusion_matrix"}},
               "results": str(result_path)}
    if "text_only_baseline" in result:
        summary["text_only"] = {k: v for k, v in result["text_only_baseline"].items()
                                if k not in {"per_class", "confusion_matrix"}}
    print(json.dumps(summary, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one trained fusion model once on the official held-out test split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--paired-data", type=Path, required=True)
    parser.add_argument("--eligibility-manifest", type=Path, required=True,
                        help="Duplicate-free eligible-sample manifest from scripts/build_fusion_eligibility.py")
    parser.add_argument("--text-only-checkpoint", type=Path, default=None,
                        help="Optional independent text-only baseline trained on the same eligible samples")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--train-split", type=Path, default=Path("train_split.txt"))
    parser.add_argument("--validation-split", type=Path, default=Path("val_split.txt"))
    parser.add_argument("--test-split", type=Path, default=Path("test_split.txt"))
    parser.add_argument("--production-checkpoint", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-repeat", action="store_true", help="Explicitly permit repeated test inspection")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
