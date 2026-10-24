"""Measure the archived EfficientNet-B0 on PlantWild's official validation and test splits.

This is diagnostic only: it never writes into models/ and fits calibration on validation
logits only. The untouched test split is evaluated once after temperature fitting.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.models.model_loader import DEFAULT_MODEL_PATH, load_checkpoint
from mvpdr.train_efficientnet_b0 import (
    DATASET_ROOT, PlantWildDataset, build_eval_transform, load_classes, load_official_split,
)

OUT = ROOT / "outputs" / "experiments" / "efficientnet_b0_error_analysis"


@torch.inference_mode()
def logits_for(model: torch.nn.Module, lines: list[str], labels: dict[str, int], device: torch.device,
               batch_size: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = PlantWildDataset(lines, labels, build_eval_transform(224))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    logits, targets = [], []
    model.eval()
    for images, y in loader:
        logits.append(model(images.to(device)).cpu())
        targets.append(y)
    return torch.cat(logits).numpy(), torch.cat(targets).numpy()


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    x = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(targets, dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_t.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(x / temperature, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    total = len(targets)
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if mask.any():
            value += mask.sum() / total * abs(float((prediction[mask] == targets[mask]).mean()) - float(confidence[mask].mean()))
    return value


def split_duplicate_audit(lines: dict[str, list[str]]) -> tuple[dict[str, object], set[str]]:
    split_by_path: dict[str, str] = {}
    for split, entries in lines.items():
        for raw in entries:
            rel = raw.rsplit("=", 2)[0].replace("\\", "/")
            split_by_path[rel] = split
    paths_by_digest: dict[str, list[str]] = {}
    for rel in split_by_path:
        path = DATASET_ROOT / "images" / rel
        import hashlib
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        paths_by_digest.setdefault(digest, []).append(rel)
    duplicate_groups = [paths for paths in paths_by_digest.values() if len(paths) > 1]
    cross = [paths for paths in duplicate_groups if len({split_by_path[p] for p in paths}) > 1]
    internal = sum(len(paths) - 1 for paths in paths_by_digest.values() if len(paths) > 1)
    excluded = {path for paths in duplicate_groups for path in paths}
    return {"unique_image_paths": len(split_by_path), "duplicate_content_groups": len(duplicate_groups),
            "duplicate_content_extra_files": internal, "cross_split_duplicate_groups": len(cross),
            "cross_split_paths_by_split": {split: sum(split_by_path[path] == split for paths in cross for path in paths)
                                            for split in lines},
            "all_duplicate_paths_by_split": {split: sum(split_by_path[path] == split for paths in duplicate_groups for path in paths)
                                              for split in lines},
            "cross_split_examples": cross[:10]}, excluded


def summarize(logits: np.ndarray, targets: np.ndarray, labels: dict[int, str], temperature: float) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float64) / temperature, dim=1).numpy()
    predictions = probs.argmax(axis=1)
    count = len(labels)
    names = [labels[index] for index in range(count)]
    report = classification_report(targets, predictions, labels=list(range(count)), target_names=names,
                                   output_dict=True, zero_division=0)
    metrics = {
        "n": int(len(targets)), "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(precision_score(targets, predictions, labels=list(range(count)), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, predictions, labels=list(range(count)), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, labels=list(range(count)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, labels=list(range(count)), average="weighted", zero_division=0)),
        "top3_accuracy": float(np.mean([target in row for target, row in zip(targets, np.argsort(probs, axis=1)[:, -3:])])),
        "nll": float(log_loss(targets, probs, labels=list(range(count)))), "ece_15_bins": ece(probs, targets),
        "mean_max_probability": float(probs.max(axis=1).mean()), "temperature": temperature,
    }
    classes = [{"class_id": i, "class_name": labels[i], "support": int(report[labels[i]]["support"]),
                "precision": report[labels[i]]["precision"], "recall": report[labels[i]]["recall"], "f1": report[labels[i]]["f1-score"]}
               for i in range(count)]
    return metrics, classes, confusion_matrix(targets, predictions, labels=list(range(count)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    global OUT
    default_output = ROOT / "outputs" / "experiments" / "efficientnet_b0_error_analysis" if args.checkpoint == DEFAULT_MODEL_PATH else ROOT / "outputs" / "experiments" / f"{args.checkpoint.parent.name}_error_analysis"
    OUT = args.output_dir or default_output
    torch.set_num_threads(args.threads)
    class_to_id = load_classes()
    train, val, test, audit = load_official_split(class_to_id)
    groups = {"train": train, "validation": val, "test": test}
    class_counts = {split: dict(sorted(Counter(int(row.rsplit("=", 2)[1]) for row in entries).items())) for split, entries in groups.items()}
    model, metadata = load_checkpoint(args.checkpoint, device="cpu")
    expected = {index: name for name, index in class_to_id.items()}
    if metadata["labels"] != expected:
        raise RuntimeError("Production checkpoint class ordering differs from PlantWild classes.txt")
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / ("baseline_logits.npz" if args.checkpoint == DEFAULT_MODEL_PATH else f"{args.checkpoint.parent.name}_split_logits.npz")
    start = time.perf_counter()
    if cache.exists():
        stored = np.load(cache)
        if str(stored["checkpoint_sha256"].item()) != __import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest():
            raise RuntimeError("Cached logits belong to a different checkpoint")
        val_logits, val_targets = stored["validation_logits"], stored["validation_targets"]
        test_logits, test_targets = stored["test_logits"], stored["test_targets"]
        print(f"Using cached official split logits from {cache}", flush=True)
    else:
        print("Evaluating official validation split", flush=True)
        val_logits, val_targets = logits_for(model, val, class_to_id, torch.device("cpu"), args.batch_size, args.workers)
    duplicate_audit, duplicate_paths = split_duplicate_audit(groups)
    split_paths = {split: [row.rsplit("=", 2)[0].replace("\\", "/") for row in entries]
                   for split, entries in groups.items()}
    val_keep = np.array([path not in duplicate_paths for path in split_paths["validation"]])
    test_keep = np.array([path not in duplicate_paths for path in split_paths["test"]])
    if not val_keep.any() or not test_keep.any():
        raise RuntimeError("Duplicate filtering removed an entire official evaluation split")
    temperature = fit_temperature(val_logits[val_keep], val_targets[val_keep])
    if not cache.exists():
        print("Evaluating official test split", flush=True)
        test_logits, test_targets = logits_for(model, test, class_to_id, torch.device("cpu"), args.batch_size, args.workers)
        np.savez_compressed(cache, checkpoint_sha256=__import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest(),
                            validation_logits=val_logits, validation_targets=val_targets,
                            test_logits=test_logits, test_targets=test_targets)
    elapsed = time.perf_counter() - start
    inference_transform = build_eval_transform(224)
    latency_samples = []
    for raw in test[:23]:
        relative = raw.rsplit("=", 2)[0].replace("\\", "/")
        started = time.perf_counter()
        with Image.open(DATASET_ROOT / "images" / relative) as image:
            tensor = inference_transform(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            model(tensor)
        if len(latency_samples) >= 3:
            latency_samples.append((time.perf_counter() - started) * 1000)
        else:
            latency_samples.append((time.perf_counter() - started) * 1000)
    measured_latency = latency_samples[3:] if len(latency_samples) > 3 else latency_samples
    id_to_name = expected
    raw_official, _, _ = summarize(test_logits, test_targets, id_to_name, 1.0)
    validation_raw_duplicate_free, _, _ = summarize(val_logits[val_keep], val_targets[val_keep], id_to_name, 1.0)
    raw, per_class, cm = summarize(test_logits[test_keep], test_targets[test_keep], id_to_name, 1.0)
    calibrated, calibrated_classes, calibrated_cm = summarize(test_logits[test_keep], test_targets[test_keep], id_to_name, temperature)
    val_before_probs = torch.softmax(torch.tensor(val_logits[val_keep], dtype=torch.float64), dim=1).numpy()
    val_after_probs = torch.softmax(torch.tensor(val_logits[val_keep], dtype=torch.float64) / temperature, dim=1).numpy()
    val_nll_before = float(log_loss(val_targets[val_keep], val_before_probs, labels=list(range(89))))
    val_nll_after = float(log_loss(val_targets[val_keep], val_after_probs, labels=list(range(89))))
    val_ece_before = ece(val_before_probs, val_targets[val_keep])
    val_ece_after = ece(val_after_probs, val_targets[val_keep])
    calibration_payload = {
        "format": "terramind-temperature-scaling-v1", "checkpoint_sha256": __import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest(),
        "method": "temperature scaling", "temperature": temperature,
        "fit_split": "duplicate-free official validation", "fit_samples": int(val_keep.sum()),
        "validation_nll_before": val_nll_before, "validation_nll_after": val_nll_after,
        "validation_ece_15_bins_before": val_ece_before, "validation_ece_15_bins_after": val_ece_after,
    }
    if val_nll_after < val_nll_before and val_ece_after < val_ece_before:
        (OUT / "calibration.json").write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")
        if args.checkpoint == DEFAULT_MODEL_PATH:
            (args.checkpoint.parent / "confidence_calibration.json").write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")
    companion_results_path = args.checkpoint.parent / "results.json"
    stored_metrics = json.loads(companion_results_path.read_text(encoding="utf-8")) if companion_results_path.is_file() else None
    pairs = []
    for actual in range(89):
        for predicted in range(89):
            if actual != predicted and cm[actual, predicted]:
                pairs.append({"actual_id": actual, "actual": id_to_name[actual], "predicted_id": predicted,
                              "predicted": id_to_name[predicted], "count": int(cm[actual, predicted])})
    pairs.sort(key=lambda item: (-item["count"], item["actual_id"], item["predicted_id"]))
    class_name_lower = {i: name.casefold() for i, name in id_to_name.items()}
    healthy_ids = [i for i, name in class_name_lower.items() if "healthy" in name]
    disease_ids = [i for i in range(89) if i not in healthy_ids]
    labels_np = test_targets[test_keep]
    pred_np = test_logits[test_keep].argmax(axis=1)
    subset = {"healthy_class_ids": healthy_ids, "diseased_class_count": len(disease_ids)}
    if healthy_ids:
        healthy_mask = np.isin(labels_np, healthy_ids)
        disease_mask = np.isin(labels_np, disease_ids)
        subset.update({"healthy_sample_count": int(healthy_mask.sum()), "healthy_recall": float(np.mean(pred_np[healthy_mask] == labels_np[healthy_mask])) if healthy_mask.any() else None,
                       "diseased_sample_count": int(disease_mask.sum()), "diseased_recall": float(np.mean(pred_np[disease_mask] == labels_np[disease_mask])) if disease_mask.any() else None})
    (OUT / "metrics.json").write_text(json.dumps({"checkpoint": str(args.checkpoint), "checkpoint_sha256": __import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest(),
        "class_order_matches": True, "official_split_audit": audit, "class_counts": class_counts,
        "checkpoint_companion_metrics": stored_metrics,
        "split_content_duplicate_audit": duplicate_audit, "test_raw_official_all_rows": raw_official,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "inference_latency_ms_per_image": {"n": len(measured_latency), "mean": float(np.mean(measured_latency)),
                                            "median": float(np.median(measured_latency)), "device": "cpu",
                                            "includes_image_decode_and_transform": True},
        "duplicate_free_eligible_counts": {"validation": int(val_keep.sum()), "test": int(test_keep.sum()),
                                            "excluded_validation_duplicates": int((~val_keep).sum()),
                                            "excluded_test_duplicates": int((~test_keep).sum())},
        "validation_raw_duplicate_free": validation_raw_duplicate_free,
        "test_raw_duplicate_free": raw,
        "calibration": {"method": "temperature scaling", "fit_split": "duplicate-free validation only",
                        "validation_samples": int(val_keep.sum()), "temperature": temperature,
                        "test": calibrated, "validation_nll_before": val_nll_before,
                        "validation_nll_after": val_nll_after,
                        "validation_ece_15_bins_before": val_ece_before, "validation_ece_15_bins_after": val_ece_after,
                        "sidecar_selected": val_nll_after < val_nll_before and val_ece_after < val_ece_before},
        "healthy_diseased": subset, "top_confusion_pairs": pairs[:30], "elapsed_eval_seconds": elapsed,
        "inference_seconds_per_image": elapsed / (len(val) + len(test))}, indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value)), encoding="utf-8")
    for filename, rows in (("per_class_metrics.csv", per_class), ("per_class_metrics_calibrated.csv", calibrated_classes)):
        with (OUT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    np.savetxt(OUT / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    np.savetxt(OUT / "confusion_matrix_calibrated.csv", calibrated_cm, fmt="%d", delimiter=",")
    print(json.dumps({"output": str(OUT), "test_raw": raw, "temperature": temperature,
                      "test_calibrated": calibrated, "healthy_diseased": subset,
                      "top_confusion_pairs": pairs[:10], "elapsed_eval_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
