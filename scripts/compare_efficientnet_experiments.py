"""Create an actual-results CSV/JSON comparison for a baseline and candidate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def row(name: str, metrics: dict, training_seconds, checkpoint_size):
    return {
        "model": name, "evaluation_subset": "duplicate-free official PlantWild test",
        "n": metrics["n"], "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"], "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"],
        "top3_accuracy": metrics["top3_accuracy"], "nll": metrics["nll"],
        "ece_15_bins": metrics["ece_15_bins"],
        "inference_mean_ms": None, "training_and_validation_wall_seconds": training_seconds,
        "checkpoint_size_bytes": checkpoint_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--baseline-metrics", type=Path, default=root / "outputs/experiments/efficientnet_b0_error_analysis/metrics.json")
    parser.add_argument("--candidate-metrics", type=Path, default=root / "outputs/experiments/efficientnet_b0_weighted_loss_v1_error_analysis/metrics.json")
    parser.add_argument("--candidate-training", type=Path, default=root / "outputs/experiments/efficientnet_b0_weighted_loss_v1/results.json")
    parser.add_argument("--baseline-checkpoint", type=Path, default=root / "models/archive/2026-09-09_efficientnet_b0_official_v3/best_model.pth")
    parser.add_argument("--candidate-checkpoint", type=Path, default=root / "outputs/experiments/efficientnet_b0_weighted_loss_v1/best_model.pth")
    parser.add_argument("--output", type=Path, default=root / "outputs/experiments/efficientnet_b0_comparison")
    args = parser.parse_args()
    baseline = read(args.baseline_metrics)
    candidate = read(args.candidate_metrics)
    training = read(args.candidate_training)
    duration = training.get("training_and_validation_wall_seconds")
    duration_basis = "trainer perf_counter" if duration is not None else None
    if duration is None:
        config_path = args.candidate_training.with_name("config.json")
        if config_path.is_file():
            duration = args.candidate_training.stat().st_mtime - config_path.stat().st_mtime
            duration_basis = "filesystem wall time from run config creation to results write"
    base_row = row("production EfficientNet-B0 v3", baseline["test_raw_duplicate_free"], None, args.baseline_checkpoint.stat().st_size)
    candidate_row = row("one-epoch class-weighted candidate", candidate["test_raw_duplicate_free"],
                        duration, args.candidate_checkpoint.stat().st_size)
    candidate_row["training_duration_basis"] = duration_basis
    base_row["training_duration_basis"] = "not recorded for production v3 run"
    base_row["inference_mean_ms"] = baseline["inference_latency_ms_per_image"]["mean"]
    candidate_row["inference_mean_ms"] = candidate["inference_latency_ms_per_image"]["mean"]
    result = {
        "protocol": {"official_split": {"train": 13045, "validation": 1820, "test": 3677},
                     "duplicate_filter": "exclude every byte-identical image path in train/validation/test",
                     "eligible": baseline["duplicate_free_eligible_counts"],
                     "selection_metric": "validation macro-F1", "test_used_for_selection": False},
        "baseline": base_row, "candidate": candidate_row,
        "candidate_validation_macro_f1": training["best_val_macro_f1"],
        "baseline_validation_macro_f1": baseline["validation_raw_duplicate_free"]["macro_f1"],
        "candidate_promoted": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(base_row))
        writer.writeheader(); writer.writerows((base_row, candidate_row))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
