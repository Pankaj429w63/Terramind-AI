"""Render plots from persisted EfficientNet-B0 measurements (no model inference)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ("production", ROOT / "outputs/experiments/efficientnet_b0_error_analysis",
     ROOT / "models/archive/2026-09-09_efficientnet_b0_official_v3"),
    ("candidate", ROOT / "outputs/experiments/efficientnet_b0_weighted_loss_v1_error_analysis",
     ROOT / "outputs/experiments/efficientnet_b0_weighted_loss_v1"),
)


def main() -> None:
    for name, analysis, run in RUNS:
        result = json.loads((run / "results.json").read_text(encoding="utf-8"))
        history = result.get("history", [])
        if history:
            epochs = [item["epoch"] for item in history]
            fig, axis = plt.subplots(figsize=(8, 4.8))
            axis.plot(epochs, [item["train_loss"] for item in history], label="Train loss")
            axis.plot(epochs, [item["val_loss"] for item in history], label="Validation loss")
            axis.set(xlabel="Epoch", ylabel="Loss", title=f"EfficientNet-B0 {name} training history")
            axis.legend()
            fig.tight_layout()
            fig.savefig(analysis / "training_curves.png", dpi=130)
            plt.close(fig)

        matrix = np.loadtxt(analysis / "confusion_matrix.csv", delimiter=",", dtype=int)
        with (analysis / "per_class_metrics.csv").open(newline="", encoding="utf-8") as source:
            labels = [item["class_name"] for item in csv.DictReader(source)]
        if matrix.shape != (len(labels), len(labels)):
            raise SystemExit(f"{name}: matrix/label ordering mismatch: {matrix.shape}, {len(labels)} labels")
        fig, axis = plt.subplots(figsize=(10, 8))
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", aspect="auto")
        axis.set(title=f"EfficientNet-B0 {name} — duplicate-free official test confusion matrix",
                 xlabel="Predicted class ID", ylabel="True class ID")
        axis.set_xticks(range(0, len(labels), 5))
        axis.set_yticks(range(0, len(labels), 5))
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Images")
        fig.tight_layout()
        fig.savefig(analysis / "confusion_matrix.png", dpi=130)
        plt.close(fig)
        print(f"Rendered {name} plots from saved metrics in {analysis}")


if __name__ == "__main__":
    main()
