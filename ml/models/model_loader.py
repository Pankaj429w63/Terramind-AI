from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torchvision.models as models

from .class_labels import DEFAULT_LABELS_PATH, load_class_labels, normalize_class_labels


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "archive" / "2026-09-09_efficientnet_b0_official_v3" / "best_model.pth"
)
LEGACY_MOBILENET_PATH = PROJECT_ROOT / "models" / "final_model.pth"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EFFICIENTNET_B0_IN_FEATURES = 1280


def build_model(num_classes: int, dropout: float = 0.45) -> nn.Module:
    """Build the exact MobileNetV3-Small classifier used by full_train.py."""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 576),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(576, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p=dropout * 0.55),
        nn.Linear(256, num_classes),
    )
    return model


def build_efficientnet_b0(num_classes: int, dropout: float = 0.3) -> nn.Module:
    """Build the EfficientNet-B0 head used by mvpdr/train_efficientnet_b0.py."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def labels_from_class_to_id(class_to_id: Mapping[str, int] | Mapping[str, Any]) -> dict[int, str]:
    inverted = {int(class_id): str(name) for name, class_id in class_to_id.items()}
    labels = normalize_class_labels(inverted)
    if len(labels) != 89:
        raise ValueError(f"EfficientNet-B0 serving requires 89 classes; found {len(labels)}")
    return labels


def is_efficientnet_b0_state(state: dict[str, Any]) -> bool:
    weight = state.get("classifier.1.weight")
    return isinstance(weight, torch.Tensor) and tuple(weight.shape[1:]) == (EFFICIENTNET_B0_IN_FEATURES,)


def _companion_metrics(model_path: Path) -> dict[str, Any]:
    results_path = model_path.parent / "results.json"
    if not results_path.is_file():
        return {}
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    return {
        "test_acc": payload.get("test_acc"),
        "test_macro_f1": payload.get("test_macro_f1"),
        "best_val_macro_f1": payload.get("best_val_macro_f1"),
        "best_epoch": payload.get("best_epoch"),
    }


def load_checkpoint(
    model_path: Path = DEFAULT_MODEL_PATH,
    labels_path: Path = DEFAULT_LABELS_PATH,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the trained model state and metadata without training or changing weights."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device_obj = torch.device(device)
    payload = torch.load(model_path, map_location=device_obj, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(payload).__name__}")

    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain the required 'model_state' mapping")

    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    class_to_id = payload.get("class_to_id") if isinstance(payload.get("class_to_id"), dict) else None
    checkpoint_labels = payload.get("idx_to_class")
    if class_to_id:
        labels = labels_from_class_to_id(class_to_id)
    elif checkpoint_labels:
        labels = normalize_class_labels(checkpoint_labels)
    else:
        labels = load_class_labels(labels_path)

    num_classes = int(payload.get("num_classes") or len(labels))
    if num_classes != len(labels):
        raise ValueError(f"Checkpoint declares {num_classes} classes but {len(labels)} labels were loaded")

    efficientnet = is_efficientnet_b0_state(state)
    if efficientnet:
        if num_classes != 89:
            raise ValueError(f"EfficientNet-B0 v3 serving requires 89 classes; found {num_classes}")
        dropout = float(config.get("dropout") or 0.3)
        model = build_efficientnet_b0(num_classes=num_classes, dropout=dropout)
        backbone = "EfficientNet-B0"
        img_size = int(config.get("img_size") or payload.get("img_size") or 224)
    else:
        model = build_model(num_classes=num_classes, dropout=0.45)
        backbone = str(payload.get("backbone") or "MobileNetV3-Small")
        img_size = int(payload.get("img_size") or 160)

    model.load_state_dict(state, strict=True)
    model.to(device_obj)
    model.eval()

    companion = _companion_metrics(model_path) if efficientnet else {}
    metadata = {
        "backbone": backbone,
        "num_classes": num_classes,
        "img_size": img_size,
        "mean": list(payload.get("mean") or IMAGENET_MEAN),
        "std": list(payload.get("std") or IMAGENET_STD),
        "model_path": str(model_path),
        "test_acc": companion.get("test_acc", payload.get("test_acc")),
        "test_macro_f1": companion.get("test_macro_f1"),
        "best_val_acc": payload.get("best_val_acc"),
        "best_val_macro_f1": companion.get("best_val_macro_f1", payload.get("val_macro_f1")),
        "best_epoch": companion.get("best_epoch", payload.get("epoch")),
        "preprocessing": "efficientnet_b0_224" if efficientnet else "legacy_mobilenet",
        "labels": labels,
    }
    return model, metadata
