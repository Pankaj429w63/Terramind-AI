from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.models.class_labels import DEFAULT_LABELS_PATH, load_class_labels
from ml.models.model_loader import DEFAULT_MODEL_PATH

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FUSION_ROOT = ROOT / "outputs" / "fusion"
FUSION_KINDS = ("early", "intermediate", "late")
TEXT_ONLY_KIND = "text_only"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_checkpoint_record(kind: str, name: str, checkpoint: Path, labels: dict[int, str]) -> dict[str, Any]:
    """Report a text-dependent experimental checkpoint from its sidecar only (never load weights)."""
    record: dict[str, Any] = {
        "name": name, "kind": kind, "implemented": True, "trained": False, "loaded": False,
        "available": False, "requires_text": True, "status": "unavailable",
        "checkpoint": str(checkpoint), "reason": "No validation-selected checkpoint is installed.",
    }
    sidecar = checkpoint.with_suffix(".manifest.json")
    if checkpoint.is_file() and sidecar.is_file():
        try:
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            manifest_labels = {int(key): value for key, value in manifest.get("class_labels", {}).items()}
            checksum_valid = bool(manifest.get("sha256")) and _sha256(checkpoint) == manifest["sha256"]
            compatible = manifest.get("fusion_kind") == kind and manifest_labels == labels
            trained = bool(manifest.get("trained")) and checksum_valid and compatible
            record.update({
                "trained": trained,
                "available": trained,
                "status": "available" if trained else "checkpoint_invalid",
                "validation_macro_f1": manifest.get("validation_macro_f1") if trained else None,
                "test_evaluation_completed": bool(manifest.get("test_evaluation_completed")) if trained else False,
                "reason": None if trained else "Checkpoint manifest, class order, or checksum validation failed.",
            })
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            record["reason"] = "Fusion checkpoint manifest is unreadable."
    return record


def discover_models(
    fusion_root: Path = DEFAULT_FUSION_ROOT,
    labels_path: Path = DEFAULT_LABELS_PATH,
    production_checkpoint: Path = DEFAULT_MODEL_PATH,
    production_loaded: bool = False,
) -> dict[str, Any]:
    """Read small sidecar manifests only; never deserialize experimental model weights."""
    labels = load_class_labels(labels_path) if labels_path.is_file() else {}
    entries = [{
        "name": "Image-only EfficientNet-B0", "kind": "image_only", "implemented": True,
        "trained": production_checkpoint.is_file(), "loaded": production_loaded,
        "available": production_checkpoint.is_file(), "requires_text": False,
        "status": "available" if production_checkpoint.is_file() else "checkpoint_missing",
        "checkpoint": str(production_checkpoint), "reason": None if production_checkpoint.is_file() else "Production checkpoint is missing.",
    }]
    for kind in FUSION_KINDS:
        checkpoint = Path(fusion_root) / kind / "best_model.pth"
        entries.append(_text_checkpoint_record(kind, f"{kind.title()} fusion", checkpoint, labels))
    # Independent text-only control for the fusion comparison: reported separately
    # from the fusion models because it is a single-modality baseline, not a fusion model.
    entries.append(_text_checkpoint_record(
        TEXT_ONLY_KIND, "Text-only baseline", Path(fusion_root) / TEXT_ONLY_KIND / "best_model.pth", labels,
    ))
    return {"models": entries, "experimental_weights_loaded": False}
