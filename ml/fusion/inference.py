"""Selectable fusion inference runtime for the three fusion architectures.

Loads exactly one trained fusion checkpoint (``early``, ``intermediate`` or
``late``) from its sidecar manifest, validates the checkpoint checksum and class
ordering, and serves a single prediction.

The runtime always reports which modalities actually ran. A ``late`` fusion
model has a genuine image-only branch, so when text is unavailable its result is
labelled ``modalities_used == ["image"]`` with ``image_only_fallback == True``.
The ``early`` and ``intermediate`` architectures have no image-only branch, so a
missing-text request raises :class:`MissingTextError` instead of silently
returning a multimodal-looking prediction.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torchvision import models

from ml.models.class_labels import DEFAULT_LABELS_PATH, load_class_labels
from ml.models.model_loader import IMAGENET_MEAN, IMAGENET_STD
from ml.serving.predictor import Predictor
from ml.serving.preprocessing import build_inference_transform
from .data import encode_text
from .models import FusionClassifierBase, FusionConfig, LateFusionClassifier, build_fusion_model

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FUSION_ROOT = ROOT / "outputs" / "fusion"
SUPPORTED_FUSION_KINDS = ("early", "intermediate", "late")
CHECKPOINT_FORMAT = "terramind-fusion-v1"


class FusionInferenceError(RuntimeError):
    """Base error for fusion checkpoint loading and inference problems."""


class FusionCheckpointUnavailable(FusionInferenceError):
    """The requested fusion checkpoint is missing, untrained, or invalid."""


class MissingTextError(FusionInferenceError):
    """A text-requiring fusion model was asked to run without text."""


def default_checkpoint(kind: str, fusion_root: Path = DEFAULT_FUSION_ROOT) -> Path:
    if kind not in SUPPORTED_FUSION_KINDS:
        raise ValueError(f"Unknown fusion kind: {kind!r}; expected one of {SUPPORTED_FUSION_KINDS}")
    return Path(fusion_root) / kind / "best_model.pth"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fusion_checkpoint(
    kind: str, checkpoint: Path, labels: Mapping[int, str]
) -> tuple[FusionClassifierBase, dict]:
    """Load one fusion checkpoint, validating its manifest, checksum, and class order."""
    if kind not in SUPPORTED_FUSION_KINDS:
        raise ValueError(f"Unknown fusion kind: {kind!r}; expected one of {SUPPORTED_FUSION_KINDS}")
    checkpoint = Path(checkpoint)
    manifest_path = checkpoint.with_suffix(".manifest.json")
    if not checkpoint.is_file():
        raise FusionCheckpointUnavailable(f"Fusion checkpoint is not installed: {checkpoint}")
    if not manifest_path.is_file():
        raise FusionCheckpointUnavailable(f"Fusion manifest is required for safe loading: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("trained"):
        raise FusionCheckpointUnavailable("Fusion checkpoint manifest does not mark completed training")
    if manifest.get("fusion_kind") != kind:
        raise FusionInferenceError("Fusion kind differs between the request and the checkpoint manifest")
    expected = manifest.get("sha256")
    if not expected or _sha256(checkpoint) != expected:
        raise FusionInferenceError("Fusion checkpoint SHA-256 does not match its manifest")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("fusion_kind") != kind:
        raise FusionInferenceError("Fusion checkpoint format is unsupported")
    if payload.get("class_labels") != dict(labels):
        raise FusionInferenceError("Fusion checkpoint class ordering differs from the official class mapping")
    config = FusionConfig(**payload["model_config"])
    image_encoder = models.efficientnet_b0(weights=None).features
    model = build_fusion_model(kind, image_encoder, config)
    model.load_state_dict(payload["model_state"], strict=True)
    if isinstance(model, LateFusionClassifier):
        model.validation_image_weight = payload.get("validation_image_weight")
        if model.validation_image_weight is None:
            raise FusionInferenceError("Late-fusion checkpoint lacks a validation-selected modality weight")
    model.eval()
    return model, payload


class FusionInferenceRuntime:
    """Load and serve exactly one fusion model with explicit modality reporting."""

    def __init__(
        self,
        kind: str,
        checkpoint: Path,
        labels: Mapping[int, str],
        device: str | torch.device = "cpu",
        *,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        image_size: int = 224,
    ) -> None:
        if kind not in SUPPORTED_FUSION_KINDS:
            raise ValueError(f"Unknown fusion kind: {kind!r}; expected one of {SUPPORTED_FUSION_KINDS}")
        self.kind = kind
        self.checkpoint = Path(checkpoint)
        self.labels: dict[int, str] = {int(index): str(label) for index, label in labels.items()}
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")
        self.device = requested
        self.model, self.payload = load_fusion_checkpoint(kind, self.checkpoint, self.labels)
        self.model.to(self.device)
        self.vocabulary: dict[str, int] = dict(self.payload["vocabulary"])
        self.max_text_length = int(self.payload["model_config"]["max_text_length"])
        self.transform = build_inference_transform(
            "EfficientNet-B0", image_size, mean if mean is not None else IMAGENET_MEAN,
            std if std is not None else IMAGENET_STD,
        )

    # -- construction helpers -------------------------------------------------
    @classmethod
    def load(
        cls,
        kind: str,
        checkpoint: Path | None = None,
        labels_path: Path = DEFAULT_LABELS_PATH,
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> "FusionInferenceRuntime":
        resolved = Path(checkpoint) if checkpoint is not None else default_checkpoint(kind)
        return cls(kind, resolved, load_class_labels(labels_path), device, **kwargs)

    @property
    def supports_image_only(self) -> bool:
        """Only late fusion has a genuine image-only prediction branch."""
        return self.kind == "late"

    @property
    def requires_text(self) -> bool:
        return not self.supports_image_only

    def status(self) -> dict:
        return {
            "kind": self.kind,
            "implemented": True,
            "trained": True,
            "loaded": True,
            "available": True,
            "requires_text": self.requires_text,
            "supports_image_only_fallback": self.supports_image_only,
            "checkpoint": str(self.checkpoint.resolve()),
            "class_count": len(self.labels),
            "validation_macro_f1": self.payload.get("best_validation_macro_f1"),
            "text_origins": list(self.payload.get("text_origins") or []),
            "production_checkpoint_sha256": self.payload.get("production_checkpoint_sha256"),
        }

    # -- inference ------------------------------------------------------------
    def _image_tensor(self, image) -> Tensor:
        pil = Predictor._to_pil(image)
        return self.transform(pil).unsqueeze(0).to(self.device)

    def _token_tensor(self, text: str) -> Tensor:
        ids = encode_text(text, self.vocabulary, self.max_text_length)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _rank(self, probabilities: Tensor, top_k: int) -> list[dict]:
        probs = probabilities.squeeze(0)
        k = max(1, min(int(top_k), len(self.labels)))
        scores, indexes = torch.topk(probs, k=k)
        return [
            {"class_id": int(index), "label": self.labels[int(index)], "confidence": round(float(score), 6)}
            for score, index in zip(scores, indexes)
        ]

    def predict(self, image, text: str | None = None, top_k: int = 5) -> dict:
        """Run one prediction and report exactly which modalities were used."""
        has_text = bool(text and text.strip())
        image_tensor = self._image_tensor(image)
        started = time.perf_counter()
        with torch.inference_mode():
            if has_text:
                token_tensor = self._token_tensor(text.strip())
                if isinstance(self.model, LateFusionClassifier):
                    probabilities = self.model.predict_proba(image_tensor, token_tensor)
                else:
                    probabilities = self.model(image_tensor, token_tensor).softmax(dim=1)
                modalities_used = ["image", "text"]
                fallback = False
            else:
                if not self.supports_image_only:
                    raise MissingTextError(
                        f"{self.kind.title()} fusion has no image-only branch and requires text; "
                        "provide text or use the production image-only model."
                    )
                image_logits = self.model.image_classifier(
                    self.model.dropout(self.model.encode_image(image_tensor))
                )
                probabilities = image_logits.softmax(dim=1)
                modalities_used = ["image"]
                fallback = True
        latency_ms = (time.perf_counter() - started) * 1000
        ranking = self._rank(probabilities, top_k)
        return {
            "fusion_kind": self.kind,
            "modalities_used": modalities_used,
            "multimodal": modalities_used == ["image", "text"],
            "image_only_fallback": fallback,
            "text_provided": has_text,
            "predicted_class": ranking[0],
            "confidence": ranking[0]["confidence"],
            "top_k": ranking,
            "inference_ms": round(latency_ms, 2),
        }


def fusion_runtime_status(
    fusion_root: Path = DEFAULT_FUSION_ROOT, labels_path: Path = DEFAULT_LABELS_PATH
) -> list[dict]:
    """Report per-kind availability from sidecar manifests without loading weights."""
    labels = load_class_labels(labels_path) if Path(labels_path).is_file() else {}
    report: list[dict] = []
    for kind in SUPPORTED_FUSION_KINDS:
        checkpoint = default_checkpoint(kind, fusion_root)
        entry = {
            "kind": kind, "implemented": True, "trained": False, "available": False,
            "requires_text": kind != "late", "status": "unavailable",
            "reason": "No validation-selected checkpoint is installed.",
        }
        sidecar = checkpoint.with_suffix(".manifest.json")
        if checkpoint.is_file() and sidecar.is_file():
            try:
                manifest = json.loads(sidecar.read_text(encoding="utf-8"))
                manifest_labels = {int(key): value for key, value in manifest.get("class_labels", {}).items()}
                trained = bool(manifest.get("trained")) and bool(manifest.get("sha256")) and (
                    _sha256(checkpoint) == manifest["sha256"]
                ) and manifest.get("fusion_kind") == kind and manifest_labels == labels
                entry.update({
                    "trained": trained,
                    "available": trained,
                    "status": "available" if trained else "checkpoint_invalid",
                    "reason": None if trained else "Checkpoint manifest, class order, or checksum validation failed.",
                })
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                entry["reason"] = "Fusion checkpoint manifest is unreadable."
        report.append(entry)
    return report
