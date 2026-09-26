from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from ml.models.model_loader import DEFAULT_MODEL_PATH
from ml.serving.predictor import Predictor

from .data import BOS_ID, EOS_ID, metadata_tokens
from .model import MultimodalAutoencoder, MultimodalConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MULTIMODAL_CHECKPOINT = PROJECT_ROOT / "models" / "multimodal" / "terramind_multimodal.pth"
DATASET_ROOT = PROJECT_ROOT / "data" / "plantwild" / "plantwild"


def multimodal_status(checkpoint_path: Path = DEFAULT_MULTIMODAL_CHECKPOINT) -> dict[str, Any]:
    labels_path = DATASET_ROOT / "classes.txt"
    train_split = DATASET_ROOT / "train_split.txt"
    validation_split = DATASET_ROOT / "val_split.txt"

    def line_count(path: Path) -> int:
        if not path.is_file():
            return 0
        with path.open("r", encoding="utf-8") as source:
            return sum(1 for line in source if line.strip())

    checkpoint_exists = checkpoint_path.is_file()
    return {
        "module": "TerraMind multimodal autoencoder",
        "status": "unavailable" if not checkpoint_exists else "checkpoint_requires_validation",
        "trained": False,
        "checkpoint_available": checkpoint_exists,
        "checkpoint_valid": False,
        "training_completed": False,
        "architecture": {
            "image_encoder": "EfficientNet-B0 feature map (1280 channels)",
            "text_encoder": "Transformer over existing PlantWild class metadata tokens",
            "shared_latent": "image and text map to shared Z",
            "decoder": "bidirectional cross-attention with image and text reconstruction heads",
        },
        "data": {
            "class_mapping_available": labels_path.is_file(),
            "train_split_entries": line_count(train_split),
            "validation_split_entries": line_count(validation_split),
            "text_source": "exact PlantWild class labels only; no generated descriptions",
            "paired_natural_language_descriptions": False,
        },
        "production_diagnosis": "EfficientNet-B0 89-class classifier remains the production model",
        "message": "Multimodal weights have not been trained and validated; analysis is unavailable until a completed checkpoint is installed.",
    }


class MultimodalRuntime:
    def __init__(self, model: MultimodalAutoencoder, vocabulary: dict[str, int], labels: dict[int, str], device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.vocabulary = vocabulary
        self.labels = labels
        self.device = torch.device(device)

    @classmethod
    def load(
        cls,
        predictor: Predictor,
        checkpoint_path: Path = DEFAULT_MULTIMODAL_CHECKPOINT,
        device: str = "cpu",
    ) -> "MultimodalRuntime":
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(payload, dict) or not payload.get("training_state", {}).get("completed"):
            raise ValueError("Multimodal checkpoint lacks a completed-training marker")
        if not payload.get("training_state", {}).get("validated"):
            raise ValueError("Multimodal checkpoint lacks a validation-completed marker")
        labels = {int(index): str(value) for index, value in payload["class_labels"].items()}
        if labels != predictor.labels:
            raise ValueError("Multimodal checkpoint class mapping differs from the production EfficientNet mapping")
        vocabulary = {str(token): int(index) for token, index in payload["vocabulary"].items()}
        config = MultimodalConfig(**payload["model_config"])
        feature_encoder = copy.deepcopy(predictor.model.features)
        model = MultimodalAutoencoder(
            feature_encoder, len(vocabulary), feature_dim=1280, config=config
        )
        model.load_state_dict(payload["model_state"], strict=True)
        return cls(model, vocabulary, labels, device)

    def encode(self, image: Tensor, metadata_text: str) -> dict[str, Any]:
        tokens = [BOS_ID, *(self.vocabulary.get(token, self.vocabulary["<unk>"]) for token in metadata_tokens(metadata_text)), EOS_ID]
        text_ids = torch.tensor([tokens[: self.model.config.max_text_length]], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            outputs = self.model(image.to(self.device), text_ids)
        vector = outputs["fused_z"][0].detach().cpu().tolist()
        return {"dimension": len(vector), "vector": vector, "metadata_text": metadata_text}


def load_runtime_if_available(predictor: Predictor | None, checkpoint_path: Path = DEFAULT_MULTIMODAL_CHECKPOINT) -> tuple[MultimodalRuntime | None, str | None]:
    if not checkpoint_path.is_file():
        return None, "No trained multimodal checkpoint is installed."
    if predictor is None:
        return None, "The production EfficientNet-B0 feature encoder is unavailable."
    try:
        return MultimodalRuntime.load(predictor, checkpoint_path), None
    except Exception as error:
        return None, f"Multimodal checkpoint could not be loaded: {error}"
