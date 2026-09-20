from __future__ import annotations

import io
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ml.models.model_loader import DEFAULT_LABELS_PATH, DEFAULT_MODEL_PATH, load_checkpoint
from ml.serving.preprocessing import build_inference_transform, prepare_image


ImageInput = Union[str, Path, bytes, bytearray, Image.Image, np.ndarray]


@dataclass(frozen=True)
class Prediction:
    class_id: int
    label: str
    confidence: float


@dataclass(frozen=True)
class InferenceResult:
    predicted_class: Prediction
    top5: list[Prediction]
    model_info: dict[str, object]
    inference_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "predicted_class": asdict(self.predicted_class),
            "top5": [asdict(item) for item in self.top5],
            "confidence": self.predicted_class.confidence,
            "model_info": self.model_info,
            "inference_ms": self.inference_ms,
        }


class Predictor:
    """CPU-safe predictor backed by the archived EfficientNet-B0 v3 checkpoint."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        labels_path: Path = DEFAULT_LABELS_PATH,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model, metadata = load_checkpoint(model_path, labels_path, self.device)
        self.labels: dict[int, str] = metadata.pop("labels")  # type: ignore[assignment]
        self.model_info = metadata
        self.transform = build_inference_transform(
            str(metadata["backbone"]), int(metadata["img_size"]), metadata["mean"], metadata["std"]  # type: ignore[arg-type]
        )
        if str(metadata["backbone"]) == "EfficientNet-B0" and (
            int(metadata["num_classes"]) != 89 or len(self.labels) != 89
        ):
            raise RuntimeError("EfficientNet-B0 production serving requires the 89-class PlantWild mapping.")

    @staticmethod
    def _to_pil(source: ImageInput) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, (str, Path)):
            with Image.open(source) as image:
                return image.convert("RGB")
        if isinstance(source, (bytes, bytearray)):
            with Image.open(io.BytesIO(bytes(source))) as image:
                return image.convert("RGB")
        if isinstance(source, np.ndarray):
            return Image.fromarray(source).convert("RGB")
        raise TypeError(f"Unsupported image input type: {type(source).__name__}")

    def predict(self, source: ImageInput, top_k: int = 5) -> InferenceResult:
        if not 1 <= top_k <= len(self.labels):
            raise ValueError(f"top_k must be between 1 and {len(self.labels)}")

        image = self._to_pil(source)
        tensor = prepare_image(image, self.transform).unsqueeze(0).to(self.device)
        if str(self.model_info.get("backbone")) == "EfficientNet-B0" and tuple(tensor.shape[-2:]) != (224, 224):
            raise RuntimeError(f"EfficientNet-B0 inference requires a 224px tensor, got {tuple(tensor.shape)}")
        started = time.perf_counter()
        with torch.inference_mode():
            probabilities = F.softmax(self.model(tensor), dim=1)[0].cpu()
        elapsed_ms = (time.perf_counter() - started) * 1000

        scores, indexes = torch.topk(probabilities, k=top_k)
        predictions = [
            Prediction(int(index), self.labels[int(index)], round(float(score), 6))
            for score, index in zip(scores, indexes)
        ]
        return InferenceResult(
            predicted_class=predictions[0],
            top5=predictions,
            model_info=dict(self.model_info),
            inference_ms=round(elapsed_ms, 2),
        )

    def predict_batch(self, sources: list[ImageInput], top_k: int = 5) -> list[InferenceResult]:
        if not sources:
            return []
        if not 1 <= top_k <= len(self.labels):
            raise ValueError(f"top_k must be between 1 and {len(self.labels)}")

        tensors = [prepare_image(self._to_pil(source), self.transform) for source in sources]
        batch = torch.stack(tensors).to(self.device)
        if str(self.model_info.get("backbone")) == "EfficientNet-B0" and tuple(batch.shape[-2:]) != (224, 224):
            raise RuntimeError(f"EfficientNet-B0 inference requires a 224px tensor, got {tuple(batch.shape)}")
        started = time.perf_counter()
        with torch.inference_mode():
            probabilities = F.softmax(self.model(batch), dim=1).cpu()
        elapsed_ms = (time.perf_counter() - started) * 1000 / len(sources)

        results: list[InferenceResult] = []
        for row in probabilities:
            scores, indexes = torch.topk(row, k=top_k)
            predictions = [
                Prediction(int(index), self.labels[int(index)], round(float(score), 6))
                for score, index in zip(scores, indexes)
            ]
            results.append(
                InferenceResult(
                    predicted_class=predictions[0],
                    top5=predictions,
                    model_info=dict(self.model_info),
                    inference_ms=round(elapsed_ms, 2),
                )
            )
        return results

    def info(self) -> dict[str, object]:
        return dict(self.model_info)

    def label_list(self) -> list[str]:
        return [self.labels[index] for index in range(len(self.labels))]
