from __future__ import annotations

from pathlib import Path

from ml.serving.predictor import ImageInput, InferenceResult, Predictor


def predict(
    image: ImageInput,
    model_path: Path | None = None,
    top_k: int = 5,
) -> InferenceResult:
    """Run one real prediction using the repository's trained checkpoint."""
    predictor = Predictor(model_path=model_path) if model_path else Predictor()
    return predictor.predict(image, top_k=top_k)


__all__ = ["InferenceResult", "Predictor", "predict"]
