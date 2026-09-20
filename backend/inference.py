from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image, ImageFile

from ml.models.model_loader import DEFAULT_MODEL_PATH
from ml.serving.predictor import InferenceResult, Predictor as ServingPredictor

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = DEFAULT_MODEL_PATH
DATA_CLASSES_TXT = PROJECT_ROOT / "data" / "plantwild" / "plantwild" / "classes.txt"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("terramind.inference")


@dataclass
class ClassPrediction:
    class_id: int
    label: str
    score: float
    confidence_pct: float


@dataclass
class PredictionResult:
    top: ClassPrediction
    top5: List[ClassPrediction] = field(default_factory=list)
    inference_ms: float = 0.0
    model: str = "EfficientNet-B0"
    img_size: int = 224
    low_confidence: bool = False
    low_confidence_threshold: float = 0.35
    notes: str = ""


def _to_class_prediction(item) -> ClassPrediction:
    score = float(item.confidence)
    return ClassPrediction(
        class_id=int(item.class_id),
        label=str(item.label),
        score=score,
        confidence_pct=round(score * 100.0, 2),
    )


def _to_result(res: InferenceResult, low_confidence_threshold: float) -> PredictionResult:
    top5 = [_to_class_prediction(item) for item in res.top5]
    top = top5[0]
    low = top.score < low_confidence_threshold
    notes = ""
    if low:
        notes = (
            f"Low confidence prediction ({top.confidence_pct:.1f}% < "
            f"{low_confidence_threshold * 100:.0f}%). "
            "Consider a clearer image of the leaf / lesion area."
        )
    return PredictionResult(
        top=top,
        top5=top5,
        inference_ms=res.inference_ms,
        model=str(res.model_info.get("backbone") or "EfficientNet-B0"),
        img_size=int(res.model_info.get("img_size") or 224),
        low_confidence=low,
        low_confidence_threshold=low_confidence_threshold,
        notes=notes,
    )


class PlantPredictor:
    """Production inference for the archived EfficientNet-B0 v3 checkpoint."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        classes_txt: Path = DATA_CLASSES_TXT,
        device: str = "cpu",
        img_size: Optional[int] = None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        del img_size, mean, std
        self._inner = ServingPredictor(
            model_path=Path(model_path),
            labels_path=Path(classes_txt),
            device=device,
        )
        info = self._inner.info()
        self.device = self._inner.device
        self.model_path = Path(str(info["model_path"]))
        self.model = self._inner.model
        self.num_classes = int(info["num_classes"])
        self.idx_to_class = dict(self._inner.labels)
        self.classes_map = {name: index for index, name in self.idx_to_class.items()}
        self.backbone = str(info["backbone"])
        self.img_size = int(info["img_size"])
        self.mean = list(info["mean"])  # type: ignore[arg-type]
        self.std = list(info["std"])  # type: ignore[arg-type]
        self.test_acc = info.get("test_acc")
        self.best_val_acc = info.get("best_val_macro_f1") or info.get("best_val_acc")
        self.transform = self._inner.transform
        self.load_time_ms = 0.0
        log.info(
            "PlantPredictor loaded: %s | classes=%d | img=%dpx | checkpoint=%s",
            self.backbone, self.num_classes, self.img_size, self.model_path,
        )

    def predict(
        self,
        image: Union[str, Path, bytes, Image.Image, np.ndarray],
        top_k: int = 5,
        low_confidence_threshold: float = 0.35,
    ) -> PredictionResult:
        return _to_result(self._inner.predict(image, top_k=top_k), low_confidence_threshold)

    def predict_batch(
        self,
        images: List[Union[str, Path, bytes, Image.Image, np.ndarray]],
        top_k: int = 5,
        low_confidence_threshold: float = 0.35,
    ) -> List[PredictionResult]:
        return [_to_result(item, low_confidence_threshold) for item in self._inner.predict_batch(images, top_k=top_k)]

    def labels(self) -> List[str]:
        return self._inner.label_list()

    def info(self) -> dict:
        return self._inner.info()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    predictor = PlantPredictor()
    sample_root = PROJECT_ROOT / "data" / "plantwild_v2" / "plantwild_v2"
    sample = None
    if sample_root.exists():
        for p in sample_root.rglob("*.jpg"):
            sample = p
            break
    if sample is None:
        for p in (PROJECT_ROOT / "data" / "plantwild").rglob("*.jpg"):
            sample = p
            break
    if sample is None:
        log.info("No sample image available; skipping real-image test.")
        raise SystemExit(0)
    log.info("Running real prediction on: %s", sample)
    res = predictor.predict(sample)
    print("\n=== TOP-1 ===")
    print(f"  Label : {res.top.label}")
    print(f"  Score : {res.top.confidence_pct:.2f}%")
    print(f"  LowConf: {res.low_confidence}  [{res.notes}]")
    print(f"  Time  : {res.inference_ms:.1f}ms")
    print("=== TOP-5 ===")
    for cp in res.top5:
        print(f"  {cp.class_id:>3}  {cp.confidence_pct:>6.2f}%  {cp.label}")
    log.info("Running 2-image batch test (same image twice)...")
    batch_res = predictor.predict_batch([sample, sample], top_k=3)
    for i, r in enumerate(batch_res):
        print(f"  batch[{i}] {r.top.label} {r.top.confidence_pct:.2f}% ({r.inference_ms:.1f}ms)")
