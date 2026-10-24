from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from ml.serving.predictor import Predictor


class FixedLogits(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((images.shape[0], 89), dtype=images.dtype, device=images.device)
        logits[:, 0] = 2.0
        logits[:, 1] = 1.0
        return logits


def load_fixture(checkpoint: Path, labels_path: Path, device: str):
    labels = {index: f"class {index}" for index in range(89)}
    return FixedLogits().to(device).eval(), {
        "labels": labels, "backbone": "EfficientNet-B0", "num_classes": 89,
        "img_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "model_path": str(checkpoint),
    }


class PredictorCalibrationTests(unittest.TestCase):
    def test_verified_temperature_calibrates_confidence_and_preserves_raw_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "best_model.pth"
            checkpoint.write_bytes(b"fixture checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            sidecar = checkpoint.with_name("confidence_calibration.json")
            sidecar.write_text(json.dumps({
                "format": "terramind-temperature-scaling-v1", "checkpoint_sha256": digest,
                "method": "temperature scaling", "temperature": 0.5,
                "fit_split": "duplicate-free official validation", "fit_samples": 100,
            }), encoding="utf-8")
            with patch("ml.serving.predictor.load_checkpoint", side_effect=load_fixture):
                predictor = Predictor(model_path=checkpoint)
                result = predictor.predict(Image.new("RGB", (300, 200)), top_k=3)
            self.assertEqual(result.predicted_class.class_id, 0)
            self.assertGreater(result.predicted_class.confidence, result.predicted_class.raw_score)
            self.assertEqual(result.top5[0].class_id, result.predicted_class.class_id)
            self.assertEqual(result.model_info["confidence_calibration"]["temperature"], 0.5)

    def test_wrong_checkpoint_checksum_falls_back_to_raw_probability(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "best_model.pth"
            checkpoint.write_bytes(b"fixture checkpoint")
            checkpoint.with_name("confidence_calibration.json").write_text(json.dumps({
                "format": "terramind-temperature-scaling-v1", "checkpoint_sha256": "0" * 64,
                "method": "temperature scaling", "temperature": 0.5,
                "fit_split": "duplicate-free official validation", "fit_samples": 100,
            }), encoding="utf-8")
            with patch("ml.serving.predictor.load_checkpoint", side_effect=load_fixture):
                result = Predictor(model_path=checkpoint).predict(Image.new("RGB", (200, 300)))
            self.assertAlmostEqual(result.predicted_class.confidence, result.predicted_class.raw_score)
            self.assertEqual(result.model_info["confidence_calibration"]["method"], "none")


if __name__ == "__main__":
    unittest.main()
