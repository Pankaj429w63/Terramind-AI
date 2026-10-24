"""Inference-runtime tests for the three fusion architectures.

These exercise checkpoint save/load and class-order consistency, CPU execution,
modality reporting, the labelled image-only fallback for late fusion, and the
missing-text guard for early/intermediate fusion. They never train a model and
never fabricate metrics.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models

from ml.fusion.inference import (
    CHECKPOINT_FORMAT,
    FusionInferenceError,
    FusionInferenceRuntime,
    MissingTextError,
    SUPPORTED_FUSION_KINDS,
    default_checkpoint,
    fusion_runtime_status,
    load_fusion_checkpoint,
)
from ml.fusion.models import FusionConfig, LateFusionClassifier, build_fusion_model

LABELS = {0: "tomato early blight", 1: "apple scab"}
VOCABULARY = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "leaf": 3, "spots": 4}


def _config() -> FusionConfig:
    return FusionConfig(
        num_classes=2, image_feature_dim=1280, text_vocab_size=len(VOCABULARY),
        text_embedding_dim=8, text_hidden_dim=8, fusion_dim=16, dropout=0.0, max_text_length=6,
    )


def _build_model(kind: str, config: FusionConfig):
    encoder = models.efficientnet_b0(weights=None).features
    return build_fusion_model(kind, encoder, config)


def _write_checkpoint(root: Path, kind: str, *, labels: dict[int, str] | None = None,
                      vocabulary: dict[str, int] | None = None) -> Path:
    config = _config()
    model = _build_model(kind, config)
    torch.manual_seed(11)
    checkpoint = default_checkpoint(kind, root)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT, "fusion_kind": kind,
        "model_config": dict(config.__dict__), "vocabulary": dict(vocabulary or VOCABULARY),
        "class_labels": dict(labels or LABELS), "model_state": model.state_dict(),
        "best_validation_macro_f1": 0.5,
        "validation_image_weight": 0.5 if kind == "late" else None,
        "text_origins": ["model_generated_observation"],
    }
    torch.save(payload, checkpoint)
    manifest = {
        "fusion_kind": kind, "implemented": True, "trained": True,
        "checkpoint": str(checkpoint.resolve()),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "class_labels": {str(key): value for key, value in (labels or LABELS).items()},
        "validation_macro_f1": 0.5, "test_evaluation_completed": False, "loaded": False,
    }
    checkpoint.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint


def _image() -> Image.Image:
    return Image.new("RGB", (48, 48), (30, 120, 40))


class FusionRuntimeLoadingTests(unittest.TestCase):
    def test_all_three_kinds_load_on_cpu_and_emit_class_ordered_predictions(self) -> None:
        for kind in SUPPORTED_FUSION_KINDS:
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    checkpoint = _write_checkpoint(root, kind)
                    runtime = FusionInferenceRuntime(kind, checkpoint, LABELS, "cpu")
                    self.assertEqual(runtime.device.type, "cpu")
                    self.assertEqual(runtime.requires_text, kind != "late")
                    self.assertEqual(runtime.supports_image_only, kind == "late")

    def test_load_classmethod_reads_official_class_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_checkpoint(root, "early")
            labels_path = root / "classes.txt"
            labels_path.write_text("\n".join(f"{i} {name}" for i, name in LABELS.items()) + "\n", encoding="utf-8")
            runtime = FusionInferenceRuntime.load("early", default_checkpoint("early", root), labels_path, "cpu")
            self.assertEqual(runtime.labels, LABELS)

    def test_class_order_mismatch_and_checksum_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = _write_checkpoint(root, "early")
            swapped = {0: LABELS[1], 1: LABELS[0]}
            with self.assertRaises(FusionInferenceError):
                load_fusion_checkpoint("early", checkpoint, swapped)
            checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
            with self.assertRaises(FusionInferenceError):
                load_fusion_checkpoint("early", checkpoint, LABELS)

    def test_untrained_manifest_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = _write_checkpoint(root, "late")
            manifest_path = checkpoint.with_suffix(".manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["trained"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(FusionInferenceError):
                load_fusion_checkpoint("late", checkpoint, LABELS)

    def test_status_reports_sidecar_without_loading_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / "classes.txt"
            labels_path.write_text("\n".join(f"{i} {name}" for i, name in LABELS.items()) + "\n", encoding="utf-8")
            _write_checkpoint(root, "intermediate")
            report = {entry["kind"]: entry for entry in fusion_runtime_status(root, labels_path)}
            self.assertTrue(report["intermediate"]["available"])
            self.assertTrue(report["intermediate"]["trained"])
            self.assertFalse(report["early"]["available"])
            self.assertFalse(report["late"]["available"])
            self.assertTrue(report["early"]["requires_text"])
            self.assertFalse(report["late"]["requires_text"])


class FusionArchitectureShapeTests(unittest.TestCase):
    def test_all_three_architectures_emit_89_class_output_for_batch_sizes(self) -> None:
        config = FusionConfig(
            num_classes=89, image_feature_dim=1280, text_vocab_size=len(VOCABULARY),
            text_embedding_dim=8, text_hidden_dim=8, fusion_dim=16, dropout=0.0, max_text_length=6,
        )
        for kind in SUPPORTED_FUSION_KINDS:
            with self.subTest(kind=kind):
                torch.manual_seed(5)
                encoder = models.efficientnet_b0(weights=None).features
                model = build_fusion_model(kind, encoder, config).eval()
                with torch.inference_mode():
                    single = model(torch.randn(1, 3, 224, 224), torch.tensor([[3, 4, 2]]))
                    batch = model(
                        torch.randn(3, 3, 224, 224),
                        torch.tensor([[3, 4, 2], [4, 2, 0], [3, 3, 2]]),
                    )
                self.assertEqual(tuple(single.shape), (1, 89))
                self.assertEqual(tuple(batch.shape), (3, 89))


class FusionRuntimeInferenceTests(unittest.TestCase):
    def test_multimodal_prediction_reports_both_modalities(self) -> None:
        for kind in SUPPORTED_FUSION_KINDS:
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as directory:
                    runtime = FusionInferenceRuntime(kind, _write_checkpoint(Path(directory), kind), LABELS, "cpu")
                    result = runtime.predict(_image(), text="leaf spots", top_k=2)
                    self.assertEqual(result["modalities_used"], ["image", "text"])
                    self.assertTrue(result["multimodal"])
                    self.assertFalse(result["image_only_fallback"])
                    self.assertTrue(result["text_provided"])
                    self.assertIn(result["predicted_class"]["class_id"], LABELS)
                    self.assertEqual(len(result["top_k"]), 2)
                    self.assertGreaterEqual(result["confidence"], 0.0)

    def test_late_fusion_image_only_fallback_is_labelled_image_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = FusionInferenceRuntime("late", _write_checkpoint(Path(directory), "late"), LABELS, "cpu")
            result = runtime.predict(_image(), text=None, top_k=1)
            self.assertEqual(result["modalities_used"], ["image"])
            self.assertFalse(result["multimodal"])
            self.assertTrue(result["image_only_fallback"])
            self.assertFalse(result["text_provided"])
            self.assertEqual(result["fusion_kind"], "late")

    def test_early_and_intermediate_require_text(self) -> None:
        for kind in ("early", "intermediate"):
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as directory:
                    runtime = FusionInferenceRuntime(kind, _write_checkpoint(Path(directory), kind), LABELS, "cpu")
                    with self.assertRaises(MissingTextError):
                        runtime.predict(_image(), text=None)
                    with self.assertRaises(MissingTextError):
                        runtime.predict(_image(), text="   ")

    def test_unknown_tokens_still_produce_a_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = FusionInferenceRuntime("early", _write_checkpoint(Path(directory), "early"), LABELS, "cpu")
            result = runtime.predict(_image(), text="zzz qqq", top_k=2)
            self.assertTrue(result["multimodal"])
            self.assertEqual(len(result["top_k"]), 2)

    def test_repeated_predictions_are_deterministic_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = FusionInferenceRuntime("intermediate", _write_checkpoint(Path(directory), "intermediate"), LABELS, "cpu")
            first = runtime.predict(_image(), text="leaf spots")
            second = runtime.predict(_image(), text="leaf spots")
            self.assertEqual(first["predicted_class"], second["predicted_class"])

    def test_status_exposes_checkpoint_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = FusionInferenceRuntime("early", _write_checkpoint(Path(directory), "early"), LABELS, "cpu")
            status = runtime.status()
            self.assertEqual(status["kind"], "early")
            self.assertTrue(status["loaded"])
            self.assertTrue(status["requires_text"])
            self.assertFalse(status["supports_image_only_fallback"])
            self.assertEqual(status["class_count"], 2)


class FusionTrainingGraphTests(unittest.TestCase):
    """Loss computation and gradient flow for all three architectures (CPU only).

    A successful forward pass is not evidence of a trained model; these tests only
    prove the training graph is wired correctly (finite loss, gradients on every
    trainable parameter). They never train and never claim accuracy.
    """

    @staticmethod
    def _dummy_image_encoder(feature_dim: int) -> nn.Module:
        # Cheap stand-in for the EfficientNet-B0 feature extractor: returns
        # [batch, feature_dim, H, W] as ``encode_image`` requires.
        return nn.Sequential(nn.Conv2d(3, feature_dim, kernel_size=1))

    def test_loss_is_finite_and_gradients_reach_all_parameters(self) -> None:
        config = FusionConfig(
            num_classes=5, image_feature_dim=32, text_vocab_size=len(VOCABULARY),
            text_embedding_dim=8, text_hidden_dim=8, fusion_dim=16, dropout=0.0, max_text_length=6,
        )
        images = torch.randn(4, 3, 12, 12)
        token_ids = torch.tensor([[3, 4, 2], [4, 2, 0], [3, 3, 2], [4, 4, 0]])
        targets = torch.tensor([0, 4, 2, 1])
        criterion = nn.CrossEntropyLoss()
        for kind in SUPPORTED_FUSION_KINDS:
            with self.subTest(kind=kind):
                torch.manual_seed(7)
                model = build_fusion_model(kind, self._dummy_image_encoder(32), config).train()
                logits = model(images, token_ids)
                self.assertEqual(tuple(logits.shape), (4, 5))
                self.assertTrue(torch.isfinite(logits).all())
                loss = criterion(logits, targets)
                self.assertTrue(torch.isfinite(loss))
                model.zero_grad()
                loss.backward()
                missing = [name for name, parameter in model.named_parameters()
                           if parameter.requires_grad and parameter.grad is None]
                self.assertEqual(missing, [], f"{kind} left parameters without gradients: {missing}")
                for _, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_late_fusion_is_deterministic_once_a_weight_is_selected(self) -> None:
        config = FusionConfig(
            num_classes=3, image_feature_dim=16, text_vocab_size=len(VOCABULARY),
            text_embedding_dim=8, text_hidden_dim=8, fusion_dim=16, dropout=0.0, max_text_length=6,
        )
        torch.manual_seed(3)
        model = build_fusion_model("late", self._dummy_image_encoder(16), config).eval()
        images = torch.randn(2, 3, 10, 10)
        token_ids = torch.tensor([[3, 4, 2], [4, 2, 0]])
        first = model.predict_proba(images, token_ids, image_weight=0.25)
        second = model.predict_proba(images, token_ids, image_weight=0.25)
        self.assertTrue(torch.allclose(first, second))
        self.assertTrue(torch.allclose(first.sum(dim=1), torch.ones(2)))

    def test_late_fusion_requires_a_selected_weight(self) -> None:
        config = FusionConfig(
            num_classes=3, image_feature_dim=16, text_vocab_size=len(VOCABULARY),
            text_embedding_dim=8, text_hidden_dim=8, fusion_dim=16, dropout=0.0, max_text_length=6,
        )
        model = build_fusion_model("late", self._dummy_image_encoder(16), config).eval()
        with self.assertRaises(RuntimeError):
            model.predict_proba(torch.randn(1, 3, 10, 10), torch.tensor([[3, 4, 2]]))


if __name__ == "__main__":
    unittest.main()
