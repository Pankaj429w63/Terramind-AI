from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from ml.multimodal.data import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    PlantWildMetadataDataset,
    build_vocabulary,
    collate_metadata_pairs,
    discover_pairs,
    encode_metadata,
    metadata_tokens,
)
from ml.multimodal.model import MultimodalAutoencoder, MultimodalConfig
from ml.multimodal.runtime import multimodal_status


class MultimodalPreparationTests(unittest.TestCase):
    def test_text_is_only_tokenized_class_metadata_and_vocabulary_is_stable(self) -> None:
        labels = {0: "Tomato early blight", 1: "Healthy tomato"}
        self.assertEqual(metadata_tokens(labels[0]), ["tomato", "early", "blight"])
        self.assertEqual(build_vocabulary(labels), build_vocabulary(labels))
        vocabulary = build_vocabulary(labels)
        encoded = encode_metadata(labels[0], vocabulary)
        self.assertEqual(encoded[0], BOS_ID)
        self.assertEqual(encoded[-1], EOS_ID)
        self.assertEqual([k for k, v in vocabulary.items() if v == PAD_ID], ["<pad>"])

    def test_split_records_pair_existing_image_paths_with_exact_class_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "tomato early blight").mkdir(parents=True)
            image = root / "images" / "tomato early blight" / "sample.jpg"
            Image.new("RGB", (32, 32), "green").save(image)
            (root / "classes.txt").write_text("0 tomato early blight\n", encoding="utf-8")
            (root / "train_split.txt").write_text("tomato early blight/sample.jpg=0=1\n", encoding="utf-8")
            records = discover_pairs(root, Path("train_split.txt"))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].metadata_text, "tomato early blight")
            dataset = PlantWildMetadataDataset(records, build_vocabulary({0: "tomato early blight"}), ToTensor())
            batch = collate_metadata_pairs([dataset[0]])
            self.assertEqual(tuple(batch["images"].shape), (1, 3, 32, 32))

    def test_split_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "classes.txt").write_text("0 tomato\n", encoding="utf-8")
            (root / "train_split.txt").write_text("../outside.jpg=0=1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                discover_pairs(root, Path("train_split.txt"))


class MultimodalArchitectureTests(unittest.TestCase):
    def test_reconstruction_cross_attention_and_fused_shared_z(self) -> None:
        torch.manual_seed(7)
        feature_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.AdaptiveAvgPool2d((7, 7)),
        )
        config = MultimodalConfig(
            latent_dim=32,
            text_embedding_dim=32,
            attention_heads=4,
            transformer_layers=1,
            dropout=0.0,
            max_image_tokens=64,
            max_text_length=8,
        )
        model = MultimodalAutoencoder(feature_encoder, vocabulary_size=12, feature_dim=16, config=config)
        images = torch.randn(2, 3, 224, 224)
        text_ids = torch.tensor([[2, 4, 5, 3], [2, 6, 3, PAD_ID]])
        class_ids = torch.tensor([1, 2])
        outputs = model(images, text_ids)
        losses = model.loss(outputs, images, text_ids, class_ids)
        self.assertEqual(tuple(outputs["image_reconstruction"].shape), tuple(images.shape))
        self.assertEqual(tuple(outputs["text_logits"].shape), (2, 3, 12))
        self.assertEqual(tuple(outputs["fused_z"].shape), (2, 32))
        self.assertEqual(tuple(outputs["image_cross_attention"].shape), (2, 49, 32))
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(model.fusion_projection[0].weight.grad)

    def test_untrained_checkpoint_is_reported_as_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status = multimodal_status(Path(directory) / "not-trained.pth")
        self.assertEqual(status["status"], "unavailable")
        self.assertFalse(status["trained"])
        self.assertFalse(status["checkpoint_available"])


if __name__ == "__main__":
    unittest.main()
