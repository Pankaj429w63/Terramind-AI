"""Smoke tests for fusion training pipeline and model architectures."""
from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ml.fusion.models import (
    FusionConfig,
    EarlyFusionClassifier,
    IntermediateFusionClassifier,
    LateFusionClassifier,
    build_fusion_model,
)

VOCAB_SIZE = 50
NUM_CLASSES = 89


def _small_config() -> FusionConfig:
    return FusionConfig(
        num_classes=NUM_CLASSES,
        image_feature_dim=1280,
        text_vocab_size=VOCAB_SIZE,
        text_embedding_dim=64,
        text_hidden_dim=64,
        fusion_dim=128,
        dropout=0.1,
        pad_id=0,
        max_text_length=32,
    )


class FusionSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.config = _small_config()
        self.encoder = models.efficientnet_b0(weights=None).features
        # Freeze image encoder as required by protocol
        for param in self.encoder.parameters():
            param.requires_grad = False

    def test_early_fusion_forward_backward_step(self) -> None:
        model = build_fusion_model("early", self.encoder, self.config)
        images = torch.randn(4, 3, 224, 224)
        tokens = torch.randint(0, VOCAB_SIZE, (4, 16))
        targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        logits = model(images, tokens)
        self.assertEqual(logits.shape, (4, NUM_CLASSES))

        loss = F.cross_entropy(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()

        # Check trainable parameters have gradients
        trainable_grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(len(trainable_grads) > 0)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in trainable_grads))

        # Check frozen image encoder has NO gradients
        frozen_grads = [p.grad for p in model.image_encoder.parameters()]
        self.assertTrue(all(g is None for g in frozen_grads))

    def test_intermediate_fusion_forward_backward_step(self) -> None:
        model = build_fusion_model("intermediate", self.encoder, self.config)
        images = torch.randn(4, 3, 224, 224)
        tokens = torch.randint(0, VOCAB_SIZE, (4, 16))
        targets = torch.tensor([10, 20, 30, 40], dtype=torch.long)

        logits = model(images, tokens)
        self.assertEqual(logits.shape, (4, NUM_CLASSES))

        loss = F.cross_entropy(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        trainable_grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in trainable_grads))

    def test_late_fusion_forward_backward_step(self) -> None:
        model = build_fusion_model("late", self.encoder, self.config)
        images = torch.randn(4, 3, 224, 224)
        tokens = torch.randint(0, VOCAB_SIZE, (4, 16))
        targets = torch.tensor([5, 15, 25, 35], dtype=torch.long)

        logits = model(images, tokens)
        self.assertEqual(logits.shape, (4, NUM_CLASSES))

        loss = F.cross_entropy(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        trainable_grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in trainable_grads))

    def test_optimization_step_reduces_batch_loss(self) -> None:
        """Verify gradient descent works properly on fusion layers."""
        model = build_fusion_model("early", self.encoder, self.config)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.01,
        )
        images = torch.randn(4, 3, 224, 224)
        tokens = torch.randint(0, VOCAB_SIZE, (4, 16))
        targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            logits = model(images, tokens)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.assertLess(losses[-1], losses[0], "Loss must decrease over optimization steps on a single batch")


if __name__ == "__main__":
    unittest.main()
