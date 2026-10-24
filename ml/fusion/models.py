from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn


FusionKind = Literal["early", "intermediate", "late"]


@dataclass(frozen=True)
class FusionConfig:
    num_classes: int = 89
    image_feature_dim: int = 1280
    text_vocab_size: int = 0
    text_embedding_dim: int = 256
    text_hidden_dim: int = 256
    fusion_dim: int = 512
    dropout: float = 0.3
    pad_id: int = 0
    max_text_length: int = 128

    def __post_init__(self) -> None:
        if self.num_classes < 2 or self.image_feature_dim < 1 or self.text_vocab_size < 2:
            raise ValueError("Fusion classifiers require valid class, image, and text vocabulary dimensions")
        if self.max_text_length < 1 or self.dropout < 0 or self.dropout >= 1:
            raise ValueError("Invalid text length or dropout")
        if self.text_hidden_dim % 2:
            raise ValueError("text_hidden_dim must be divisible by two for the bidirectional GRU")


class TextEncoder(nn.Module):
    """Word embedding + bidirectional GRU over supplied, non-target text."""

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.pad_id = config.pad_id
        self.embedding = nn.Embedding(config.text_vocab_size, config.text_embedding_dim, padding_idx=config.pad_id)
        self.rnn = nn.GRU(config.text_embedding_dim, config.text_hidden_dim // 2, batch_first=True, bidirectional=True)
        self.projection = nn.Sequential(
            nn.Linear(config.text_hidden_dim, config.fusion_dim), nn.LayerNorm(config.fusion_dim), nn.GELU()
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        lengths = token_ids.ne(self.pad_id).sum(dim=1).clamp_min(1).cpu()
        embedded = self.embedding(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        pooled = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.projection(pooled)


class FusionClassifierBase(nn.Module):
    fusion_kind: FusionKind

    def __init__(self, image_encoder: nn.Module, config: FusionConfig) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.config = config
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.image_projection = nn.Sequential(
            nn.Linear(config.image_feature_dim, config.fusion_dim), nn.LayerNorm(config.fusion_dim), nn.GELU()
        )
        self.text_encoder = TextEncoder(config)
        self.dropout = nn.Dropout(config.dropout)

    def encode_image(self, images: Tensor) -> Tensor:
        features = self.image_encoder(images)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if features.ndim != 4 or features.shape[1] != self.config.image_feature_dim:
            raise ValueError(f"image_encoder must return [batch, {self.config.image_feature_dim}, H, W]")
        return self.image_projection(self.image_pool(features).flatten(1))

    def encode_text(self, token_ids: Tensor) -> Tensor:
        return self.text_encoder(token_ids)

    def forward(self, images: Tensor, token_ids: Tensor) -> Tensor:
        raise NotImplementedError

    def model_metadata(self) -> dict:
        return {"fusion_kind": self.fusion_kind, "config": asdict(self.config)}


class EarlyFusionClassifier(FusionClassifierBase):
    """Concatenate projected modality features immediately before the class head."""

    fusion_kind: FusionKind = "early"

    def __init__(self, image_encoder: nn.Module, config: FusionConfig) -> None:
        super().__init__(image_encoder, config)
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_dim * 2, config.fusion_dim), nn.GELU(), self.dropout,
            nn.Linear(config.fusion_dim, config.num_classes),
        )

    def forward(self, images: Tensor, token_ids: Tensor) -> Tensor:
        image = self.encode_image(images)
        text = self.encode_text(token_ids)
        if image.shape[0] != text.shape[0]:
            raise ValueError("Image and text batch sizes do not match")
        return self.classifier(self.dropout(torch.cat((image, text), dim=1)))


class IntermediateFusionClassifier(FusionClassifierBase):
    """Gated cross-modal fusion: each modality gates a learned joint embedding."""

    fusion_kind: FusionKind = "intermediate"

    def __init__(self, image_encoder: nn.Module, config: FusionConfig) -> None:
        super().__init__(image_encoder, config)
        self.joint_projection = nn.Sequential(
            nn.Linear(config.fusion_dim * 2, config.fusion_dim), nn.LayerNorm(config.fusion_dim), nn.GELU()
        )
        self.gate = nn.Sequential(nn.Linear(config.fusion_dim * 2, config.fusion_dim), nn.Sigmoid())
        self.classifier = nn.Linear(config.fusion_dim, config.num_classes)

    def forward(self, images: Tensor, token_ids: Tensor) -> Tensor:
        image = self.encode_image(images)
        text = self.encode_text(token_ids)
        if image.shape[0] != text.shape[0]:
            raise ValueError("Image and text batch sizes do not match")
        joined = torch.cat((image, text), dim=1)
        gate = self.gate(joined)
        fused = gate * self.joint_projection(joined) + (1.0 - gate) * (image + text) * 0.5
        return self.classifier(self.dropout(fused))


class LateFusionClassifier(FusionClassifierBase):
    """Independent modality classifiers; validation chooses their probability weight."""

    fusion_kind: FusionKind = "late"

    def __init__(self, image_encoder: nn.Module, config: FusionConfig) -> None:
        super().__init__(image_encoder, config)
        self.image_classifier = nn.Linear(config.fusion_dim, config.num_classes)
        self.text_classifier = nn.Linear(config.fusion_dim, config.num_classes)
        self.validation_image_weight: float | None = None

    def modality_logits(self, images: Tensor, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        image = self.encode_image(images)
        text = self.encode_text(token_ids)
        if image.shape[0] != text.shape[0]:
            raise ValueError("Image and text batch sizes do not match")
        return self.image_classifier(self.dropout(image)), self.text_classifier(self.dropout(text))

    def forward(self, images: Tensor, token_ids: Tensor) -> Tensor:
        image_logits, text_logits = self.modality_logits(images, token_ids)
        # Equal-weight logits are used only while training. Evaluation/inference
        # should combine probabilities using a weight selected on validation.
        return 0.5 * image_logits + 0.5 * text_logits

    @staticmethod
    def fuse_probabilities(image_probabilities: Tensor, text_probabilities: Tensor, image_weight: float) -> Tensor:
        if not 0.0 <= image_weight <= 1.0:
            raise ValueError("image_weight must lie in [0, 1]")
        if image_probabilities.shape != text_probabilities.shape:
            raise ValueError("Modality probability matrices must have the same shape and class ordering")
        return image_weight * image_probabilities + (1.0 - image_weight) * text_probabilities

    def predict_proba(self, images: Tensor, token_ids: Tensor, image_weight: float | None = None) -> Tensor:
        weight = self.validation_image_weight if image_weight is None else image_weight
        if weight is None:
            raise RuntimeError("Late fusion weight has not been selected on validation data")
        image_logits, text_logits = self.modality_logits(images, token_ids)
        return self.fuse_probabilities(image_logits.softmax(dim=1), text_logits.softmax(dim=1), weight)


def build_fusion_model(kind: FusionKind, image_encoder: nn.Module, config: FusionConfig) -> FusionClassifierBase:
    models = {"early": EarlyFusionClassifier, "intermediate": IntermediateFusionClassifier, "late": LateFusionClassifier}
    try:
        return models[kind](image_encoder, config)
    except KeyError as error:
        raise ValueError(f"Unknown fusion kind: {kind}") from error
