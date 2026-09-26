from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .data import PAD_ID


@dataclass(frozen=True)
class MultimodalConfig:
    latent_dim: int = 256
    text_embedding_dim: int = 256
    attention_heads: int = 8
    transformer_layers: int = 2
    dropout: float = 0.1
    max_image_tokens: int = 256
    max_text_length: int = 32
    image_size: int = 224
    image_reconstruction_weight: float = 1.0
    text_reconstruction_weight: float = 1.0
    alignment_weight: float = 0.2

    def __post_init__(self) -> None:
        if self.latent_dim <= 0 or self.text_embedding_dim <= 0:
            raise ValueError("Embedding dimensions must be positive")
        if self.latent_dim % self.attention_heads:
            raise ValueError("latent_dim must be divisible by attention_heads")
        if self.image_size % 32:
            raise ValueError("image_size must be divisible by 32 for the image reconstruction decoder")


class MultimodalAutoencoder(nn.Module):
    """EfficientNet feature + class-metadata autoencoder with bidirectional cross-attention."""

    def __init__(
        self,
        image_encoder: nn.Module,
        vocabulary_size: int,
        feature_dim: int = 1280,
        config: MultimodalConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MultimodalConfig()
        self.feature_dim = feature_dim
        self.vocabulary_size = vocabulary_size
        self.image_encoder = image_encoder
        dim = self.config.latent_dim
        self.image_projection = nn.Sequential(nn.Linear(feature_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.image_position = nn.Parameter(torch.zeros(1, self.config.max_image_tokens, dim))
        nn.init.trunc_normal_(self.image_position, std=0.02)

        self.token_embedding = nn.Embedding(vocabulary_size, self.config.text_embedding_dim, padding_idx=PAD_ID)
        self.text_input_projection = nn.Linear(self.config.text_embedding_dim, dim)
        self.text_position = nn.Parameter(torch.zeros(1, self.config.max_text_length, dim))
        nn.init.trunc_normal_(self.text_position, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            dim, self.config.attention_heads, dim * 4, self.config.dropout, batch_first=True, norm_first=True
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.transformer_layers)

        def cross_decoder() -> nn.TransformerDecoder:
            layer = nn.TransformerDecoderLayer(
                dim, self.config.attention_heads, dim * 4, self.config.dropout, batch_first=True, norm_first=True
            )
            return nn.TransformerDecoder(layer, num_layers=self.config.transformer_layers)

        self.image_cross_decoder = cross_decoder()
        self.text_cross_decoder = cross_decoder()
        self.image_feature_projection = nn.Linear(dim, feature_dim)
        self.image_reconstruction_decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        self.text_reconstruction_head = nn.Linear(dim, vocabulary_size)
        self.fusion_projection = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim))

    def forward(self, images: Tensor, text_ids: Tensor) -> dict[str, Tensor]:
        if text_ids.ndim != 2 or text_ids.shape[1] < 2:
            raise ValueError("text_ids must be a batch of BOS/EOS token sequences")
        image_features = self.image_encoder(images)
        if image_features.ndim != 4 or image_features.shape[1] != self.feature_dim:
            raise ValueError(f"image_encoder must return [batch, {self.feature_dim}, height, width] features")
        batch, channels, height, width = image_features.shape
        image_sequence = image_features.flatten(2).transpose(1, 2)
        if image_sequence.shape[1] > self.config.max_image_tokens:
            raise ValueError("image feature grid exceeds configured max_image_tokens")
        image_z = self.image_projection(image_sequence) + self.image_position[:, : image_sequence.shape[1]]

        if text_ids.shape[1] > self.config.max_text_length:
            text_ids = text_ids[:, : self.config.max_text_length]
        text_padding = text_ids.eq(PAD_ID)
        text_embedded = self.text_input_projection(self.token_embedding(text_ids))
        text_embedded = text_embedded + self.text_position[:, : text_ids.shape[1]]
        causal = torch.triu(torch.ones(text_ids.shape[1], text_ids.shape[1], device=text_ids.device, dtype=torch.bool), diagonal=1)
        text_z = self.text_encoder(text_embedded, mask=causal, src_key_padding_mask=text_padding)

        image_cross = self.image_cross_decoder(image_z, text_z, memory_key_padding_mask=text_padding)
        text_cross = self.text_cross_decoder(
            text_z[:, :-1], image_z,
            tgt_mask=torch.triu(torch.ones(text_ids.shape[1] - 1, text_ids.shape[1] - 1, device=text_ids.device, dtype=torch.bool), diagonal=1),
        )
        valid_text = (~text_padding).to(text_z.dtype).unsqueeze(-1)
        text_pool = (text_cross * valid_text[:, :-1]).sum(dim=1) / valid_text[:, :-1].sum(dim=1).clamp_min(1.0)
        image_pool = image_cross.mean(dim=1)
        fused_z = self.fusion_projection(torch.cat((image_pool, text_pool), dim=-1))

        reconstructed_features = self.image_feature_projection(image_cross).transpose(1, 2).reshape(batch, channels, height, width)
        image_reconstruction = self.image_reconstruction_decoder(reconstructed_features)
        if image_reconstruction.shape[-2:] != images.shape[-2:]:
            image_reconstruction = F.interpolate(image_reconstruction, size=images.shape[-2:], mode="bilinear", align_corners=False)
        text_logits = self.text_reconstruction_head(text_cross)
        return {
            "image_reconstruction": image_reconstruction,
            "text_logits": text_logits,
            "image_z": image_z,
            "text_z": text_z,
            "fused_z": fused_z,
            "image_cross_attention": image_cross,
            "text_cross_attention": text_cross,
        }

    def loss(self, outputs: dict[str, Tensor], images: Tensor, text_ids: Tensor, class_ids: Tensor | None = None) -> dict[str, Tensor]:
        targets = text_ids[:, 1 : 1 + outputs["text_logits"].shape[1]]
        logits = outputs["text_logits"]
        if targets.shape[1] < logits.shape[1]:
            logits = logits[:, : targets.shape[1]]
        image_loss = F.mse_loss(outputs["image_reconstruction"], images)
        text_loss = F.cross_entropy(logits.reshape(-1, self.vocabulary_size), targets[:, : logits.shape[1]].reshape(-1), ignore_index=PAD_ID)
        image_repr = F.normalize(outputs["image_z"].mean(dim=1), dim=-1)
        mask = text_ids.ne(PAD_ID).to(outputs["text_z"].dtype).unsqueeze(-1)
        text_repr = F.normalize((outputs["text_z"] * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0), dim=-1)
        temperature = 0.07
        similarities = image_repr @ text_repr.transpose(0, 1) / temperature
        if class_ids is None:
            class_ids = torch.arange(images.shape[0], device=images.device)
        positive_pairs = class_ids[:, None].eq(class_ids[None, :]).to(similarities.dtype)

        def multi_positive_loss(scores: Tensor, positives: Tensor) -> Tensor:
            log_probability = scores - torch.logsumexp(scores, dim=1, keepdim=True)
            positive_log_probability = torch.logsumexp(log_probability.masked_fill(positives.eq(0), -torch.inf), dim=1)
            return -positive_log_probability.mean()

        pair_alignment_loss = (multi_positive_loss(similarities, positive_pairs) + multi_positive_loss(similarities.transpose(0, 1), positive_pairs.transpose(0, 1))) / 2
        fused_repr = F.normalize(outputs["fused_z"], dim=-1)
        fusion_alignment_loss = 1.0 - (
            (fused_repr * image_repr).sum(dim=-1).mean() + (fused_repr * text_repr).sum(dim=-1).mean()
        ) / 2
        alignment_loss = pair_alignment_loss + fusion_alignment_loss
        total = (
            self.config.image_reconstruction_weight * image_loss
            + self.config.text_reconstruction_weight * text_loss
            + self.config.alignment_weight * alignment_loss
        )
        return {
            "total": total,
            "image_reconstruction": image_loss,
            "text_reconstruction": text_loss,
            "alignment": alignment_loss,
            "fusion_alignment": fusion_alignment_loss,
        }

    def architecture_info(self) -> dict[str, Any]:
        return {
            "image_encoder": "EfficientNet-B0 features (1280 channels)",
            "text_encoder": "causal Transformer over PlantWild class metadata tokens",
            "shared_latent_dim": self.config.latent_dim,
            "cross_attention": "bidirectional Transformer decoders",
            "objectives": ["image pixel reconstruction", "class-metadata token reconstruction", "symmetric image-text contrastive alignment"],
            "config": asdict(self.config),
        }
