from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from ml.models.model_loader import DEFAULT_MODEL_PATH, load_checkpoint
from ml.serving.preprocessing import build_inference_transform
from .data import PlantWildMetadataDataset, build_vocabulary, collate_metadata_pairs, discover_pairs
from .model import MultimodalAutoencoder, MultimodalConfig
from .runtime import DATASET_ROOT, DEFAULT_MULTIMODAL_CHECKPOINT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(dataset: PlantWildMetadataDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_metadata_pairs, generator=generator)


def train(args: argparse.Namespace) -> Path:
    config_values = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    model_config = MultimodalConfig(**{key: value for key, value in config_values.items() if key in MultimodalConfig.__dataclass_fields__})
    seed = int(config_values.get("seed", 42))
    _seed_everything(seed)

    production_model, metadata = load_checkpoint(args.production_checkpoint, args.dataset_root / "classes.txt", "cpu")
    labels: dict[int, str] = metadata["labels"]
    train_records = discover_pairs(args.dataset_root, args.train_split, labels)
    validation_records = discover_pairs(args.dataset_root, args.validation_split, labels)
    vocabulary = build_vocabulary(labels)
    transform = build_inference_transform("EfficientNet-B0", model_config.image_size, metadata["mean"], metadata["std"])
    train_data = PlantWildMetadataDataset(train_records, vocabulary, transform, model_config.max_text_length)
    validation_data = PlantWildMetadataDataset(validation_records, vocabulary, transform, model_config.max_text_length)
    batch_size = int(config_values.get("batch_size", 16))
    train_loader = _loader(train_data, batch_size, True, seed)
    validation_loader = _loader(validation_data, batch_size, False, seed)

    model = MultimodalAutoencoder(production_model.features, len(vocabulary), feature_dim=1280, config=model_config)
    if bool(config_values.get("freeze_image_encoder", True)):
        for parameter in model.image_encoder.parameters():
            parameter.requires_grad = False
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config_values.get("learning_rate", 1e-4)),
        weight_decay=float(config_values.get("weight_decay", 1e-4)),
    )

    epochs = args.epochs if args.epochs is not None else int(config_values.get("epochs", 25))
    if epochs < 1:
        raise ValueError("Training requires at least one epoch")
    for epoch in range(epochs):
        model.train()
        model.image_encoder.eval() if bool(config_values.get("freeze_image_encoder", True)) else None
        for batch in train_loader:
            images = batch["images"].to(device)
            text_ids = batch["text_ids"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images, text_ids)
            losses = model.loss(outputs, images, text_ids, batch["class_ids"].to(device))
            losses["total"].backward()
            optimizer.step()

        model.eval()
        totals = {"total": 0.0, "image_reconstruction": 0.0, "text_reconstruction": 0.0, "alignment": 0.0, "fusion_alignment": 0.0}
        examples = 0
        with torch.inference_mode():
            for batch in validation_loader:
                images = batch["images"].to(device)
                text_ids = batch["text_ids"].to(device)
                losses = model.loss(model(images, text_ids), images, text_ids, batch["class_ids"].to(device))
                count = images.shape[0]
                for key in totals:
                    totals[key] += float(losses[key]) * count
                examples += count
        if examples == 0:
            raise RuntimeError("Validation split produced no examples; refusing to save an unvalidated checkpoint")
        val_losses = {key: value / examples for key, value in totals.items()}
        if not all(math.isfinite(value) for value in val_losses.values()):
            raise RuntimeError("Validation produced a non-finite loss; refusing to save a checkpoint")
        print(f"epoch={epoch + 1}/{epochs} validation_losses={val_losses}")

    output = args.output.resolve()
    protected_paths = [DEFAULT_MODEL_PATH.resolve(), (PROJECT_ROOT / "models" / "archive").resolve(), (PROJECT_ROOT / "outputs").resolve()]
    if any(output == protected or protected in output.parents for protected in protected_paths):
        raise ValueError("Multimodal output must not overwrite the production checkpoint or existing training outputs")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "terramind-multimodal-v1",
        "training_state": {"completed": True, "validated": True, "epochs": epochs, "validation_losses": val_losses},
        "model_state": model.cpu().state_dict(),
        "model_config": asdict(model_config),
        "vocabulary": vocabulary,
        "class_labels": labels,
        "data": {"train_pairs": len(train_records), "validation_pairs": len(validation_records), "text_source": "PlantWild class labels only"},
        "production_checkpoint": str(args.production_checkpoint.resolve()),
        "seed": seed,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the separate multimodal autoencoder from PlantWild class metadata pairs.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--train-split", type=Path, default=Path("train_split.txt"))
    parser.add_argument("--validation-split", type=Path, default=Path("val_split.txt"))
    parser.add_argument("--production-checkpoint", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_MULTIMODAL_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print("Using paired PlantWild images and exact class labels; no natural-language descriptions are created.")
    print(f"Multimodal checkpoint output: {args.output}")
    print(f"Production EfficientNet-B0 checkpoint is read-only: {args.production_checkpoint}")
    print(f"Saved checkpoint: {train(args)}")


if __name__ == "__main__":
    main()
