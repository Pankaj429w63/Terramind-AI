"""Independent text-only baseline over the same eligible paired text and official splits.

This is a control for the fusion comparison, not a fusion model: it consumes only
the validated paired text (no images) and is trained/evaluated on exactly the same
duplicate-free eligible samples and official split boundaries as the fusion models.
It must not be trained or evaluated until valid, image-specific paired text exists.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ml.models.class_labels import load_class_labels
from .data import PAD_ID, PairedExample, build_vocabulary, encode_text, load_paired_text_csv
from .eligibility import eligible_sample_ids, filter_paired_examples, load_eligibility_manifest
from .models import FusionConfig, TextEncoder
from .train import seed_everything

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "data" / "plantwild" / "plantwild"
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
DEFAULT_OUTPUT = ROOT / "outputs" / "fusion" / "text_only" / "best_model.pth"
CHECKPOINT_FORMAT = "terramind-fusion-text-only-v1"


class TextOnlyClassifier(nn.Module):
    """Word-embedding + BiGRU text classifier reusing the fusion text encoder."""

    fusion_kind = "text_only"

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        self.text_encoder = TextEncoder(config)
        self.classifier = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.fusion_dim, config.num_classes))

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        return self.classifier(self.text_encoder(token_ids))


class TextOnlyDataset(Dataset[dict[str, object]]):
    def __init__(self, examples: Sequence[PairedExample], vocabulary: Mapping[str, int], max_length: int):
        self.examples = tuple(examples)
        self.vocabulary = dict(vocabulary)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        return {
            "token_ids": torch.tensor(encode_text(example.text, self.vocabulary, self.max_length), dtype=torch.long),
            "class_id": example.class_id,
            "split": example.split,
            "image_path": example.relative_path,
            "sample_id": example.sample_id,
        }


def collate_text_only(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    width = max(item["token_ids"].numel() for item in batch)  # type: ignore[union-attr]
    token_ids = torch.full((len(batch), width), PAD_ID, dtype=torch.long)
    for index, item in enumerate(batch):
        ids = item["token_ids"]
        token_ids[index, : ids.numel()] = ids  # type: ignore[union-attr]
    return {
        "token_ids": token_ids,
        "class_ids": torch.tensor([int(item["class_id"]) for item in batch], dtype=torch.long),
        "sample_ids": [str(item["sample_id"]) for item in batch],
    }


def _atomic_torch_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _loader(dataset: TextOnlyDataset, batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_text_only,
                      num_workers=0, pin_memory=torch.cuda.is_available(), generator=generator)


def _validation(model: TextOnlyClassifier, loader: DataLoader, device: torch.device, labels: Mapping[int, str]) -> float:
    model.eval()
    true, predicted = [], []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["token_ids"].to(device, non_blocking=True))
            true.append(batch["class_ids"])
            predicted.append(logits.argmax(1).cpu())
    if not true:
        raise RuntimeError("No validation examples are available")
    return float(f1_score(torch.cat(true).numpy(), torch.cat(predicted).numpy(),
                          average="macro", labels=list(labels), zero_division=0))


def _eligible_pairs(args: argparse.Namespace, labels: Mapping[int, str]) -> list[PairedExample]:
    split_files = {"train": args.train_split, "validation": args.validation_split, "test": args.test_split}
    pairs = load_paired_text_csv(args.paired_data, args.dataset_root, split_files, labels)
    eligibility = load_eligibility_manifest(args.eligibility_manifest)
    eligible = eligible_sample_ids(eligibility, args.dataset_root, split_files, labels)
    return filter_paired_examples(pairs, eligible)


def train_text_only(args: argparse.Namespace) -> Path:
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    labels = load_class_labels(args.dataset_root / "classes.txt")
    pairs = _eligible_pairs(args, labels)
    train_pairs = [item for item in pairs if item.split == "train"]
    val_pairs = [item for item in pairs if item.split == "validation"]
    if not train_pairs or not val_pairs:
        raise ValueError("Text-only baseline needs at least one eligible official train and validation example")

    # Vocabulary is fit on eligible train text only; labels never enter the encoder.
    vocabulary = build_vocabulary([item.text for item in train_pairs])
    model_cfg = FusionConfig(
        num_classes=len(labels), text_vocab_size=len(vocabulary),
        text_embedding_dim=int(cfg.get("text_embedding_dim", 256)),
        text_hidden_dim=int(cfg.get("text_hidden_dim", 256)), fusion_dim=int(cfg.get("fusion_dim", 512)),
        dropout=float(cfg.get("dropout", 0.3)), pad_id=PAD_ID, max_text_length=int(cfg.get("max_text_length", 128)),
    )
    model = TextOnlyClassifier(model_cfg)
    batch_size = int(cfg.get("batch_size", 16))
    train_ds = TextOnlyDataset(train_pairs, vocabulary, model_cfg.max_text_length)
    val_ds = TextOnlyDataset(val_pairs, vocabulary, model_cfg.max_text_length)
    train_loader = _loader(train_ds, batch_size, shuffle=True, seed=seed)
    val_loader = _loader(val_ds, batch_size, shuffle=False, seed=seed)

    counts = np.bincount([item.class_id for item in train_pairs], minlength=len(labels))
    class_weights = torch.tensor(1.0 / np.sqrt(np.maximum(counts, 1)), dtype=torch.float32)
    class_weights /= class_weights.mean()
    balance = str(cfg.get("class_balancing", "weighted_loss"))
    if balance not in {"weighted_loss", "none"}:
        raise ValueError("Text-only baseline supports class_balancing weighted_loss or none")

    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-4)),
                                  weight_decay=float(cfg.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.get("epochs", 20)))
    loss_weights = class_weights.to(device) if balance == "weighted_loss" else None
    start_epoch, best_f1, best_state, previous_duration = 0, -1.0, None, 0.0
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume.get("format") != CHECKPOINT_FORMAT or resume.get("class_labels") != labels or resume.get("vocabulary") != vocabulary:
            raise ValueError("Resume checkpoint architecture/class-order/train-vocabulary mismatch")
        if resume.get("eligibility_manifest_sha256") != hashlib.sha256(args.eligibility_manifest.read_bytes()).hexdigest():
            raise ValueError("Resume checkpoint was trained with a different eligible sample manifest")
        model.load_state_dict(resume["model_state"], strict=True)
        if resume.get("optimizer_state"):
            optimizer.load_state_dict(resume["optimizer_state"])
        if resume.get("scheduler_state"):
            scheduler.load_state_dict(resume["scheduler_state"])
        start_epoch = int(resume.get("epoch", 0))
        best_f1 = float(resume.get("best_validation_macro_f1", -1.0))
        previous_duration = float(resume.get("training_duration_seconds", 0.0))
    elif args.output.exists():
        raise FileExistsError(f"Refusing to overwrite text-only checkpoint: {args.output}; pass --resume")

    started = time.perf_counter()
    patience, stale = int(cfg.get("patience", 5)), 0
    for epoch in range(start_epoch, int(cfg.get("epochs", 20))):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in train_loader:
            tokens = batch["token_ids"].to(device, non_blocking=True)
            targets = batch["class_ids"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(tokens), targets, weight=loss_weights,
                                   label_smoothing=float(cfg.get("label_smoothing", 0.05)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            optimizer.step()
            total_loss += float(loss.detach()) * targets.numel()
            seen += targets.numel()

        val_f1 = _validation(model, val_loader, device, labels)
        print(json.dumps({"epoch": epoch + 1, "train_loss": total_loss / max(1, seen),
                          "validation_macro_f1": val_f1}))
        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = copy.deepcopy(model.state_dict())
            payload = {
                "format": CHECKPOINT_FORMAT, "fusion_kind": "text_only", "epoch": epoch + 1,
                "best_validation_macro_f1": best_f1, "model_config": model_cfg.__dict__,
                "vocabulary": vocabulary, "class_labels": labels, "model_state": best_state,
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "seed": seed,
                "split_counts": {"train_paired": len(train_pairs), "validation_paired": len(val_pairs),
                                 "test_paired": sum(1 for item in pairs if item.split == "test")},
                "text_origins": sorted({item.text_origin for item in pairs}),
                "eligible_sample_count": len(pairs),
                "eligibility_manifest_sha256": hashlib.sha256(args.eligibility_manifest.read_bytes()).hexdigest(),
                "training_duration_seconds": previous_duration + time.perf_counter() - started,
            }
            _atomic_torch_save(payload, args.output)
            digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
            manifest = args.output.with_suffix(".manifest.json")
            manifest.write_text(json.dumps({
                "fusion_kind": "text_only", "implemented": True, "trained": True,
                "checkpoint": str(args.output.resolve()), "sha256": digest,
                "class_labels": labels, "validation_macro_f1": best_f1,
                "test_evaluation_completed": False, "loaded": False,
            }, indent=2), encoding="utf-8")
        else:
            stale += 1
        if stale >= patience:
            break
        scheduler.step()

    if best_state is None:
        raise RuntimeError("Training did not produce a validation-selected checkpoint")
    elapsed = previous_duration + time.perf_counter() - started
    saved = torch.load(args.output, map_location="cpu", weights_only=False)
    saved["training_duration_seconds"] = elapsed
    saved.pop("optimizer_state", None)
    _atomic_torch_save(saved, args.output)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"training_duration_seconds": elapsed,
                     "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return args.output


def load_text_only_model(checkpoint: Path, labels: Mapping[int, str]) -> tuple[TextOnlyClassifier, dict]:
    """Load an independent text-only baseline for evaluation, validating its sidecar manifest."""
    checkpoint = Path(checkpoint)
    manifest_path = checkpoint.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Text-only manifest is required for evaluation safety: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("trained") or manifest.get("fusion_kind") != "text_only":
        raise ValueError("Text-only checkpoint manifest does not mark a completed text-only run")
    if not manifest.get("sha256") or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Text-only checkpoint SHA-256 does not match its manifest")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("fusion_kind") != "text_only":
        raise ValueError("Text-only checkpoint format is unsupported")
    if payload.get("class_labels") != labels:
        raise ValueError("Text-only checkpoint class ordering differs from the official class mapping")
    model = TextOnlyClassifier(FusionConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an independent text-only baseline on validated paired text.")
    parser.add_argument("--paired-data", type=Path, required=True,
                        help="CSV columns: sample_id,image_path,text,text_origin,provenance,source_reference")
    parser.add_argument("--eligibility-manifest", type=Path, required=True,
                        help="Duplicate-free eligible-sample manifest from scripts/build_fusion_eligibility.py")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--train-split", type=Path, default=Path("train_split.txt"))
    parser.add_argument("--validation-split", type=Path, default=Path("val_split.txt"))
    parser.add_argument("--test-split", type=Path, default=Path("test_split.txt"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.paired_data.is_file():
        parser.error(
            f"Text-only baseline blocked: paired image-text CSV does not exist at {args.paired_data}. "
            "Grammar/label prompts are target-derived and cannot be substituted."
        )
    if not args.eligibility_manifest.is_file():
        parser.error(
            f"Text-only baseline blocked: eligibility manifest does not exist at {args.eligibility_manifest}. "
            "Build it with scripts/build_fusion_eligibility.py."
        )
    print("Text-only baseline uses the same eligible samples and official splits as the fusion models.")
    print(f"Device={args.device}; paired data={args.paired_data}; output={args.output}")
    print(f"Saved validation-selected checkpoint: {train_text_only(args)}")


if __name__ == "__main__":
    main()
