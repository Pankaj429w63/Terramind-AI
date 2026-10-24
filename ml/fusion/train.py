from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
import yaml

from ml.models.class_labels import load_class_labels
from ml.models.model_loader import DEFAULT_MODEL_PATH, load_checkpoint
from ml.serving.preprocessing import build_inference_transform
from .data import PAD_ID, PairedImageTextDataset, build_vocabulary, collate_paired, load_paired_text_csv
from .eligibility import eligible_sample_ids, filter_paired_examples, load_eligibility_manifest
from .models import FusionConfig, FusionClassifierBase, LateFusionClassifier, build_fusion_model

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "data" / "plantwild" / "plantwild"
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
DEFAULT_OUTPUT = ROOT / "outputs" / "fusion"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(dataset: PairedImageTextDataset, batch_size: int, *, shuffle: bool, seed: int, sampler=None) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
                      sampler=sampler, num_workers=0, collate_fn=collate_paired,
                      pin_memory=torch.cuda.is_available(), generator=generator)


def _probabilities(model: FusionClassifierBase, images: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    if isinstance(model, LateFusionClassifier):
        weight = model.validation_image_weight
        if weight is None:
            raise RuntimeError("Late fusion requires a validation-selected image weight")
        return model.predict_proba(images, token_ids, weight)
    return model(images, token_ids).softmax(dim=1)


def _validation(model: FusionClassifierBase, loader: DataLoader, device: torch.device, labels: dict[int, str]) -> tuple[float, dict]:
    model.eval()
    true, probs = [], []
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            tokens = batch["token_ids"].to(device, non_blocking=True)
            prob = _probabilities(model, images, tokens)
            probs.append(prob.cpu())
            true.append(batch["class_ids"])
    if not true:
        raise RuntimeError("No validation examples are available")
    y_true = torch.cat(true).numpy()
    y_prob = torch.cat(probs).numpy()
    y_pred = y_prob.argmax(1)
    score = float(f1_score(y_true, y_pred, average="macro", labels=list(labels), zero_division=0))
    return score, {"targets": y_true, "probabilities": y_prob}


def _select_late_weight(model: LateFusionClassifier, loader: DataLoader, device: torch.device, labels: dict[int, str]) -> tuple[float, float]:
    model.eval()
    targets, image_probs, text_probs = [], [], []
    with torch.inference_mode():
        for batch in loader:
            image_logits, text_logits = model.modality_logits(
                batch["images"].to(device), batch["token_ids"].to(device)
            )
            targets.extend(batch["class_ids"].tolist())
            image_probs.append(image_logits.softmax(1).cpu())
            text_probs.append(text_logits.softmax(1).cpu())
    if not targets:
        raise RuntimeError("No validation examples are available for late-fusion weight selection")
    y_true = np.asarray(targets)
    img = torch.cat(image_probs)
    txt = torch.cat(text_probs)
    best_weight, best_f1 = 0.5, -1.0
    for weight in np.linspace(0.0, 1.0, 21):
        y_pred = model.fuse_probabilities(img, txt, float(weight)).argmax(1).numpy()
        score = float(f1_score(y_true, y_pred, average="macro", labels=list(labels), zero_division=0))
        if score > best_f1 or (score == best_f1 and abs(weight - 0.5) < abs(best_weight - 0.5)):
            best_weight, best_f1 = float(weight), score
    model.validation_image_weight = best_weight
    return best_weight, best_f1


def _atomic_torch_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _write_manifest(path: Path, payload: dict) -> None:
    manifest = path.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train(args: argparse.Namespace) -> Path:
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    labels = load_class_labels(args.dataset_root / "classes.txt")
    split_files = {"train": args.train_split, "validation": args.validation_split, "test": args.test_split}
    pairs = load_paired_text_csv(args.paired_data, args.dataset_root, split_files, labels)
    # All compared systems (image-only, text-only, early/intermediate/late) must use
    # the same duplicate-free eligible samples. The manifest is re-derived from the
    # untouched official files, so no eligible sample can be smuggled in or dropped.
    eligibility = load_eligibility_manifest(args.eligibility_manifest)
    eligible = eligible_sample_ids(eligibility, args.dataset_root, split_files, labels)
    pairs = filter_paired_examples(pairs, eligible)
    train_pairs = [item for item in pairs if item.split == "train"]
    val_pairs = [item for item in pairs if item.split == "validation"]
    if not train_pairs or not val_pairs:
        raise ValueError("Paired dataset needs at least one official train and validation example")

    # The vocabulary is fit on train text only. Labels are carried separately
    # from the official split; they never enter the text encoder.
    vocabulary = build_vocabulary([item.text for item in train_pairs])
    model_cfg = FusionConfig(
        num_classes=len(labels), image_feature_dim=1280, text_vocab_size=len(vocabulary),
        text_embedding_dim=int(cfg.get("text_embedding_dim", 256)),
        text_hidden_dim=int(cfg.get("text_hidden_dim", 256)), fusion_dim=int(cfg.get("fusion_dim", 512)),
        dropout=float(cfg.get("dropout", 0.3)), pad_id=PAD_ID,
        max_text_length=int(cfg.get("max_text_length", 128)),
    )
    production_model, model_info = load_checkpoint(args.production_checkpoint, args.dataset_root / "classes.txt", "cpu")
    if (model_info["backbone"] != "EfficientNet-B0" or model_info["num_classes"] != len(labels)
            or model_info["labels"] != labels):
        raise ValueError("Fusion initialization requires the compatible 89-class EfficientNet-B0 checkpoint")
    encoder = copy.deepcopy(production_model.features)
    del production_model
    model = build_fusion_model(args.kind, encoder, model_cfg)
    freeze_image = bool(cfg.get("freeze_image_encoder", True))
    if freeze_image:
        for parameter in model.image_encoder.parameters():
            parameter.requires_grad_(False)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10),
        transforms.ToTensor(), transforms.Normalize(model_info["mean"], model_info["std"]),
    ])
    eval_transform = build_inference_transform("EfficientNet-B0", 224, model_info["mean"], model_info["std"])
    train_ds = PairedImageTextDataset(train_pairs, vocabulary, train_transform, model_cfg.max_text_length)
    val_ds = PairedImageTextDataset(val_pairs, vocabulary, eval_transform, model_cfg.max_text_length)
    batch_size = int(cfg.get("batch_size", 16))
    balance = str(cfg.get("class_balancing", "weighted_loss"))
    counts = np.bincount([item.class_id for item in train_pairs], minlength=len(labels))
    class_weights = torch.tensor(1.0 / np.sqrt(np.maximum(counts, 1)), dtype=torch.float32)
    class_weights /= class_weights.mean()
    sampler = None
    if balance == "weighted_sampler":
        sample_weights = [1.0 / max(counts[item.class_id], 1) for item in train_pairs]
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)
    elif balance not in {"weighted_loss", "none"}:
        raise ValueError("class_balancing must be weighted_loss, weighted_sampler, or none")
    train_loader = _loader(train_ds, batch_size, shuffle=True, seed=seed, sampler=sampler)
    val_loader = _loader(val_ds, batch_size, shuffle=False, seed=seed)

    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=float(cfg.get("learning_rate", 1e-4)),
                                  weight_decay=float(cfg.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.get("epochs", 20)))
    loss_weights = class_weights.to(device) if balance == "weighted_loss" else None
    start_epoch, best_f1, best_state, previous_duration = 0, -1.0, None, 0.0
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume.get("fusion_kind") != args.kind or resume.get("class_labels") != labels or resume.get("vocabulary") != vocabulary:
            raise ValueError("Resume checkpoint architecture/class-order/train-vocabulary mismatch")
        if resume.get("production_checkpoint_sha256") != hashlib.sha256(args.production_checkpoint.read_bytes()).hexdigest():
            raise ValueError("Resume checkpoint was initialized from a different production EfficientNet checkpoint")
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
        raise FileExistsError(f"Refusing to overwrite fusion checkpoint: {args.output}; pass --resume")

    if start_epoch >= int(cfg.get("epochs", 20)):
        raise ValueError("Checkpoint already reached configured epoch count")
    started = time.perf_counter()
    patience, stale = int(cfg.get("patience", 5)), 0
    for epoch in range(start_epoch, int(cfg.get("epochs", 20))):
        model.train()
        if freeze_image:
            model.image_encoder.eval()
        total_loss, seen = 0.0, 0
        for batch in train_loader:
            images = batch["images"].to(device, non_blocking=True)
            tokens = batch["token_ids"].to(device, non_blocking=True)
            targets = batch["class_ids"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, tokens)
            loss = F.cross_entropy(logits, targets, weight=loss_weights, label_smoothing=float(cfg.get("label_smoothing", 0.05)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            optimizer.step()
            total_loss += float(loss.detach()) * targets.numel()
            seen += targets.numel()

        if isinstance(model, LateFusionClassifier):
            val_weight, val_f1 = _select_late_weight(model, val_loader, device, labels)
        else:
            val_f1, _ = _validation(model, val_loader, device, labels)
            val_weight = None
        print(json.dumps({"epoch": epoch + 1, "train_loss": total_loss / max(1, seen),
                          "validation_macro_f1": val_f1, "late_image_weight": val_weight}))
        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = copy.deepcopy(model.state_dict())
            payload = {
                "format": "terramind-fusion-v1", "fusion_kind": args.kind,
                "epoch": epoch + 1, "best_validation_macro_f1": best_f1,
                "validation_image_weight": val_weight, "model_config": model_cfg.__dict__,
                "vocabulary": vocabulary, "class_labels": labels, "model_state": best_state,
                "optimizer_state": optimizer.state_dict(), "seed": seed,
                "scheduler_state": scheduler.state_dict(),
                "split_counts": {"train_paired": len(train_pairs), "validation_paired": len(val_pairs),
                                 "test_paired": sum(1 for item in pairs if item.split == "test")},
                "text_origins": sorted({item.text_origin for item in pairs}),
                "eligible_sample_count": len(pairs),
                "eligibility_manifest_sha256": hashlib.sha256(args.eligibility_manifest.read_bytes()).hexdigest(),
                "freeze_image_encoder": freeze_image,
                "production_checkpoint_sha256": hashlib.sha256(args.production_checkpoint.read_bytes()).hexdigest(),
                "training_duration_seconds": previous_duration + time.perf_counter() - started,
            }
            _atomic_torch_save(payload, args.output)
            digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
            _write_manifest(args.output, {
                "fusion_kind": args.kind, "implemented": True, "trained": True,
                "checkpoint": str(args.output.resolve()), "sha256": digest,
                "class_labels": labels, "validation_macro_f1": best_f1,
                "test_evaluation_completed": False, "loaded": False,
            })
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
    _write_manifest(args.output, manifest)
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one fusion classifier on validated paired non-target text.")
    parser.add_argument("--kind", choices=("early", "intermediate", "late"), required=True)
    parser.add_argument("--paired-data", type=Path, required=True,
                        help="CSV columns: sample_id,image_path,text,text_origin,provenance,source_reference")
    parser.add_argument("--eligibility-manifest", type=Path, required=True,
                        help="Duplicate-free eligible-sample manifest from scripts/build_fusion_eligibility.py")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--train-split", type=Path, default=Path("train_split.txt"))
    parser.add_argument("--validation-split", type=Path, default=Path("val_split.txt"))
    parser.add_argument("--test-split", type=Path, default=Path("test_split.txt"))
    parser.add_argument("--production-checkpoint", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output = args.output or (DEFAULT_OUTPUT / args.kind / "best_model.pth")
    if not args.paired_data.is_file():
        parser.error(
            f"Training blocked: paired image-text CSV does not exist at {args.paired_data}. "
            "PlantWild class labels/prompts are target-derived and cannot be substituted."
        )
    if not args.eligibility_manifest.is_file():
        parser.error(
            f"Training blocked: duplicate-free eligibility manifest does not exist at {args.eligibility_manifest}. "
            "Build it with scripts/build_fusion_eligibility.py so every baseline uses the same eligible samples."
        )
    print("Text must be independently observed/captioned for each image and available at inference; class labels/prompts are rejected.")
    print(f"Device={args.device}; paired data={args.paired_data}; experimental output={args.output}")
    print(f"Saved validation-selected checkpoint: {train(args)}")


if __name__ == "__main__":
    main()
