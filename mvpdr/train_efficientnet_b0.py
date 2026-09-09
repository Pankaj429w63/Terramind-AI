"""Controlled EfficientNet-B0 experiment for PlantWild's official 89-class split.

Safety rules:
  * Uses only the dataset-provided mode labels: 1=train, 2=validation, 0=test.
  * Never writes to models/, models/archive/, or a shared checkpoint directory.
  * Requires --confirm-train and a unique --run-name before any training starts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "plantwild" / "plantwild"
EXPERIMENTS_ROOT = PROJECT_ROOT / "outputs" / "experiments"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", help="Unique name under outputs/experiments/.")
    parser.add_argument("--confirm-train", action="store_true",
                        help="Required safety switch before training can begin.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.002,
                        help="Minimum validation macro-F1 gain required to reset patience.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_classes() -> dict[str, int]:
    classes: dict[str, int] = {}
    with (DATASET_ROOT / "classes.txt").open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                class_id, class_name = raw.split(" ", 1)
                classes[class_name] = int(class_id)
    if len(classes) != 89:
        raise RuntimeError(f"Expected 89 PlantWild classes; found {len(classes)}.")
    return classes


def load_official_split() -> tuple[list[str], list[str], list[str]]:
    split_file = DATASET_ROOT / "trainval.txt"
    splits = {"0": [], "1": [], "2": []}
    with split_file.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                _, _, mode = line.rsplit("=", 2)
            except ValueError as exc:
                raise ValueError(f"Malformed PlantWild split entry: {line!r}") from exc
            if mode not in splits:
                raise ValueError(f"Unknown PlantWild split mode {mode!r}: {line!r}")
            splits[mode].append(line)

    train, val, test = splits["1"], splits["2"], splits["0"]
    path_sets = [{line.split("=", 1)[0] for line in values} for values in (train, val, test)]
    if path_sets[0] & path_sets[1] or path_sets[0] & path_sets[2] or path_sets[1] & path_sets[2]:
        raise RuntimeError("Official PlantWild split contains an image-path overlap.")
    if (len(train), len(val), len(test)) != (13045, 1820, 3677):
        raise RuntimeError(
            "Unexpected official split counts; expected train=13045, val=1820, test=3677; "
            f"got train={len(train)}, val={len(val)}, test={len(test)}."
        )
    return train, val, test


class PlantWildDataset(Dataset):
    def __init__(self, lines: list[str], class_to_id: dict[str, int], transform: T.Compose):
        self.transform = transform
        self.items: list[tuple[Path, int]] = []
        missing: list[str] = []
        for line in lines:
            image_rel = line.split("=", 1)[0]
            class_name = Path(image_rel).parts[0]
            image_path = DATASET_ROOT / "images" / image_rel
            if class_name not in class_to_id:
                raise RuntimeError(f"Unknown class in split: {class_name!r}")
            if not image_path.is_file():
                missing.append(str(image_path))
                continue
            self.items.append((image_path, class_to_id[class_name]))
        if missing:
            raise FileNotFoundError(f"{len(missing)} listed images are missing; first: {missing[0]}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.items[index]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label


def build_model(dropout: float, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(features, 89))
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def make_optimizer(model: nn.Module, args: argparse.Namespace, backbone_trainable: bool) -> optim.Optimizer:
    if not backbone_trainable:
        return optim.AdamW(model.classifier.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
    return optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": args.backbone_lr},
            {"params": model.classifier.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )


def make_weighted_sampler(dataset: PlantWildDataset) -> WeightedRandomSampler:
    """Balance class exposure in the training loader only, with replacement."""
    class_counts = Counter(label for _, label in dataset.items)
    weights = torch.as_tensor(
        [1.0 / class_counts[label] for _, label in dataset.items], dtype=torch.double
    )
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def macro_f1(predictions: list[int], targets: list[int], num_classes: int = 89) -> float:
    values = []
    for class_id in range(num_classes):
        tp = sum(pred == class_id and actual == class_id for pred, actual in zip(predictions, targets))
        fp = sum(pred == class_id and actual != class_id for pred, actual in zip(predictions, targets))
        fn = sum(pred != class_id and actual == class_id for pred, actual in zip(predictions, targets))
        denominator = 2 * tp + fp + fn
        values.append((2 * tp / denominator) if denominator else 0.0)
    return float(np.mean(values))


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device, accum_steps: int,
                epoch: int) -> tuple[float, float]:
    model.train()
    total = correct = 0
    loss_sum = 0.0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        (loss / accum_steps).backward()
        if (batch_index + 1) % accum_steps == 0 or batch_index + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
        done = batch_index + 1
        if done == 1 or done % 10 == 0 or done == len(loader):
            print(
                f"[train] epoch {epoch} batch {done}/{len(loader)} "
                f"loss={loss_sum / total:.4f} acc={correct / total:.4f}",
                flush=True,
            )
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    predictions: list[int] = []
    targets: list[int] = []
    for batch_index, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * images.size(0)
        batch_predictions = logits.argmax(dim=1)
        correct += (batch_predictions == labels).sum().item()
        total += images.size(0)
        predictions.extend(batch_predictions.cpu().tolist())
        targets.extend(labels.cpu().tolist())
        done = batch_index + 1
        if done == 1 or done % 10 == 0 or done == len(loader):
            print(
                f"[eval] batch {done}/{len(loader)} loss={loss_sum / total:.4f} acc={correct / total:.4f}",
                flush=True,
            )
    return loss_sum / total, correct / total, macro_f1(predictions, targets)


def prepare_run_directory(args: argparse.Namespace) -> Path:
    if not args.run_name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name):
        raise ValueError("--run-name is required and may contain only letters, digits, underscores, and hyphens.")
    run_dir = EXPERIMENTS_ROOT / args.run_name
    if run_dir.exists():
        leftover = {path.name for path in run_dir.iterdir()}
        if leftover and leftover != {"config.json"}:
            raise FileExistsError(f"Refusing to overwrite existing experiment directory: {run_dir}")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    args = parse_args()
    class_to_id = load_classes()
    train_lines, val_lines, test_lines = load_official_split()
    plan = {
        "experiment": "EfficientNet-B0 pretrained PlantWild official split",
        "split": {"train": len(train_lines), "validation": len(val_lines), "test": len(test_lines)},
        "num_classes": len(class_to_id),
        "model": "torchvision EfficientNet_B0_Weights.DEFAULT",
        "output_policy": "unique outputs/experiments/<run-name>; never writes models/ or models/archive/",
        "training_requires": "--confirm-train plus a unique --run-name",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.confirm_train:
        print("Dry safety check complete. No training or files were created.", flush=True)
        return

    run_dir = prepare_run_directory(args)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = vars(args) | {"device": str(device), "official_split": plan["split"]}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Training started run={args.run_name} device={device} epochs={args.epochs}", flush=True)
    print(f"Writing checkpoints only to {run_dir}", flush=True)

    train_transform = T.Compose([
        T.Resize((256, 256)), T.RandomCrop((args.img_size, args.img_size)),
        T.RandomHorizontalFlip(), T.RandomRotation(15),
        T.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15),
        T.ToTensor(), T.RandomErasing(p=0.15, scale=(0.02, 0.15), value="random"),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = T.Compose([
        T.Resize((256, 256)), T.CenterCrop((args.img_size, args.img_size)),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = PlantWildDataset(train_lines, class_to_id, train_transform)
    val_ds = PlantWildDataset(val_lines, class_to_id, eval_transform)
    test_ds = PlantWildDataset(test_lines, class_to_id, eval_transform)
    train_sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.dropout).to(device)
    set_backbone_trainable(model, trainable=False)
    optimizer = make_optimizer(model, args, backbone_trainable=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_f1, stale, history = -1.0, 0, []

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            optimizer = make_optimizer(model, args, backbone_trainable=True)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
        print(f"[epoch {epoch}/{args.epochs}] starting (backbone_frozen={epoch <= args.freeze_epochs})", flush=True)
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args.accum_steps, epoch
        )
        print(f"[val] epoch {epoch} evaluating...", flush=True)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_f1)
        entry = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                 "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if val_f1 > best_f1 + args.min_delta:
            best_f1, stale = val_f1, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_macro_f1": val_f1,
                        "class_to_id": class_to_id, "config": config}, run_dir / "best_model.pth")
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(run_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    results = {"best_epoch": checkpoint["epoch"], "best_val_macro_f1": checkpoint["val_macro_f1"],
               "test_loss": test_loss, "test_acc": test_acc, "test_macro_f1": test_f1, "history": history}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    main()
