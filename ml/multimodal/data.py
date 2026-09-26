from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from ml.models.class_labels import load_class_labels


SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
PAD_ID, UNK_ID, BOS_ID, EOS_ID = range(len(SPECIAL_TOKENS))
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def metadata_tokens(label: str) -> list[str]:
    """Tokenize the supplied class metadata; never synthesize a prose description."""
    return TOKEN_PATTERN.findall(label.lower())


def build_vocabulary(labels: Mapping[int, str]) -> dict[str, int]:
    tokens = sorted({token for label in labels.values() for token in metadata_tokens(label)})
    return {token: index for index, token in enumerate((*SPECIAL_TOKENS, *tokens))}


def encode_metadata(label: str, vocabulary: Mapping[str, int], max_length: int = 32) -> list[int]:
    if max_length < 2:
        raise ValueError("max_length must leave room for BOS and EOS")
    body = [vocabulary.get(token, UNK_ID) for token in metadata_tokens(label)]
    return [BOS_ID, *body[: max_length - 2], EOS_ID]


@dataclass(frozen=True)
class ImageClassPair:
    image_path: Path
    class_id: int
    metadata_text: str


def discover_pairs(
    dataset_root: Path,
    split_file: Path,
    labels: Mapping[int, str] | None = None,
) -> list[ImageClassPair]:
    """Read PlantWild's image-to-class split; class labels are the only text modality."""
    dataset_root = Path(dataset_root).resolve()
    split_file = Path(split_file)
    if not split_file.is_absolute():
        split_file = dataset_root / split_file
    split_file = split_file.resolve()
    if not split_file.is_file():
        raise FileNotFoundError(f"PlantWild split file not found: {split_file}")
    class_labels = dict(labels or load_class_labels(dataset_root / "classes.txt"))
    images_root = (dataset_root / "images").resolve()
    records: list[ImageClassPair] = []
    for line_number, raw in enumerate(split_file.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split("=")
        if len(fields) < 2:
            raise ValueError(f"Malformed PlantWild split row at {split_file}:{line_number}")
        relative = Path(fields[0])
        if relative.parts and relative.parts[0].lower() == "images":
            relative = Path(*relative.parts[1:])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe image path at {split_file}:{line_number}")
        try:
            class_id = int(fields[1])
        except ValueError as error:
            raise ValueError(f"Invalid class ID at {split_file}:{line_number}") from error
        if class_id not in class_labels:
            raise ValueError(f"Unknown class ID {class_id} at {split_file}:{line_number}")
        image_path = (images_root / relative).resolve()
        if not image_path.is_relative_to(images_root):
            raise ValueError(f"Image path escapes dataset images directory at {split_file}:{line_number}")
        if not image_path.is_file():
            raise FileNotFoundError(f"PlantWild image listed in split is missing: {image_path}")
        records.append(ImageClassPair(image_path, class_id, class_labels[class_id]))
    if not records:
        raise ValueError(f"PlantWild split has no usable image/class pairs: {split_file}")
    return records


class PlantWildMetadataDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        records: Sequence[ImageClassPair],
        vocabulary: Mapping[str, int],
        transform: Callable[[Image.Image], torch.Tensor],
        max_text_length: int = 32,
    ) -> None:
        self.records = tuple(records)
        self.vocabulary = dict(vocabulary)
        self.transform = transform
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = self.transform(source.convert("RGB"))
        return {
            "image": image,
            "text_ids": torch.tensor(encode_metadata(record.metadata_text, self.vocabulary, self.max_text_length), dtype=torch.long),
            "class_id": record.class_id,
            "metadata_text": record.metadata_text,
            "image_path": str(record.image_path),
        }


def collate_metadata_pairs(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    max_length = max(len(item["text_ids"]) for item in batch)  # type: ignore[arg-type]
    texts = torch.full((len(batch), max_length), PAD_ID, dtype=torch.long)
    for row, item in enumerate(batch):
        ids = item["text_ids"]
        assert isinstance(ids, torch.Tensor)
        texts[row, : ids.numel()] = ids
    return {
        "images": torch.stack([item["image"] for item in batch]),  # type: ignore[list-item]
        "text_ids": texts,
        "class_ids": torch.tensor([int(item["class_id"]) for item in batch], dtype=torch.long),
        "metadata_text": [str(item["metadata_text"]) for item in batch],
        "image_paths": [str(item["image_path"]) for item in batch],
    }
