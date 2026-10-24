from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.parse import urlparse

import torch
from PIL import Image
from torch.utils.data import Dataset


SPECIAL_TOKENS = ("<pad>", "<unk>", "<eos>")
PAD_ID, UNK_ID, EOS_ID = range(len(SPECIAL_TOKENS))
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
# ``model_generated_observation`` is a distinct experimental origin for candidate
# descriptions produced by an automated image-description model. It must never be
# relabelled as ``user_observation`` or ``paired_human_observation``: the three sources
# are separate experimental tracks and may not be mixed in one training run.
ALLOWED_TEXT_ORIGINS = {
    "paired_human_observation", "paired_caption", "user_observation", "model_generated_observation",
}
HUMAN_TEXT_ORIGINS = frozenset({"paired_human_observation", "paired_caption", "user_observation"})
MODEL_TEXT_ORIGIN = "model_generated_observation"


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def build_vocabulary(texts: Sequence[str]) -> dict[str, int]:
    words = sorted({word for text in texts for word in tokenize(text)})
    return {token: index for index, token in enumerate((*SPECIAL_TOKENS, *words))}


def encode_text(text: str, vocabulary: Mapping[str, int], max_length: int) -> list[int]:
    if max_length < 1:
        raise ValueError("max_length must be positive")
    tokens = [vocabulary.get(word, UNK_ID) for word in tokenize(text)]
    return [*tokens[: max_length - 1], EOS_ID]


def _key(path: str | Path) -> str:
    return Path(str(path).replace("\\", "/")).as_posix().lstrip("./").casefold()


def _normalized_label(text: str) -> str:
    return " ".join(tokenize(text))


# Bare values that carry no real collection information.
PLACEHOLDER_VALUES = frozenset({
    "unknown", "n/a", "na", "none", "tbd", "todo", "placeholder", "xxx", "sample",
    "dummy", "fake", "changeme", "replace me", "replace_me", "fill me", "fillme",
})
# Unfilled template markers: <...>, ACTUAL_*, YOUR_*, *_HERE, replace-me.
PLACEHOLDER_PATTERN = re.compile(
    r"<\s*[^<>]*\s*>|\b(actual_\w+|your_actual\w*|your_\w+|\w*_here|replace_?me|changeme)\b",
    re.IGNORECASE,
)
ISO_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Schemes that point at local storage rather than a citable record.
LOCAL_ONLY_SCHEMES = frozenset({"file", "data"})


def _is_placeholder(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return True
    if stripped.casefold().strip(" .") in PLACEHOLDER_VALUES:
        return True
    return bool(PLACEHOLDER_PATTERN.search(stripped))


def validate_source_reference(source_reference: str) -> None:
    """Require a resolvable URL or stable URI; reject placeholders and local paths."""
    value = (source_reference or "").strip()
    if not value:
        raise ValueError("source_reference is required: provide the URL or stable URI of the annotation records")
    if _is_placeholder(value):
        raise ValueError(
            f"source_reference is still a placeholder ({value!r}); provide the real URL or stable URI of the annotation records"
        )
    parsed = urlparse(value)
    scheme = parsed.scheme.casefold()
    if not scheme:
        raise ValueError(
            f"source_reference must be a URL or stable URI with a scheme, not a local path or plain text ({value!r})"
        )
    if len(scheme) == 1:
        raise ValueError(
            f"source_reference looks like a local drive path, not a source URI ({value!r}); cite a public or registry record"
        )
    if scheme in LOCAL_ONLY_SCHEMES:
        raise ValueError(
            f"source_reference must not be a local {scheme}: reference ({value!r}); cite a public or registry record"
        )
    if scheme in {"http", "https"}:
        if not parsed.netloc or "." not in parsed.netloc:
            raise ValueError(f"source_reference URL is malformed or missing a host ({value!r})")
    elif not (parsed.netloc or parsed.path.strip("/")):
        raise ValueError(f"source_reference URI has no record identifier ({value!r})")


def validate_provenance(provenance: str, source_reference: str) -> None:
    """Require real, unambiguous collection provenance plus a resolvable source reference."""
    text = (provenance or "").strip()
    if len(text) < 12:
        raise ValueError(
            "provenance is missing or too short: name the collection method, observer pool, collection dates, and instrument"
        )
    if _is_placeholder(text):
        raise ValueError(
            f"provenance is still a placeholder ({text!r}); replace ACTUAL_*/YOUR_*/<...> values with the real "
            "collection method, observer pool, collection dates, and instrument version"
        )
    if not ISO_DATE_PATTERN.search(text):
        raise ValueError(
            "provenance must record the actual collection dates as YYYY-MM-DD (for example 'collected 2026-03-04')"
        )
    validate_source_reference(source_reference)


def validate_text_not_label_derived(text: str, labels: Mapping[int, str]) -> None:
    """Reject a direct class-name string in text; provenance remains mandatory too."""
    normalized = _normalized_label(text)
    if not normalized:
        raise ValueError("Paired text is empty after tokenization")
    for label in labels.values():
        phrase = _normalized_label(label)
        if phrase and re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", normalized):
            raise ValueError("Paired text contains a PlantWild class name; target-derived text is prohibited")


@dataclass(frozen=True)
class PairedExample:
    image_path: Path
    relative_path: str
    text: str
    class_id: int
    split: str
    text_origin: str
    provenance: str
    sample_id: str
    source_reference: str


def load_official_split(dataset_root: Path, split_files: Mapping[str, Path]) -> dict[str, tuple[int, str]]:
    """Return normalized image path -> (class_id, official split)."""
    dataset_root = Path(dataset_root).resolve()
    image_to_split: dict[str, tuple[int, str]] = {}
    for split, split_file in split_files.items():
        file_path = Path(split_file)
        if not file_path.is_absolute():
            file_path = dataset_root / file_path
        if not file_path.is_file():
            raise FileNotFoundError(f"Official {split} split missing: {file_path}")
        for line_number, raw in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            fields = raw.strip().split("=")
            if len(fields) < 2:
                raise ValueError(f"Malformed split entry {file_path}:{line_number}")
            rel = fields[0].replace("\\", "/")
            if rel.casefold().startswith("images/"):
                rel = rel[7:]
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise ValueError(f"Unsafe split image path {file_path}:{line_number}")
            key = _key(rel_path)
            try:
                class_id = int(fields[1])
            except ValueError as error:
                raise ValueError(f"Invalid class id {file_path}:{line_number}") from error
            if key in image_to_split:
                raise ValueError(f"Image appears in more than one official split: {rel}")
            image_to_split[key] = (class_id, split)
    return image_to_split


def load_paired_text_csv(
    csv_path: Path,
    dataset_root: Path,
    split_files: Mapping[str, Path],
    labels: Mapping[int, str],
    verify_hashes: bool = True,
) -> list[PairedExample]:
    """Load externally supplied image/text pairs; labels come only from official split files.

    CSV columns: sample_id, image_path, text, text_origin, provenance, source_reference.
    Targets and split names are joined from the official split files, never from this CSV.
    Paths are split-checked and optional byte hashes detect duplicate images crossing splits.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Paired text CSV not found: {csv_path}")
    dataset_root = Path(dataset_root).resolve()
    images_root = (dataset_root / "images").resolve()
    official = load_official_split(dataset_root, split_files)
    examples: list[PairedExample] = []
    seen: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "image_path", "text", "text_origin", "provenance", "source_reference"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Paired CSV requires columns: {sorted(required)}")
        forbidden = {"class_id", "label", "class_name", "target", "target_label", "split", "official_split"}
        if forbidden.intersection(reader.fieldnames):
            raise ValueError("Paired CSV must not include target labels or split columns; these come from official split metadata")
        seen_sample_ids: set[str] = set()
        for line_number, row in enumerate(reader, 2):
            sample_id = (row.get("sample_id") or "").strip()
            raw_path = (row.get("image_path") or "").strip()
            text = (row.get("text") or "").strip()
            origin = (row.get("text_origin") or "").strip()
            provenance = (row.get("provenance") or "").strip()
            source_reference = (row.get("source_reference") or "").strip()
            if not sample_id or not raw_path or not text or not provenance or not source_reference:
                raise ValueError(f"Missing sample_id, image_path, text, provenance, or source_reference in {csv_path}:{line_number}")
            validate_provenance(provenance, source_reference)
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id {sample_id!r} in {csv_path}:{line_number}")
            seen_sample_ids.add(sample_id)
            if origin not in ALLOWED_TEXT_ORIGINS:
                raise ValueError(f"Unapproved text_origin {origin!r} at {csv_path}:{line_number}")
            validate_text_not_label_derived(text, labels)
            input_path = Path(raw_path)
            if input_path.is_absolute():
                absolute = input_path.resolve()
                if not absolute.is_relative_to(images_root):
                    raise ValueError(f"Paired image is outside dataset images directory: {raw_path}")
                relative = absolute.relative_to(images_root)
            else:
                relative = input_path
                if relative.parts and relative.parts[0].casefold() == "images":
                    relative = Path(*relative.parts[1:])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe paired image path: {raw_path}")
                absolute = (images_root / relative).resolve()
                if not absolute.is_relative_to(images_root):
                    raise ValueError(f"Paired image escapes dataset image directory: {raw_path}")
            key = _key(relative)
            if _key(sample_id) != key:
                raise ValueError(f"sample_id must equal the normalized image_path to prevent pair misalignment: {sample_id!r}")
            if key in seen:
                raise ValueError(f"Duplicate paired image row: {raw_path}")
            seen.add(key)
            if key not in official:
                raise ValueError(f"Image is not present in the official train/validation/test splits: {raw_path}")
            if not absolute.is_file():
                raise FileNotFoundError(f"Paired image file is missing: {absolute}")
            class_id, split = official[key]
            if class_id not in labels:
                raise ValueError(f"Official split references unknown class index {class_id}")
            examples.append(PairedExample(
                absolute, relative.as_posix(), text, class_id, split, origin, provenance, sample_id, source_reference
            ))

    if not examples:
        raise ValueError("Paired text CSV contains no usable examples")
    if verify_hashes:
        hashes: dict[str, tuple[str, str]] = {}
        for example in examples:
            digest = hashlib.sha256(example.image_path.read_bytes()).hexdigest()
            previous = hashes.get(digest)
            if previous and previous[1] != example.split:
                raise ValueError(f"Duplicate image content crosses official splits: {previous[0]} and {example.relative_path}")
            hashes[digest] = (example.relative_path, example.split)
    return examples


class PairedImageTextDataset(Dataset[dict[str, object]]):
    def __init__(self, examples: Sequence[PairedExample], vocabulary: Mapping[str, int], transform: Callable, max_length: int):
        self.examples = tuple(examples)
        self.vocabulary = dict(vocabulary)
        self.transform = transform
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        with Image.open(example.image_path) as image:
            pixels = self.transform(image.convert("RGB"))
        return {
            "image": pixels,
            "token_ids": torch.tensor(encode_text(example.text, self.vocabulary, self.max_length), dtype=torch.long),
            "class_id": example.class_id,
            "split": example.split,
            "image_path": example.relative_path,
            "sample_id": example.sample_id,
            "source_reference": example.source_reference,
        }


def collate_paired(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    width = max(item["token_ids"].numel() for item in batch)  # type: ignore[union-attr]
    token_ids = torch.full((len(batch), width), PAD_ID, dtype=torch.long)
    for index, item in enumerate(batch):
        ids = item["token_ids"]
        token_ids[index, : ids.numel()] = ids  # type: ignore[union-attr]
    return {
        "images": torch.stack([item["image"] for item in batch]),  # type: ignore[list-item]
        "token_ids": token_ids,
        "class_ids": torch.tensor([int(item["class_id"]) for item in batch], dtype=torch.long),
        "image_paths": [str(item["image_path"]) for item in batch],
        "sample_ids": [str(item["sample_id"]) for item in batch],
        "source_references": [str(item["source_reference"]) for item in batch],
    }
