from __future__ import annotations

from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_PATH = PROJECT_ROOT / "data" / "plantwild" / "plantwild" / "classes.txt"


def load_class_labels(path: Path = DEFAULT_LABELS_PATH) -> dict[int, str]:
    """Load the PlantWild class index file in the training format: ``index label``."""
    if not path.is_file():
        raise FileNotFoundError(f"Class labels file not found: {path}")

    labels: dict[int, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid class label at {path}:{line_number}: {raw_line!r}")
        index = int(parts[0])
        if index in labels:
            raise ValueError(f"Duplicate class index {index} in {path}")
        labels[index] = parts[1]

    if not labels:
        raise ValueError(f"No class labels found in {path}")
    return labels


def normalize_class_labels(labels: Mapping[int | str, str]) -> dict[int, str]:
    """Normalize checkpoint label mappings whose keys may be serialized as strings."""
    normalized = {int(index): label for index, label in labels.items()}
    if sorted(normalized) != list(range(len(normalized))):
        raise ValueError("Class labels must contain contiguous indexes starting at zero")
    return normalized
