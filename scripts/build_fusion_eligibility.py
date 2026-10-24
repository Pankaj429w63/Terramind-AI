"""Build deterministic duplicate-free manifests without changing official splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.fusion.eligibility import build_eligibility_manifest, write_eligibility_manifest
from ml.models.class_labels import load_class_labels

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "plantwild" / "plantwild"
DEFAULT_OUTPUT = ROOT / "outputs" / "research" / "fusion_eligibility_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SHA-256 eligible sample lists per official PlantWild split.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--train-split", type=Path, default=Path("train_split.txt"))
    parser.add_argument("--validation-split", type=Path, default=Path("val_split.txt"))
    parser.add_argument("--test-split", type=Path, default=Path("test_split.txt"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    split_files = {"train": args.train_split, "validation": args.validation_split, "test": args.test_split}
    manifest = build_eligibility_manifest(
        args.dataset_root, split_files, load_class_labels(args.dataset_root / "classes.txt")
    )
    write_eligibility_manifest(manifest, args.output)
    print(json.dumps({
        "output": str(args.output), "official_rows": manifest["official_path_rows"],
        "eligible_rows": manifest["eligible_rows"],
        "eligible_counts_by_split": manifest["eligible_counts_by_split"],
        "duplicate_groups": manifest["duplicate_groups"],
        "excluded_counts_by_reason": manifest["excluded_counts_by_reason"],
    }, indent=2))


if __name__ == "__main__":
    main()
