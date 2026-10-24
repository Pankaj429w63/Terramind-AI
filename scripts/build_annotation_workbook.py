"""Build a blinded annotator workbook from the duplicate-free eligibility manifest.

Each eligible image gets an opaque alias and a blank observation row. No image path,
class name, label, or split is written into the workbook, so annotators cannot infer the
target. The alias -> sample_id mapping is written to a separate sealed file that must never
be shared with annotators.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.fusion.annotation import (
    build_alias_map,
    build_workbook_rows,
    load_alias_map,
    new_salt,
    write_alias_map,
    write_workbook,
)
from ml.fusion.eligibility import load_eligibility_manifest

import random
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs" / "research" / "fusion_eligibility_manifest.json"
DEFAULT_WORKBOOK = ROOT / "outputs" / "annotation" / "annotator_workbook.csv"
DEFAULT_ALIAS_MAP = ROOT / "outputs" / "annotation" / "sealed" / "alias_map.json"
DEFAULT_PILOT_DIR = ROOT / "outputs" / "annotation" / "pilot"


def select_balanced_pilot_samples(
    manifest: dict,
    *,
    seed: int = 42,
    train_per_class: int = 1,
    val_per_class: int = 1,
) -> list[str]:
    """Select a deterministic, class-balanced pilot subset from train and val splits only."""
    tr = manifest["eligible_by_split"].get("train", [])
    val = manifest["eligible_by_split"].get("validation", [])
    tr_by_class = defaultdict(list)
    val_by_class = defaultdict(list)
    for s in sorted(tr):
        tr_by_class[s.split("/")[0]].append(s)
    for s in sorted(val):
        val_by_class[s.split("/")[0]].append(s)

    rng = random.Random(seed)
    selected = []
    for c in sorted(set(tr_by_class).union(val_by_class)):
        if train_per_class > 0 and c in tr_by_class:
            selected.extend(rng.sample(tr_by_class[c], min(train_per_class, len(tr_by_class[c]))))
        if val_per_class > 0 and c in val_by_class:
            selected.extend(rng.sample(val_by_class[c], min(val_per_class, len(val_by_class[c]))))
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a blinded annotation workbook for eligible PlantWild images.")
    parser.add_argument("--eligibility-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--alias-map", type=Path, default=None,
                        help="Sealed alias map; reused when present so the workbook stays reproducible")
    parser.add_argument("--pilot", action="store_true",
                        help="Build a class-balanced pilot subset (default 178 samples: 1 train + 1 val per class)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible pilot selection")
    parser.add_argument("--train-per-class", type=int, default=1, help="Number of train samples per class for pilot")
    parser.add_argument("--val-per-class", type=int, default=1, help="Number of validation samples per class for pilot")
    args = parser.parse_args()

    manifest = load_eligibility_manifest(args.eligibility_manifest)
    if args.pilot:
        sample_ids = select_balanced_pilot_samples(
            manifest, seed=args.seed,
            train_per_class=args.train_per_class,
            val_per_class=args.val_per_class,
        )
        workbook_path = args.workbook or (DEFAULT_PILOT_DIR / "annotator_workbook.csv")
        alias_map_path = args.alias_map or (DEFAULT_PILOT_DIR / "sealed" / "alias_map.json")
    else:
        sample_ids = [sample_id for split_ids in manifest["eligible_by_split"].values() for sample_id in split_ids]
        workbook_path = args.workbook or DEFAULT_WORKBOOK
        alias_map_path = args.alias_map or DEFAULT_ALIAS_MAP

    if alias_map_path.is_file():
        alias_map, salt = load_alias_map(alias_map_path)
    else:
        salt = new_salt()
        alias_map = build_alias_map(sample_ids, salt)
        write_alias_map(alias_map, alias_map_path, salt)
    if set(alias_map.values()) != set(sample_ids):
        raise ValueError("Sealed alias map does not match the eligible samples; delete it to rebuild")

    write_workbook(build_workbook_rows(alias_map), workbook_path)

    metadata = {
        "workbook": str(workbook_path),
        "sealed_alias_map": str(alias_map_path),
        "is_pilot": args.pilot,
        "seed": args.seed if args.pilot else None,
        "eligible_samples": len(sample_ids),
        "workbook_rows": len(alias_map),
        "blinding": "workbook exposes only opaque aliases; no path, class, label, or split",
        "warning": "Do not share the sealed alias map with annotators.",
    }
    if args.pilot:
        pilot_manifest_path = workbook_path.parent / "pilot_manifest.json"
        pilot_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        pilot_manifest_path.write_text(json.dumps({
            **metadata,
            "classes_represented": len(set(s.split('/')[0] for s in sample_ids)),
            "train_samples": len([s for s in sample_ids if s in manifest['eligible_by_split'].get('train', [])]),
            "val_samples": len([s for s in sample_ids if s in manifest['eligible_by_split'].get('validation', [])]),
            "test_samples": len([s for s in sample_ids if s in manifest['eligible_by_split'].get('test', [])]),
            "sample_ids": sample_ids,
        }, indent=2), encoding="utf-8")
        metadata["pilot_manifest"] = str(pilot_manifest_path)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
