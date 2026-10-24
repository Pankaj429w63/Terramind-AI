"""Deterministic duplicate-free subset derived from untouched official splits."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from .data import PairedExample, load_official_split


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rel_key(path: str | Path) -> str:
    value = Path(str(path).replace("\\", "/")).as_posix()
    return value.casefold()


def build_eligibility_manifest(
    dataset_root: Path,
    split_files: Mapping[str, Path],
    class_labels: Mapping[int, str],
) -> dict:
    root = Path(dataset_root).resolve()
    official = load_official_split(root, split_files)
    split_hashes = {}
    for split, file_path in split_files.items():
        resolved = Path(file_path)
        if not resolved.is_absolute():
            resolved = root / resolved
        split_hashes[split] = sha256_file(resolved)

    hash_groups: dict[str, list[dict]] = defaultdict(list)
    for rel, (class_id, split) in sorted(official.items()):
        if class_id not in class_labels:
            raise ValueError(f"Unknown class id {class_id} for official image {rel}")
        image_path = (root / "images" / rel).resolve()
        if not image_path.is_relative_to((root / "images").resolve()) or not image_path.is_file():
            raise FileNotFoundError(f"Missing or unsafe official image: {rel}")
        hash_groups[sha256_file(image_path)].append({
            "sample_id": Path(rel).as_posix(), "class_id": class_id,
            "class_name": class_labels[class_id], "split": split,
        })

    included: list[dict] = []
    excluded: list[dict] = []
    reason_counts: Counter[str] = Counter()
    duplicate_group_counts = {"all": 0, "cross_split": 0, "within_split": 0}
    for digest, group in sorted(hash_groups.items(), key=lambda pair: min(x["sample_id"].casefold() for x in pair[1])):
        ordered = sorted(group, key=lambda item: item["sample_id"].casefold())
        if len(group) == 1:
            included.append({**ordered[0], "sha256": digest})
            continue
        duplicate_group_counts["all"] += 1
        split_set = {item["split"] for item in group}
        class_set = {item["class_id"] for item in group}
        if len(split_set) > 1:
            duplicate_group_counts["cross_split"] += 1
            reason = "duplicate_content_crosses_official_splits"
            excluded.extend({**item, "sha256": digest, "exclusion_reason": reason} for item in ordered)
            reason_counts[reason] += len(group)
        elif len(class_set) > 1:
            duplicate_group_counts["within_split"] += 1
            reason = "duplicate_content_has_conflicting_labels"
            excluded.extend({**item, "sha256": digest, "exclusion_reason": reason} for item in ordered)
            reason_counts[reason] += len(group)
        else:
            duplicate_group_counts["within_split"] += 1
            included.append({**ordered[0], "sha256": digest})
            reason = "duplicate_copy_within_same_split"
            excluded.extend({**item, "sha256": digest, "exclusion_reason": reason} for item in ordered[1:])
            reason_counts[reason] += len(ordered) - 1

    split_order = {name: index for index, name in enumerate(split_files)}
    included.sort(key=lambda item: (split_order[item["split"]], item["sample_id"].casefold()))
    excluded.sort(key=lambda item: (split_order[item["split"]], item["sample_id"].casefold()))
    counts = Counter(item["split"] for item in included)
    class_counts = {
        split: {str(class_id): sum(item["split"] == split and item["class_id"] == class_id for item in included)
                for class_id in sorted(class_labels)}
        for split in split_files
    }
    # A compact per-split view of the same eligible set: no image ever changes
    # official split, this only groups the retained paths by their official split.
    eligible_by_split = {
        split: [item["sample_id"] for item in included if item["split"] == split]
        for split in split_files
    }
    excluded_by_split = {
        split: [item["sample_id"] for item in excluded if item["split"] == split]
        for split in split_files
    }
    return {
        "format": "terramind-duplicate-free-eligibility-v1",
        "policy": {
            "official_assignments_are_immutable": True,
            "cross_split_duplicate_policy": "exclude every member of any exact-SHA256 group represented in more than one official split",
            "within_split_duplicate_policy": "retain lexicographically first relative path only when all copies have the same official class; exclude all copies when labels conflict",
            "comparison_rule": "image-only, text-only, early/intermediate/late fusion must all evaluate on these same eligible sample IDs",
            "hash": "SHA-256 over image file bytes",
        },
        "dataset_root": str(root),
        "class_labels": {str(index): label for index, label in sorted(class_labels.items())},
        "official_split_sha256": split_hashes,
        "official_path_rows": len(official),
        "eligible_rows": len(included),
        "eligible_counts_by_split": dict(counts),
        "eligible_class_counts_by_split": class_counts,
        "duplicate_groups": duplicate_group_counts,
        "excluded_counts_by_reason": dict(reason_counts),
        "eligible_by_split": eligible_by_split,
        "excluded_by_split": excluded_by_split,
        "eligible": included,
        "excluded": excluded,
    }


def validate_eligibility_manifest(
    manifest: dict,
    dataset_root: Path,
    split_files: Mapping[str, Path],
    class_labels: Mapping[int, str],
) -> set[str]:
    if manifest.get("format") != "terramind-duplicate-free-eligibility-v1":
        raise ValueError("Unsupported eligibility manifest format")
    expected = build_eligibility_manifest(dataset_root, split_files, class_labels)
    if manifest.get("official_split_sha256") != expected["official_split_sha256"]:
        raise ValueError("Eligibility manifest was built from different official split files")
    if manifest.get("class_labels") != expected["class_labels"]:
        raise ValueError("Eligibility manifest class mapping differs from the official mapping")
    entries = manifest.get("eligible")
    if not isinstance(entries, list):
        raise ValueError("Eligibility manifest lacks eligible sample entries")
    expected_entries = {item["sample_id"]: item for item in expected["eligible"]}
    seen: set[str] = set()
    for entry in entries:
        sample_id = entry.get("sample_id")
        trusted = expected_entries.get(sample_id)
        if not trusted or any(entry.get(key) != trusted[key] for key in ("class_id", "class_name", "split", "sha256")):
            raise ValueError(f"Eligibility sample is unsupported or altered: {sample_id}")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample in eligibility manifest: {sample_id}")
        seen.add(sample_id)
    if seen != set(expected_entries):
        raise ValueError("Eligibility manifest omits or adds eligible samples")
    return {_rel_key(item) for item in seen}


def filter_paired_examples(
    examples: Sequence[PairedExample], eligible_sample_ids: set[str]
) -> list[PairedExample]:
    selected: list[PairedExample] = []
    for example in examples:
        key = _rel_key(example.relative_path)
        if key not in eligible_sample_ids:
            continue
        selected.append(example)
    return selected


def write_eligibility_manifest(manifest: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_eligibility_manifest(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Eligibility manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "terramind-duplicate-free-eligibility-v1":
        raise ValueError("Unsupported eligibility manifest format")
    return manifest


def eligible_sample_ids(
    manifest: dict,
    dataset_root: Path,
    split_files: Mapping[str, Path],
    class_labels: Mapping[int, str],
) -> set[str]:
    """Validate a manifest against the untouched official files and return eligible keys."""
    return validate_eligibility_manifest(manifest, dataset_root, split_files, class_labels)
