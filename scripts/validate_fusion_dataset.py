"""Runnable dataset validation for the three fusion models.

Validates, without training:

* the untouched official train/validation/test split files,
* the duplicate-free eligibility manifest against those same official files,
* optionally, a paired image/text CSV (structure, provenance, label leakage,
  duplicate IDs, cross-split isolation),
* per-split coverage of the paired data against the eligible sample set,
* the excluded-row reasons recorded in the eligibility manifest.

It reports the real state and never fabricates coverage. Exit code is 0 by
default (informational), or 1 with ``--strict`` if a hard validation step fails.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ml.fusion.data import ALLOWED_TEXT_ORIGINS, HUMAN_TEXT_ORIGINS, load_official_split, load_paired_text_csv
from ml.fusion.eligibility import eligible_sample_ids, load_eligibility_manifest
from ml.models.class_labels import load_class_labels

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "plantwild" / "plantwild"
DEFAULT_ELIGIBILITY = ROOT / "outputs" / "research" / "fusion_eligibility_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "research" / "fusion_dataset_validation.json"


def _rel_key(path: str) -> str:
    return Path(str(path).replace("\\", "/")).as_posix().lstrip("./").casefold()


def validate(
    dataset_root: Path,
    split_files: dict[str, Path],
    eligibility_manifest: Path,
    paired_csv: Path | None = None,
) -> dict:
    dataset_root = Path(dataset_root).resolve()
    result: dict = {
        "dataset_root": str(dataset_root),
        "report": "fusion_dataset_validation",
        "hard_errors": [],
        "paired_csv": str(paired_csv) if paired_csv else None,
        "eligible": {},
        "official": {},
    }
    labels = load_class_labels(dataset_root / "classes.txt")

    # 1. Official splits: validates path uniqueness across the untouched files.
    try:
        official = load_official_split(dataset_root, split_files)
    except (OSError, ValueError, FileNotFoundError) as error:
        result["hard_errors"].append(f"official_splits: {type(error).__name__}: {error}")
        return result
    official_counts = Counter(split for _, split in official.values())
    result["official"] = {
        "path_rows": len(official),
        "counts_by_split": {name: official_counts.get(name, 0) for name in split_files},
        "classes": len(labels),
    }

    # 2. Eligibility manifest: validated by recomputing it from the same files.
    try:
        manifest = load_eligibility_manifest(eligibility_manifest)
        eligible = eligible_sample_ids(manifest, dataset_root, split_files, labels)
    except (OSError, ValueError, FileNotFoundError) as error:
        result["hard_errors"].append(f"eligibility_manifest: {type(error).__name__}: {error}")
        return result
    eligible_by_split = Counter(item["split"] for item in manifest["eligible"])
    result["eligible"] = {
        "manifest": str(Path(eligibility_manifest).resolve()),
        "sample_count": len(eligible),
        "counts_by_split": {name: eligible_by_split.get(name, 0) for name in split_files},
        "duplicate_groups": manifest.get("duplicate_groups"),
        "excluded_counts_by_reason": manifest.get("excluded_counts_by_reason"),
        "policy": manifest.get("policy"),
    }

    # 3. Optional paired text CSV coverage against the eligible set.
    if paired_csv is not None:
        result["paired_data"] = _validate_paired(
            Path(paired_csv), dataset_root, split_files, labels, eligible, result["eligible"]["counts_by_split"]
        )
    result["status"] = "VALID" if not result["hard_errors"] else "INVALID"
    return result


def _validate_paired(
    paired_csv: Path,
    dataset_root: Path,
    split_files: dict[str, Path],
    labels,
    eligible: set[str],
    eligible_counts: dict[str, int],
) -> dict:
    report: dict = {"path": str(paired_csv.resolve()) if paired_csv.is_file() else str(paired_csv)}
    if not paired_csv.is_file():
        report.update({"exists": False, "valid": False, "reason": "paired CSV file is not present"})
        return report
    report["exists"] = True
    try:
        pairs = load_paired_text_csv(paired_csv, dataset_root, split_files, labels)
    except (OSError, ValueError, FileNotFoundError) as error:
        report.update({"valid": False, "reason": f"{type(error).__name__}: {error}"})
        return report

    by_split = Counter(pair.split for pair in pairs)
    origins = Counter(pair.text_origin for pair in pairs)
    off_manifest = [pair.relative_path for pair in pairs if _rel_key(pair.relative_path) not in eligible]
    seen_ids = {pair.sample_id for pair in pairs}
    texts = {" ".join(pair.text.casefold().split()) for pair in pairs}

    coverage = {}
    for split, count in eligible_counts.items():
        covered = sum(1 for pair in pairs if pair.split == split)
        coverage[split] = {
            "eligible_samples": count,
            "paired_rows": covered,
            "fraction": round(covered / count, 4) if count else 0.0,
        }
    empty = sorted(name for name in split_files if by_split.get(name, 0) == 0)

    report.update({
        "valid": True,
        "rows": len(pairs),
        "unique_sample_ids": len(seen_ids),
        "duplicate_sample_ids": len(pairs) - len(seen_ids),
        "unique_texts": len(texts),
        "repeated_text_rows": len(pairs) - len(texts),
        "rows_by_split": {name: by_split.get(name, 0) for name in split_files},
        "empty_splits": empty,
        "coverage_by_split": coverage,
        "text_origins": dict(origins),
        "unapproved_origins": sorted(set(origins) - ALLOWED_TEXT_ORIGINS),
        "human_origin_only": set(origins) <= HUMAN_TEXT_ORIGINS if origins else False,
        "rows_outside_eligible_manifest": len(off_manifest),
        "off_manifest_examples": sorted(off_manifest)[:10],
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate official splits, eligibility manifest, and paired text coverage.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--train-split", type=Path, default=Path("train_split.txt"))
    parser.add_argument("--validation-split", type=Path, default=Path("val_split.txt"))
    parser.add_argument("--test-split", type=Path, default=Path("test_split.txt"))
    parser.add_argument("--eligibility-manifest", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument("--paired-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true", help="Exit 1 when a hard validation step fails")
    args = parser.parse_args()
    split_files = {"train": args.train_split, "validation": args.validation_split, "test": args.test_split}
    report = validate(args.dataset_root, split_files, args.eligibility_manifest, args.paired_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")
    if args.strict and report["status"] != "VALID":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
