"""Read-only audit of PlantWild inputs for supervised image/text fusion."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASET_ROOT = ROOT / "data" / "plantwild" / "plantwild"
DEFAULT_OUTPUT = ROOT / "outputs" / "research" / "fusion_data_audit.json"


def _normalized(path: str) -> str:
    value = path.replace("\\", "/")
    return value[7:] if value.casefold().startswith("images/") else value


def audit(dataset_root: Path = DATASET_ROOT, paired_csv: Path | None = None) -> dict:
    dataset_root = Path(dataset_root).resolve()
    classes = {}
    for line in (dataset_root / "classes.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            class_id, name = line.split(" ", 1)
            classes[int(class_id)] = name.strip()

    split_files = {"train": "train_split.txt", "validation": "val_split.txt", "test": "test_split.txt"}
    rows: list[tuple[str, int, str]] = []
    split_counts: dict[str, int] = {}
    class_counts: Counter[int] = Counter()
    split_class_counts: dict[str, Counter[int]] = {}
    path_owners: dict[str, str] = {}
    for split, filename in split_files.items():
        current: Counter[int] = Counter()
        split_rows = 0
        for line_number, line in enumerate((dataset_root / filename).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split("=")
            if len(fields) < 2:
                raise ValueError(f"Malformed {filename}:{line_number}")
            path, class_id = _normalized(fields[0]), int(fields[1])
            if class_id not in classes:
                raise ValueError(f"Unknown class id {class_id} in {filename}:{line_number}")
            key = path.casefold()
            if key in path_owners:
                raise ValueError(f"Official image path occurs in multiple splits: {path}")
            path_owners[key] = split
            rows.append((path, class_id, split))
            split_rows += 1
            current[class_id] += 1
            class_counts[class_id] += 1
        split_counts[split] = split_rows
        split_class_counts[split] = current

    hashes: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    missing: list[str] = []
    corrupt: list[dict[str, str]] = []
    for rel_path, class_id, split in rows:
        path = (dataset_root / "images" / rel_path).resolve()
        if not path.is_relative_to((dataset_root / "images").resolve()):
            raise ValueError(f"Unsafe official split path: {rel_path}")
        if not path.is_file():
            missing.append(rel_path)
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[digest].append((rel_path, split, class_id))
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:  # noqa: BLE001 - preserve per-file audit diagnostics
            corrupt.append({"path": rel_path, "error": str(error)})

    duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    cross_split_groups = [group for group in duplicate_groups if len({item[1] for item in group}) > 1]

    prompt_path = dataset_root / "plantwild_prompts.json"
    prompts = json.loads(prompt_path.read_text(encoding="utf-8")) if prompt_path.is_file() else {}
    prompt_rows = [str(text).strip() for values in prompts.values() if isinstance(values, list) for text in values]
    prompt_classes_matching_labels = sorted(set(prompts).intersection(classes.values()))
    prompt_count = len(prompt_rows)
    paired_csv = Path(paired_csv).resolve() if paired_csv else None
    paired_csv_exists = bool(paired_csv and paired_csv.is_file())

    result = {
        "decision": "BLOCKED",
        "blocker": None if paired_csv_exists else "No image-specific, independently sourced text-pair CSV is present.",
        "dataset_root": str(dataset_root),
        "official_split_counts": split_counts,
        "official_class_count": len(classes),
        "class_image_count_min": min(class_counts.values(), default=0),
        "class_image_count_max": max(class_counts.values(), default=0),
        "class_counts_by_split": {
            split: {str(class_id): count for class_id, count in sorted(counts.items())}
            for split, counts in split_class_counts.items()
        },
        "path_rows": len(rows),
        "unique_official_paths": len(path_owners),
        "missing_images": missing,
        "corrupt_images": corrupt,
        "hashed_images": sum(len(group) for group in hashes.values()),
        "unique_image_sha256": len(hashes),
        "exact_duplicate_image_groups": len(duplicate_groups),
        "extra_duplicate_image_rows": sum(len(group) - 1 for group in duplicate_groups),
        "exact_duplicate_groups_crossing_splits": len(cross_split_groups),
        "cross_split_duplicate_examples": [group for group in cross_split_groups[:20]],
        "local_text_assets": {
            "plantwild_prompts_file_present": prompt_path.is_file(),
            "prompt_class_keys": len(prompts),
            "prompt_classes_matching_official_labels": len(prompt_classes_matching_labels),
            "prompt_rows": prompt_count,
            "unique_exact_prompts_casefolded": len({text.casefold() for text in prompt_rows}),
            "duplicate_exact_prompt_rows": prompt_count - len({text.casefold() for text in prompt_rows}),
            "association": "50 class-conditioned GPT-generated prompts per class; not image-specific observations",
            "usable_as_paired_training_text": False,
        },
        "url_record": {
            "present": (dataset_root / "url_record.json").is_file(),
            "record_count": len(json.loads((dataset_root / "url_record.json").read_text(encoding="utf-8")))
            if (dataset_root / "url_record.json").is_file() else 0,
            "role": "image source URL provenance only; not text descriptions",
        },
        "paired_csv": str(paired_csv) if paired_csv else None,
        "paired_csv_exists": paired_csv_exists,
        "paired_csv_validated": False,
        "notes": [
            "A CSV's existence alone does not establish text provenance or image-text correspondence.",
            "When supplied, validate with ml.fusion.data.load_paired_text_csv; labels and splits must be joined from official split files.",
            "Exact class-name rejection is a partial automated leakage check; provenance still requires curator verification.",
        ],
    }
    if paired_csv_exists:
        from ml.fusion.data import (
            ALLOWED_TEXT_ORIGINS,
            HUMAN_TEXT_ORIGINS,
            load_paired_text_csv,
        )
        from ml.fusion.eligibility import eligible_sample_ids, load_eligibility_manifest
        from ml.models.class_labels import load_class_labels

        labels = load_class_labels(dataset_root / "classes.txt")
        split_paths = {split: Path(filename) for split, filename in split_files.items()}
        gates: dict[str, dict] = {}
        result["gates"] = gates

        try:
            pairs = load_paired_text_csv(paired_csv, dataset_root, split_paths, labels)
        except Exception as error:  # noqa: BLE001 - report the failure instead of crashing
            detail = f"{error.__class__.__name__}: {error}"
            result["paired_csv_validated"] = False
            result["paired_csv_error"] = detail
            gates["paired_csv_loaded"] = {"passed": False, "detail": detail}
            result["decision"] = "BLOCKED"
            result["blocker"] = "The paired CSV failed structural validation; no experiment may run."
            return result

        result["paired_csv_validated"] = True
        rows_by_split = Counter(pair.split for pair in pairs)
        result["paired_rows_by_split"] = dict(rows_by_split)
        texts = [" ".join(pair.text.casefold().split()) for pair in pairs]
        result["paired_text_rows"] = len(texts)
        result["unique_exact_paired_texts"] = len(set(texts))
        result["duplicate_exact_paired_text_rows"] = len(texts) - len(set(texts))
        origins = Counter(pair.text_origin for pair in pairs)
        result["paired_text_origins"] = dict(origins)

        sample_ids = [pair.sample_id for pair in pairs]
        splits_by_sample: dict[str, set[str]] = defaultdict(set)
        for pair in pairs:
            splits_by_sample[pair.sample_id].add(pair.split)

        manifest_path = ROOT / "outputs" / "research" / "fusion_eligibility_manifest.json"
        eligible = None
        if manifest_path.is_file():
            eligible = eligible_sample_ids(
                load_eligibility_manifest(manifest_path), dataset_root, split_paths, labels
            )
        off_manifest = sorted(set(sample_ids) - eligible) if eligible is not None else []

        empty_splits = sorted(
            {"train", "validation", "test"} - {name for name, count in rows_by_split.items() if count}
        )
        non_human = sorted(set(origins) - HUMAN_TEXT_ORIGINS)
        ledger_path = ROOT / "data" / "annotation" / "review_ledger.csv"

        gates["paired_csv_loaded"] = {"passed": True, "detail": f"{len(pairs)} row(s) validated"}
        gates["text_origins_registered"] = {
            "passed": set(origins) <= ALLOWED_TEXT_ORIGINS,
            "detail": f"origins={sorted(origins)}",
        }
        gates["no_duplicate_sample_ids"] = {
            "passed": len(set(sample_ids)) == len(sample_ids),
            "detail": f"{len(sample_ids) - len(set(sample_ids))} duplicate sample_id row(s)",
        }
        gates["single_split_per_sample"] = {
            "passed": all(len(splits) == 1 for splits in splits_by_sample.values()),
            "detail": "every sample_id maps to exactly one official split",
        }
        gates["eligibility_aligned"] = {
            "passed": eligible is not None and not off_manifest,
            "detail": (
                "not evaluated (no eligibility manifest present)"
                if eligible is None
                else f"{len(off_manifest)} row(s) outside the duplicate-free eligible manifest"
            ),
        }
        gates["all_splits_covered"] = {
            "passed": not empty_splits,
            "detail": f"rows_by_split={dict(rows_by_split)}; empty={empty_splits}",
        }
        gates["human_origin_only"] = {
            "passed": not non_human,
            "detail": (
                f"non-human origins present: {non_human}; these are not independent evidence"
                if non_human
                else "all text originates from human observations"
            ),
        }
        gates["independent_review_ledger"] = {
            "passed": ledger_path.is_file(),
            "detail": (
                f"review ledger present at {ledger_path}"
                if ledger_path.is_file()
                else "no independent review ledger exists"
            ),
        }

        structural = (
            "paired_csv_loaded", "text_origins_registered", "no_duplicate_sample_ids",
            "single_split_per_sample", "eligibility_aligned",
        )
        failed_structural = [name for name in structural if not gates[name]["passed"]]
        if failed_structural:
            result["decision"] = "BLOCKED"
            result["blocker"] = f"Structural gate(s) failed: {failed_structural}"
        elif not gates["all_splits_covered"]["passed"]:
            result["decision"] = "REVIEW_REQUIRED"
            result["blocker"] = (
                f"Split coverage is incomplete (empty split(s): {empty_splits}); a fair "
                "train/validation/test comparison is impossible until every split has text."
            )
        elif not gates["human_origin_only"]["passed"]:
            result["decision"] = "REVIEW_REQUIRED"
            result["blocker"] = (
                "Only non-human (model-generated) candidate text is present; it is not independent "
                "evidence and cannot support a deployable multimodal claim."
            )
        elif not gates["independent_review_ledger"]["passed"]:
            result["decision"] = "REVIEW_REQUIRED"
            result["blocker"] = "Human observations exist but no independent review ledger is present."
        else:
            result["decision"] = "GO"
            result["blocker"] = None
        return result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PlantWild splits and paired-text readiness without training.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--paired-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.dataset_root, args.paired_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "decision", "blocker", "official_split_counts", "official_class_count", "hashed_images",
        "exact_duplicate_image_groups", "exact_duplicate_groups_crossing_splits", "local_text_assets",
        "paired_csv_exists", "paired_csv_validated", "paired_rows_by_split", "paired_text_origins",
        "gates",
    ) if key in report}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
