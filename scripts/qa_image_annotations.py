"""Run automated quality checks over model-generated image descriptions.

    python -m scripts.qa_image_annotations
    python -m scripts.qa_image_annotations --skip-image-verification

Writes ``outputs/research/model_annotation_quality.json`` with actual counts. Automated
checks are structural and lexical: they do not prove that a description matches its image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.annotation.generation import DEFAULT_DATASET_ROOT, DEFAULT_MANIFEST, DEFAULT_OUTPUT
from ml.annotation.quality import audit_generation
from ml.models.class_labels import load_class_labels

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "outputs" / "research" / "model_annotation_quality.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--eligibility-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-split", type=Path, default=Path("train_split.txt"))
    parser.add_argument("--validation-split", type=Path, default=Path("val_split.txt"))
    parser.add_argument("--test-split", type=Path, default=Path("test_split.txt"))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-image-verification", action="store_true")
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    labels = load_class_labels(args.dataset_root / "classes.txt")
    report, ok_rows, verdicts = audit_generation(
        args.store, dataset_root=args.dataset_root, manifest_path=args.eligibility_manifest,
        split_files={"train": args.train_split, "validation": args.validation_split, "test": args.test_split},
        labels=labels, verify_images=not args.skip_image_verification,
    )
    report["excluded_records"] = [
        {"sample_id": verdict.sample_id, "reasons": verdict.reasons}
        for verdict in verdicts if not verdict.ok
    ][: args.examples]
    report["clean_sample_examples"] = [row["sample_id"] for row in ok_rows[: args.examples]]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        key: report[key] for key in (
            "records", "status_counts", "eligible_samples", "missing_generations",
            "successful_generated", "flagged_uncertain", "failed", "unreadable_or_changed_images",
            "duplicate_description_groups", "duplicate_description_rows", "clean_records",
            "records_requiring_human_verification", "decision",
        )
    }, indent=2))
    for name, check in report["checks"].items():
        print(f"  {name}: count={check['count']} blocking={check['blocking']}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
