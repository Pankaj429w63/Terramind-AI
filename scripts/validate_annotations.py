"""Validate blinded annotations and, only when approved, emit the paired-text CSV.

Workflow:
1. Load the sealed alias map and the (partially) filled annotator workbook.
2. Report per-row problems (missing observer, bad timestamp, unknown alias, label-derived
   text) without repeating observation text, class labels, or image paths.
3. Require an approved review-ledger entry for every observation and confirm the
   elicitation instrument version is recorded in provenance.
4. When ``--emit-observations`` is given, stage the CSV, re-load it through the real
   training loader on the real dataset, and only then atomically replace the destination.
   A failure never leaves a partial or misleading training CSV behind.

No text is ever generated; only annotator-supplied observations are passed through.
Decisions: ``INCOMPLETE`` (missing observations), ``BLOCKED`` (invalid rows/review),
``READY`` (complete and valid, emission not requested), ``EMITTED`` (validated output written).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from ml.fusion.annotation import (
    REVIEW_STATUSES,
    collate_observations,
    load_alias_map,
    load_annotator_workbook,
    load_review_ledger,
    observation_issues,
    require_independent_review,
    require_instrument_version,
    require_reviewed,
    write_observations_csv,
)
from ml.fusion.data import load_paired_text_csv
from ml.fusion.eligibility import eligible_sample_ids, filter_paired_examples, load_eligibility_manifest
from ml.models.class_labels import load_class_labels

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "plantwild" / "plantwild"
DEFAULT_MANIFEST = ROOT / "outputs" / "research" / "fusion_eligibility_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "research" / "annotation_validation.json"
REPORT_LIMIT = 20


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate blinded annotations before any training.")
    parser.add_argument("--workbook", type=Path, required=True, help="Filled annotator workbook CSV")
    parser.add_argument("--alias-map", type=Path, required=True, help="Sealed alias map from build_annotation_workbook.py")
    parser.add_argument("--review-ledger", type=Path, default=None,
                        help="Review ledger CSV; required to emit observations")
    parser.add_argument("--eligibility-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--train-split", type=Path, default=Path("train_split.txt"))
    parser.add_argument("--validation-split", type=Path, default=Path("val_split.txt"))
    parser.add_argument("--test-split", type=Path, default=Path("test_split.txt"))
    parser.add_argument("--text-origin", default="user_observation")
    parser.add_argument("--instrument-version", default=None,
                        help="Elicitation instrument version; required to emit and must appear in --provenance")
    parser.add_argument("--provenance", default=None, help="Collection/annotation provenance recorded in the CSV")
    parser.add_argument("--source-reference", default=None, help="Citation/URL or stable URI for the source records")
    parser.add_argument("--emit-observations", type=Path, default=None,
                        help="Write the paired-text CSV here (requires --review-ledger, --provenance, --source-reference)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    labels = load_class_labels(args.dataset_root / "classes.txt")
    manifest = load_eligibility_manifest(args.eligibility_manifest)
    split_files = {"train": args.train_split, "validation": args.validation_split, "test": args.test_split}
    eligible = eligible_sample_ids(manifest, args.dataset_root, split_files, labels)

    alias_map, _salt = load_alias_map(args.alias_map)
    workbook_rows = load_annotator_workbook(args.workbook)
    filled = sum(1 for row in workbook_rows if (row.get("observation") or "").strip())
    issues = observation_issues(workbook_rows, alias_map, labels)

    report: dict[str, object] = {
        "workbook_rows": len(workbook_rows),
        "aliases_defined": len(alias_map),
        "eligible_samples": len(eligible),
        "filled_rows": filled,
        "missing_rows": len(alias_map) - filled,
        "rejected_count": len(issues),
        "rejected_rows": issues[:REPORT_LIMIT],
        "emitted": False,
    }

    ledger = load_review_ledger(args.review_ledger) if args.review_ledger else None
    if ledger is not None:
        counts = Counter(entry.get("review_status", "").casefold() for entry in ledger.values())
        review_counts: dict[str, int] = {"ledger_entries": len(ledger)}
        review_counts.update({status: counts.get(status, 0) for status in REVIEW_STATUSES})
        report["review"] = review_counts

    if filled < len(alias_map):
        report["decision"] = "INCOMPLETE"
        report["blocker"] = (
            f"{len(alias_map) - filled} of {len(alias_map)} eligible images still have no observation"
        )
    elif issues:
        report["decision"] = "BLOCKED"
        report["blocker"] = f"{len(issues)} filled observation(s) were rejected; fix the listed aliases and re-run"
    elif not args.emit_observations:
        report["decision"] = "READY"
        report["blocker"] = (
            "All observations are present and valid; re-run with --emit-observations, --review-ledger, "
            "--provenance, --source-reference, and --instrument-version to emit"
        )
    else:
        if ledger is None or not args.provenance or not args.source_reference or not args.instrument_version:
            parser.error(
                "--emit-observations requires --review-ledger, --provenance, --source-reference, and --instrument-version"
            )
        report["instrument_version"] = require_instrument_version(args.provenance, args.instrument_version)
        collated = collate_observations(
            workbook_rows, alias_map, labels,
            text_origin=args.text_origin, provenance=args.provenance, source_reference=args.source_reference,
        )
        report["collated_rows"] = len(collated)
        out_of_scope = sorted({row["sample_id"] for row in collated} - eligible)
        if out_of_scope:
            raise ValueError(
                f"Observations reference {len(out_of_scope)} non-eligible sample(s); "
                "eligible samples come from the duplicate-free manifest"
            )
        report["review_gate"] = require_reviewed(collated, ledger)
        report["independent_review_gate"] = require_independent_review(workbook_rows, alias_map, ledger)

        target = Path(args.emit_observations)
        staged = target.with_name(f"{target.name}.staged")
        try:
            write_observations_csv(collated, staged)
            pairs = filter_paired_examples(
                load_paired_text_csv(staged, args.dataset_root, split_files, labels), eligible
            )
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        os.replace(staged, target)
        report["emitted"] = True
        report["emitted_path"] = str(target)
        report["validated_rows_by_split"] = dict(Counter(pair.split for pair in pairs))
        report["decision"] = "EMITTED"
        report["blocker"] = (
            "Validated output written; human review of provenance adequacy and inference "
            "availability is still required before training."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
