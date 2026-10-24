"""Emit the model-generated-description experiment CSV, gated on automated quality checks.

    python -m scripts.emit_model_observations                  :: validate and emit
    python -m scripts.emit_model_observations --dry-run        :: report only, write nothing

The output is a separate experimental artifact (default
``data/paired_model_observations.csv``) with ``text_origin = model_generated_observation``.
It is never written to ``data/paired_observations.csv`` (the human-observation track), and
no record is rewritten, repaired, or fabricated: records that fail a check are excluded.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from ml.annotation.generation import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    MODEL_TEXT_ORIGIN,
)
from ml.annotation.quality import audit_generation
from ml.fusion.annotation import write_observations_csv
from ml.fusion.data import load_paired_text_csv
from ml.fusion.eligibility import eligible_sample_ids, filter_paired_examples, load_eligibility_manifest
from ml.models.class_labels import load_class_labels

ROOT = Path(__file__).resolve().parents[1]
HUMAN_TRACK = (ROOT / "data" / "paired_observations.csv").resolve()
DEFAULT_MODEL_OUTPUT = ROOT / "data" / "paired_model_observations.csv"
DEFAULT_REPORT = ROOT / "outputs" / "research" / "model_observation_emission.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--eligibility-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-split", type=Path, default=Path("train_split.txt"))
    parser.add_argument("--validation-split", type=Path, default=Path("val_split.txt"))
    parser.add_argument("--test-split", type=Path, default=Path("test_split.txt"))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing the CSV")
    parser.add_argument("--skip-image-verification", action="store_true",
                        help="Skip re-hashing images (faster; weaker identity check)")
    args = parser.parse_args()

    target = Path(args.output).resolve()
    if target == HUMAN_TRACK:
        raise SystemExit(
            "refusing to write model-generated text into the human-observation track; "
            "generated descriptions belong in data/paired_model_observations.csv"
        )

    labels = load_class_labels(args.dataset_root / "classes.txt")
    splits = {"train": args.train_split, "validation": args.validation_split, "test": args.test_split}

    report, ok_rows, verdicts = audit_generation(
        args.store, dataset_root=args.dataset_root, manifest_path=args.eligibility_manifest,
        split_files=splits, labels=labels, verify_images=not args.skip_image_verification,
    )
    report["excluded_records"] = [
        {"sample_id": verdict.sample_id, "reasons": verdict.reasons}
        for verdict in verdicts if not verdict.ok
    ][:50]
    report["emitted"] = False
    report["output"] = str(target)

    if not report["emittable"]:
        report["blocker"] = (
            "automated quality checks failed (a blocking check has a non-zero count) or no clean "
            "record exists; nothing was emitted"
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("decision", "blocker", "clean_records", "checks")}, indent=2))
        raise SystemExit(f"emission blocked: {report['blocker']}")

    models = sorted({row.get("model", "") for row in ok_rows})
    prompts = sorted({row.get("prompt_version", "") for row in ok_rows})
    if len(models) != 1 or len(prompts) != 1:
        report["blocker"] = (
            f"refusing to emit a mixed-source dataset: models={models} prompt_versions={prompts}; "
            "one experimental track must come from one model and one prompt version"
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit(f"emission blocked: {report['blocker']}")
    report["model"] = models[0]
    report["prompt_version"] = prompts[0]

    collated = [{
        "sample_id": row["sample_id"], "image_path": row["sample_id"], "text": row["description"],
        "text_origin": MODEL_TEXT_ORIGIN, "provenance": row["provenance"],
        "source_reference": row["source_reference"],
    } for row in ok_rows]

    if args.dry_run:
        report["decision"] = "DRY_RUN_OK"
        report["would_emit_rows"] = len(collated)
        report["blocker"] = None
    else:
        manifest = load_eligibility_manifest(args.eligibility_manifest)
        eligible = eligible_sample_ids(manifest, args.dataset_root, splits, labels)
        staged = target.with_name(f"{target.name}.staged")
        try:
            write_observations_csv(collated, staged)
            pairs = filter_paired_examples(
                load_paired_text_csv(staged, args.dataset_root, splits, labels), eligible
            )
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        os.replace(staged, target)
        report["emitted"] = True
        report["validated_rows"] = len(pairs)
        report["validated_rows_by_split"] = dict(Counter(pair.split for pair in pairs))
        report["text_origins"] = sorted({pair.text_origin for pair in pairs})
        report["unique_texts"] = len({pair.text.casefold() for pair in pairs})
        report["decision"] = "EMITTED"
        report["blocker"] = (
            "Model-generated candidate descriptions only. They are not human observations, "
            "not independently reviewed, and cannot support a deployable multimodal claim."
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "decision": report["decision"], "clean_records": report["clean_records"],
        "emitted": report["emitted"], "output": report["output"],
        "validated_rows_by_split": report.get("validated_rows_by_split"),
    }, indent=2))
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
