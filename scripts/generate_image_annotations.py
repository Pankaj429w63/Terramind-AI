"""Generate candidate image descriptions for every eligible PlantWild image.

The pipeline is resumable, label-free, and never touches human annotations:

    python -m scripts.generate_image_annotations --check
    python -m scripts.generate_image_annotations --limit 10
    python -m scripts.generate_image_annotations --resume
    python -m scripts.generate_image_annotations --all

Generated text is recorded as ``model_generated_observation`` in a store that is
separate from ``outputs/annotation/annotator_workbook.csv``; nothing here can overwrite a
genuine human observation. If the requested backend is unavailable, the exact missing
dependency is reported and no description is fabricated.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ml.annotation.generation import (
    DONE_STATUSES,
    FAILED,
    PROMPTS,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    DEFAULT_PREFLIGHT,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_WORKBOOK,
    GENERATION_STATUSES,
    UNREADABLE,
    GenerationError,
    GenerationRecord,
    GenerationStore,
    assess,
    build_backend,
    iter_eligible_targets,
    preflight,
    provenance_for,
    sanitize,
    sha256_file,
    source_reference_for,
)
from ml.models.class_labels import load_class_labels

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_REMOTE_MODEL = "Salesforce/blip-image-captioning-large"

logger = logging.getLogger("generate_image_annotations")


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def _splits(args: argparse.Namespace) -> dict[str, Path]:
    return {"train": args.train_split, "validation": args.validation_split, "test": args.test_split}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--eligibility-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-split", type=Path, default=Path("train_split.txt"))
    parser.add_argument("--validation-split", type=Path, default=Path("val_split.txt"))
    parser.add_argument("--test-split", type=Path, default=Path("test_split.txt"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK,
                        help="Human annotator workbook; only used to prove the two files are distinct")
    parser.add_argument("--backend", choices=("local", "hf-inference", "openai-compatible"), default="local")
    parser.add_argument("--model", default=None, help="Model id; defaults depend on the backend")
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION, choices=sorted(PROMPTS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N new images (pilot)")
    parser.add_argument("--all", action="store_true", help="Process every image that still needs a description")
    parser.add_argument("--resume", action="store_true", help="Continue from the existing store")
    parser.add_argument("--check", action="store_true", help="Run preflight and exit without generating")
    parser.add_argument("--retry-failed", action="store_true", help="Retry rows currently marked failed/unreadable")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--price-per-1k-images", type=float, default=None,
                        help="Explicit price for remote backends; required before any paid request")
    parser.add_argument("--confirm-paid-usage", action="store_true",
                        help="Acknowledge that a remote backend may incur charges")
    args = parser.parse_args(argv)
    if args.model is None:
        args.model = DEFAULT_LOCAL_MODEL if args.backend == "local" else DEFAULT_REMOTE_MODEL
    return args


def _estimate_remote_cost(args: argparse.Namespace, pending: int) -> None:
    if args.price_per_1k_images is None:
        raise SystemExit(
            "refusing to use a remote backend without a cost estimate: pass "
            "--price-per-1k-images <price per 1000 images> (and --confirm-paid-usage) so the "
            f"charge for {pending} image(s) is explicit before any request is sent."
        )
    estimate = args.price_per_1k_images * pending / 1000.0
    print(f"REMOTE COST ESTIMATE: {pending} images x {args.price_per_1k_images} per 1000 = "
          f"{estimate:.4f} (billing units as priced by the operator)")
    if not args.confirm_paid_usage:
        raise SystemExit("refusing to send paid requests: pass --confirm-paid-usage to proceed.")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    labels = load_class_labels(args.dataset_root / "classes.txt")
    splits = _splits(args)
    prompt = PROMPTS[args.prompt_version]

    report = preflight(
        dataset_root=args.dataset_root, manifest_path=args.eligibility_manifest, split_files=splits,
        output_path=args.output, workbook_path=args.workbook, backend_kind=args.backend,
        model_id=args.model, prompt_version=args.prompt_version,
    )
    DEFAULT_PREFLIGHT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_PREFLIGHT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.check:
        decision = "READY" if report["ready"] else "BLOCKED"
        print(f"PREFLIGHT DECISION: {decision}")
        if not report["ready"]:
            raise SystemExit(f"generation blocked: {'; '.join(report['missing_dependencies'])}")
        return

    if not any((args.all, args.resume, args.limit is not None)):
        raise SystemExit(
            "choose an action: --check (preflight), --limit N (pilot), --all, or --resume"
        )
    if report["missing_dependencies"]:
        raise SystemExit(
            "generation blocked by missing dependencies: " + "; ".join(report["missing_dependencies"])
        )

    log_path = args.output.parent / "generation_log.jsonl"
    _configure_logging(log_path)

    store = GenerationStore(args.output)
    store.load()
    if store.discarded_lines:
        logger.warning("dropped %d unreadable line(s) while resuming", store.discarded_lines)

    completed = store.completed()
    attempted = store.attempted()
    targets = []
    for sample_id, image_path, digest, split in iter_eligible_targets(
        args.dataset_root, args.eligibility_manifest, splits, labels
    ):
        if sample_id in completed:
            continue
        if sample_id in attempted and not args.retry_failed:
            continue
        targets.append((sample_id, image_path, digest, split))
    if args.limit is not None:
        targets = targets[: max(0, args.limit)]
    if not targets:
        print("Nothing to do: every eligible image already has a description in the store.")
        return

    if args.backend != "local":
        _estimate_remote_cost(args, len(targets))

    logger.info("loading backend=%s model=%s prompt=%s for %d image(s)",
                args.backend, args.model, prompt.version, len(targets))
    started = time.perf_counter()
    backend = build_backend(
        args.backend, args.model, prompt=prompt, device=args.device, max_new_tokens=args.max_new_tokens
    )
    provenance = provenance_for(backend, prompt, args.model)
    source_reference = source_reference_for(backend, args.model)
    counts: dict[str, int] = {status: 0 for status in GENERATION_STATUSES}
    durations: list[float] = []
    try:
        for index, (sample_id, image_path, expected_digest, _split) in enumerate(targets, 1):
            record = GenerationRecord(
                sample_id=sample_id, status=FAILED, model=args.model,
                model_version=backend.model_version, prompt_version=prompt.version,
                prompt_sha256=prompt.sha256(),
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                backend=backend.name, provenance=provenance, source_reference=source_reference,
            )
            image_started = time.perf_counter()
            if not image_path.is_file():
                record.status, record.notes = UNREADABLE, ["image file is missing"]
            else:
                digest = sha256_file(image_path)
                record.image_sha256 = digest
                if expected_digest and digest != expected_digest:
                    record.status = UNREADABLE
                    record.notes = "image bytes do not match the eligibility manifest hash"
                else:
                    last_error = ""
                    for attempt in range(1, args.max_retries + 2):
                        try:
                            description = backend.describe(image_path, prompt)
                            record.description = sanitize(description.text)
                            record.mean_token_logprob = (
                                "" if description.mean_token_logprob is None
                                else str(description.mean_token_logprob)
                            )
                            record.status, notes = assess(record.description, description.mean_token_logprob)
                            record.notes = "; ".join(notes)
                            break
                        except GenerationError as error:
                            last_error = str(error)
                        except Exception as error:  # noqa: BLE001 - one bad image must not stop the batch
                            last_error = f"{type(error).__name__}: {error}"
                        if attempt <= args.max_retries:
                            time.sleep(min(2.0 * attempt, 5.0))
                    else:
                        record.status, record.notes = FAILED, last_error[:300]
            store.append(record)
            counts[record.status] = counts.get(record.status, 0) + 1
            durations.append(time.perf_counter() - image_started)
            store.maybe_checkpoint(args.checkpoint_every)
            if args.progress_every and index % args.progress_every == 0:
                mean = sum(durations) / len(durations)
                remaining = len(targets) - index
                logger.info(json.dumps({
                    "processed": index, "of": len(targets), "seconds_per_image": round(mean, 3),
                    "eta_minutes": round(mean * remaining / 60.0, 1), "counts": counts,
                }))
    finally:
        store.checkpoint()
        backend.close()

    elapsed = time.perf_counter() - started
    summary = {
        "processed": len(targets), "counts": counts,
        "seconds": round(elapsed, 1),
        "seconds_per_image": round(elapsed / max(1, len(targets)), 3),
        "store": str(args.output), "store_rows_total": len(store.records),
        "completed_total": len(store.completed()), "log": str(log_path),
    }
    print(json.dumps(summary, indent=2))
    logger.info("run summary %s", json.dumps(summary))


if __name__ == "__main__":
    main()
