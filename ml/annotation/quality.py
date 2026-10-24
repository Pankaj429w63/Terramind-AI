"""Automated quality checks for model-generated image descriptions.

These checks are structural and lexical. They can prove that a record is malformed,
duplicated, label-leaking, missing provenance, or attached to bytes that no longer match
the eligibility manifest. They **cannot** prove that a description semantically matches
the image, so every report states that limitation instead of implying semantic review.
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ml.fusion.data import validate_provenance, validate_text_not_label_derived
from ml.fusion.eligibility import load_eligibility_manifest, validate_eligibility_manifest
from ml.annotation.generation import (
    DONE_STATUSES,
    FAILED,
    FLAGGED,
    GENERATED,
    GENERATION_STATUSES,
    RECORD_COLUMNS,
    PROMPTS,
    UNREADABLE,
    sha256_file,
)

# Words that assert a cause, disease, treatment, or severity rather than visible evidence.
DISEASE_CLAIM_TERMS = (
    "fungus", "fungal", "bacteri", "viral", "virus", "pathogen", "infect", "disease",
    "blight", "rust", "mildew", "scab", "canker", "mosaic", "wilt disease", "necrosis caused",
    "diagnos", "treatment", "fungicide", "pesticide", "herbicide", "spray", "cure",
    "severity", "early stage", "late stage", "deficien", "nutrient", "overwater",
    "underwater", "infestation", "pest",
)
# Tokens that echo metadata rather than describing pixels.
METADATA_TOKENS = (".jpg", ".jpeg", ".png", "images/", "sample_id", "split", "http://", "https://")
GENERIC_ONLY = re.compile(r"^\s*(an? )?(close[- ]?up )?(photo|photograph|image|picture)( of)?( (a|the))? ?(plant|leaf|leaves|plant leaf)?\.?\s*$", re.IGNORECASE)

# How the two kinds of check affect emission:
# * ``STORE_INTEGRITY_CHECKS`` are properties of the *store*, not of a single sentence
#   (a duplicated record, a missing credential field, image bytes that no longer match the
#   manifest). If one fires, the store cannot be trusted and nothing is emitted.
# * Every other check is a property of individual records. A hit excludes those records
#   from emission (they are never rewritten, repaired, or replaced) while the remaining
#   clean records may still be emitted, with the exclusions reported in full.
STORE_INTEGRITY_CHECKS = frozenset({
    "duplicate_records", "image_identity_or_corruption",
    "missing_or_invalid_provenance", "invalid_model_metadata",
})


@dataclass
class Verdict:
    sample_id: str
    ok: bool
    reasons: list[str]


def _normalized(text: str) -> str:
    return " ".join((text or "").casefold().split())


def read_records(path: Path) -> list[dict[str, str]]:
    """Read one generation store, tolerating a torn trailing line."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Generation store not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != list(RECORD_COLUMNS):
            raise ValueError(f"Generation store {path} has an unexpected header")
        return [{key: (value or "").strip() for key, value in row.items() if key} for row in reader]


def audit_generation(
    store_path: Path,
    *,
    dataset_root: Path,
    manifest_path: Path,
    split_files: Mapping[str, Path],
    labels: Mapping[int, str],
    verify_images: bool = True,
) -> tuple[dict, list[dict[str, str]], list[Verdict]]:
    """Audit a generation store; return (report, ok_rows, verdicts)."""
    records = read_records(store_path)
    manifest = load_eligibility_manifest(manifest_path)
    validate_eligibility_manifest(manifest, dataset_root, split_files, labels)
    eligible = {str(entry["sample_id"]): entry for entry in manifest.get("eligible", [])}
    images_root = (dataset_root / "images").resolve()

    duplicates: Counter[str] = Counter()
    duplicate_examples: list[dict[str, str]] = []
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        by_sample[row.get("sample_id", "")].append(row)
        duplicates[_normalized(row.get("description", ""))] += 1

    text_groups: dict[str, list[str]] = defaultdict(list)
    for row in records:
        if row.get("description"):
            text_groups[_normalized(row["description"])].append(row.get("sample_id", ""))

    checks: dict[str, dict] = {}

    def record_check(name: str, hits: list[tuple[str, str]], note: str) -> None:
        checks[name] = {
            "count": len(hits), "blocking": name in STORE_INTEGRITY_CHECKS, "note": note,
            "examples": [{"sample_id": sid, "detail": detail} for sid, detail in hits[:10]],
        }

    verdicts: list[Verdict] = []
    empty_hits, label_hits, claim_hits, meta_hits, provenance_hits, model_hits, image_hits = [], [], [], [], [], [], []
    duplicate_record_hits: list[tuple[str, str]] = []
    duplicate_text_rows = 0

    for sample_id, rows in by_sample.items():
        if len(rows) > 1:
            duplicate_record_hits.append((sample_id, f"{len(rows)} records for the same sample_id"))
        row = rows[-1]
        reasons: list[str] = []
        status = row.get("status", "")
        description = row.get("description", "")

        if status in DONE_STATUSES:
            if not description:
                empty_hits.append((sample_id, "status says generated but the description is empty"))
                reasons.append("empty description")
            else:
                if len(description.split()) < 3:
                    empty_hits.append((sample_id, f"suspiciously short description ({len(description.split())} words)"))
                    reasons.append("malformed description")
                if GENERIC_ONLY.match(description):
                    empty_hits.append((sample_id, "description is generic boilerplate only"))
                    reasons.append("generic boilerplate")
                try:
                    validate_text_not_label_derived(description, labels)
                except ValueError as error:
                    label_hits.append((sample_id, str(error)))
                    reasons.append("class-label leakage")
                lowered = description.casefold()
                claimed = [term for term in DISEASE_CLAIM_TERMS if term in lowered]
                if claimed:
                    claim_hits.append((sample_id, f"disease/cause/treatment claim terms: {sorted(set(claimed))[:4]}"))
                    reasons.append("unsupported disease claim")
                echoed = [token for token in METADATA_TOKENS if token in lowered]
                if echoed or sample_id and sample_id.casefold() in lowered:
                    meta_hits.append((sample_id, f"description repeats metadata: {echoed or [sample_id]}"))
                    reasons.append("metadata echo")
            try:
                validate_provenance(row.get("provenance", ""), row.get("source_reference", ""))
            except ValueError as error:
                provenance_hits.append((sample_id, str(error)))
                reasons.append("invalid provenance")

            prompt_version = row.get("prompt_version", "")
            if prompt_version not in PROMPTS:
                model_hits.append((sample_id, f"unknown prompt_version {prompt_version!r}"))
                reasons.append("invalid prompt metadata")
            elif row.get("prompt_sha256") != PROMPTS[prompt_version].sha256():
                model_hits.append((sample_id, "prompt_sha256 does not match the registered prompt"))
                reasons.append("invalid prompt metadata")
            if not (row.get("model") and row.get("model_version") and row.get("backend")):
                model_hits.append((sample_id, "missing model/model_version/backend"))
                reasons.append("invalid model metadata")

            entry = eligible.get(sample_id)
            if entry is None:
                image_hits.append((sample_id, "record is not in the eligible manifest"))
                reasons.append("not eligible")
            elif verify_images:
                image_path = (images_root / sample_id).resolve()
                if not image_path.is_relative_to(images_root) or not image_path.is_file():
                    image_hits.append((sample_id, "image file is missing or unsafe"))
                    reasons.append("unreadable image")
                else:
                    try:
                        digest = sha256_file(image_path)
                    except OSError as error:
                        image_hits.append((sample_id, error.__class__.__name__))
                        reasons.append("unreadable image")
                    else:
                        if row.get("image_sha256") and row["image_sha256"] != digest:
                            image_hits.append((sample_id, "image bytes changed after generation"))
                            reasons.append("image changed")
                        if entry.get("sha256") and entry["sha256"] != digest:
                            image_hits.append((sample_id, "image bytes differ from the eligibility manifest"))
                            reasons.append("image mismatch")
        verdicts.append(Verdict(sample_id, ok=not reasons, reasons=reasons))

    for text, sample_ids in text_groups.items():
        if text and len(sample_ids) > 1:
            duplicate_text_rows += len(sample_ids)
            if len(duplicate_examples) < 10:
                duplicate_examples.append({"text": text[:120], "sample_ids": sample_ids[:5], "size": len(sample_ids)})

    status_counts = Counter(row.get("status", "") for row in records)
    flagged = sum(1 for row in records if row.get("status") == FLAGGED)
    ok_rows = [
        by_sample[sid][-1] for sid, verdict in
        ((v.sample_id, v) for v in verdicts) if verdict.ok and by_sample[sid][-1].get("status") in DONE_STATUSES
    ]
    ok_ids = {row["sample_id"] for row in ok_rows}
    missing = sorted(set(eligible) - {row.get("sample_id", "") for row in records})

    record_check("empty_or_malformed", empty_hits, "empty, too short, or boilerplate-only text")
    record_check("class_label_leakage", label_hits, "text contains a PlantWild class name")
    record_check("unsupported_disease_claim", claim_hits, "text asserts a cause/disease/treatment beyond visible evidence")
    record_check("metadata_echo", meta_hits, "text repeats paths, ids, or URLs instead of visible evidence")
    record_check("missing_or_invalid_provenance", provenance_hits, "provenance/source_reference failed validation")
    record_check("invalid_model_metadata", model_hits, "model, version, backend, or prompt metadata is missing/inconsistent")
    record_check("image_identity_or_corruption", image_hits, "image missing, changed, or not matching the manifest hash")
    record_check("duplicate_records", duplicate_record_hits, "more than one record for the same sample_id")
    record_check(
        "duplicate_descriptions", [(ex["sample_ids"][0], f"text shared by {ex['size']} images")
                                   for ex in duplicate_examples],
        "identical text across different images (not image-specific)",
    )
    record_check("missing_generations", [(sid, "no generation record") for sid in missing[:10]],
                 f"{len(missing)} eligible image(s) have no record")

    integrity_blocked = any(checks[name]["count"] for name in STORE_INTEGRITY_CHECKS)
    report = {
        "store": str(store_path),
        "records": len(records),
        "status_counts": {status: status_counts.get(status, 0) for status in GENERATION_STATUSES},
        "eligible_samples": len(eligible),
        "missing_generations": len(missing),
        "successful_generated": status_counts.get(GENERATED, 0),
        "flagged_uncertain": status_counts.get(FLAGGED, 0),
        "failed": status_counts.get(FAILED, 0),
        "unreadable_or_changed_images": status_counts.get(UNREADABLE, 0),
        "duplicate_description_groups": sum(1 for ids in text_groups.values() if len(ids) > 1),
        "duplicate_description_rows": duplicate_text_rows,
        "duplicate_description_examples": duplicate_examples,
        "checks": checks,
        "clean_records": len(ok_rows),
        "records_requiring_human_verification": len(verdicts) - len(ok_rows),
        "integrity_blocked": integrity_blocked,
        "emittable": bool(ok_rows) and not integrity_blocked,
        "decision": "EMITTABLE" if (ok_rows and not integrity_blocked) else "BLOCKED",
        "limitations": [
            "Automated checks are structural and lexical; they do not prove that a description "
            "semantically matches its image.",
            "Records excluded here are excluded, never rewritten or replaced.",
            "Only store-integrity checks halt emission outright; other checks exclude the "
            "individual offending records and report the counts.",
            "Text/vocabulary overlap between generated descriptions and class labels is checked "
            "only for exact class-name phrases, not for paraphrases.",
        ],
    }
    return report, ok_rows, verdicts


def clean_rows(store_path: Path, *, dataset_root: Path, manifest_path: Path,
               split_files: Mapping[str, Path], labels: Mapping[int, str],
               verify_images: bool = True) -> tuple[list[dict[str, str]], dict]:
    report, ok_rows, _verdicts = audit_generation(
        store_path, dataset_root=dataset_root, manifest_path=manifest_path,
        split_files=split_files, labels=labels, verify_images=verify_images,
    )
    return ok_rows, report


def describe_counts(values: Sequence[str]) -> dict[str, int]:
    return dict(Counter(values))
