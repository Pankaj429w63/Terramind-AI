"""Blinded annotation workflow for collecting genuine, image-specific observations.

Annotators must never see disease labels, class names, label-derived prompts, or the
official image path: PlantWild paths embed the class name (for example
``apple scab/1.jpg``), so exposing them would leak the target. Annotators therefore work
against opaque aliases. The alias -> sample_id mapping is kept sealed and is used only by
the curator when collating the final six-column paired-text CSV that the training loader
accepts unchanged (``sample_id, image_path, text, text_origin, provenance,
source_reference``).

Nothing in this module invents observations. Collation copies annotated text verbatim and
refuses to emit a training CSV until every observation is approved by review.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import re
import secrets
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from .data import ALLOWED_TEXT_ORIGINS, validate_provenance, validate_text_not_label_derived

WORKBOOK_COLUMNS = ("alias", "observation", "observer_id", "observed_at")
REVIEW_COLUMNS = ("sample_id", "review_status", "reviewer", "reviewed_at", "notes")
OBSERVATION_COLUMNS = ("sample_id", "image_path", "text", "text_origin", "provenance", "source_reference")
REVIEW_STATUSES = ("pending", "approved", "rejected", "disputed")
ALIAS_PREFIX = "obs-"
# Columns that would break annotator blinding if they appeared in the workbook.
BLINDING_FORBIDDEN_COLUMNS = frozenset({
    "image_path", "path", "relative_path", "sample_id", "class_id", "class_name",
    "label", "target", "target_label", "split", "official_split", "disease",
})


def new_salt() -> str:
    """Return a fresh random salt so aliases cannot be brute-forced back to paths."""
    return secrets.token_hex(16)


def alias_for(sample_id: str, salt: str) -> str:
    if not salt:
        raise ValueError("Alias salt must not be empty")
    digest = hashlib.sha256(f"{salt}:{sample_id}".encode("utf-8")).hexdigest()
    return f"{ALIAS_PREFIX}{digest[:12]}"


def build_alias_map(sample_ids: Sequence[str], salt: str) -> dict[str, str]:
    """Return a deterministic alias -> sample_id mapping for the given eligible samples."""
    mapping: dict[str, str] = {}
    for sample_id in sorted(set(sample_ids)):
        alias = alias_for(sample_id, salt)
        if alias in mapping:
            raise ValueError("Alias collision detected; generate a new salt")
        mapping[alias] = sample_id
    if not mapping:
        raise ValueError("No eligible sample ids were provided for the annotation workbook")
    return mapping


def build_workbook_rows(alias_map: Mapping[str, str]) -> list[dict[str, str]]:
    """Return blank, alias-only rows; no path or label is ever placed in the workbook."""
    return [
        {"alias": alias, "observation": "", "observer_id": "", "observed_at": ""}
        for alias in sorted(alias_map)
    ]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [{key: (value or "").strip() for key, value in row.items() if key} for row in reader]
    return fieldnames, rows


def validate_observed_at(observed_at: str, alias: str = "") -> str:
    """Require the annotation timestamp to be an ISO 8601 date (``YYYY-MM-DD``)."""
    value = (observed_at or "").strip()
    label = f" for {alias!r}" if alias else ""
    if not value:
        raise ValueError(f"Observation{label} needs an observed_at date (YYYY-MM-DD)")
    try:
        datetime.date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"observed_at{label} must be an ISO date (YYYY-MM-DD), got {value!r}"
        ) from error
    return value


def load_annotator_workbook(path: Path) -> list[dict[str, str]]:
    """Load a (possibly partially) filled workbook, rejecting any label/path exposure."""
    fieldnames, rows = _read_csv(path)
    if not set(WORKBOOK_COLUMNS).issubset(fieldnames):
        raise ValueError(f"Annotation workbook requires columns: {list(WORKBOOK_COLUMNS)}")
    exposed = BLINDING_FORBIDDEN_COLUMNS.intersection({name.casefold() for name in fieldnames})
    if exposed:
        raise ValueError(f"Annotation workbook must not expose labels, paths, or splits: {sorted(exposed)}")
    seen: set[str] = set()
    for line_number, row in enumerate(rows, 2):
        alias = row.get("alias", "")
        if not alias:
            raise ValueError(f"Missing alias in annotation workbook at line {line_number}")
        if alias in seen:
            raise ValueError(f"Duplicate alias {alias!r} in annotation workbook")
        seen.add(alias)
        if row.get("observation"):
            if not row.get("observer_id"):
                raise ValueError(f"Observation for {alias!r} needs an observer_id")
            validate_observed_at(row.get("observed_at", ""), alias)
    return rows


def observation_issues(
    workbook_rows: Sequence[Mapping[str, str]],
    alias_map: Mapping[str, str],
    labels: Mapping[int, str],
) -> list[dict[str, str]]:
    """Return one actionable problem per invalid filled row; unfilled rows are not rejected.

    Reasons never repeat observation text, class labels, or image paths, so the report
    cannot be used as a shortcut back to the sealed mapping.
    """
    issues: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in workbook_rows:
        alias = row.get("alias", "")
        observation = (row.get("observation") or "").strip()
        if not observation:
            continue
        if not row.get("observer_id"):
            issues.append({"alias": alias, "reason": "missing observer_id"})
            continue
        try:
            validate_observed_at(row.get("observed_at", ""), alias)
        except ValueError as error:
            issues.append({"alias": alias, "reason": str(error)})
            continue
        if alias not in alias_map:
            issues.append({"alias": alias, "reason": "unknown alias: not present in the sealed mapping"})
            continue
        try:
            validate_text_not_label_derived(observation, labels)
        except ValueError as error:
            issues.append({"alias": alias, "reason": str(error)})
            continue
        if alias in seen:
            issues.append({"alias": alias, "reason": "duplicate observation row"})
            continue
        seen.add(alias)
    return issues


def load_review_ledger(path: Path) -> dict[str, dict[str, str]]:
    """Load the review ledger keyed by sample_id, validating statuses and uniqueness."""
    fieldnames, rows = _read_csv(path)
    if not set(REVIEW_COLUMNS).issubset(fieldnames):
        raise ValueError(f"Review ledger requires columns: {list(REVIEW_COLUMNS)}")
    ledger: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(rows, 2):
        sample_id = row.get("sample_id", "")
        status = row.get("review_status", "").casefold()
        if not sample_id:
            raise ValueError(f"Missing sample_id in review ledger at line {line_number}")
        if status not in REVIEW_STATUSES:
            raise ValueError(f"Unapproved review_status {status!r} at line {line_number}")
        if sample_id in ledger:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in review ledger")
        ledger[sample_id] = row
    return ledger


def collate_observations(
    workbook_rows: Sequence[Mapping[str, str]],
    alias_map: Mapping[str, str],
    labels: Mapping[int, str],
    *,
    text_origin: str,
    provenance: str,
    source_reference: str,
) -> list[dict[str, str]]:
    """Turn filled alias rows into the six-column paired CSV, verbatim, without inventing text."""
    if text_origin not in ALLOWED_TEXT_ORIGINS:
        raise ValueError(f"Unapproved text_origin {text_origin!r}")
    validate_provenance(provenance, source_reference)
    collated: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in workbook_rows:
        observation = (row.get("observation") or "").strip()
        if not observation:
            continue
        alias = row.get("alias", "")
        sample_id = alias_map.get(alias)
        if not sample_id:
            raise ValueError(f"Observation references an unknown alias: {alias!r}")
        validate_text_not_label_derived(observation, labels)
        if sample_id in seen:
            raise ValueError(f"Duplicate observation for sample {sample_id!r}")
        seen.add(sample_id)
        collated.append({
            "sample_id": sample_id, "image_path": sample_id, "text": observation,
            "text_origin": text_origin, "provenance": provenance, "source_reference": source_reference,
        })
    return collated


def require_reviewed(
    collated: Sequence[Mapping[str, str]], review_ledger: Mapping[str, Mapping[str, str]]
) -> dict[str, int]:
    """Block emission unless every observation has an approved, matching review entry."""
    missing: list[str] = []
    not_approved: list[str] = []
    for row in collated:
        sample_id = str(row["sample_id"])
        entry = review_ledger.get(sample_id)
        if entry is None:
            missing.append(sample_id)
        elif entry.get("review_status", "").casefold() != "approved":
            not_approved.append(sample_id)
    if missing:
        raise ValueError(f"Observations lack a review-ledger entry: {missing[:5]} ({len(missing)} total)")
    if not_approved:
        raise ValueError(f"Observations are not approved by review: {not_approved[:5]} ({len(not_approved)} total)")
    observed = {str(row["sample_id"]) for row in collated}
    extra = sorted(set(review_ledger) - observed)
    if extra:
        raise ValueError(f"Review ledger references unobserved samples: {extra[:5]} ({len(extra)} total)")
    return {"collated": len(collated), "approved": len(collated)}


def require_independent_review(
    workbook_rows: Sequence[Mapping[str, str]],
    alias_map: Mapping[str, str],
    review_ledger: Mapping[str, Mapping[str, str]],
) -> dict[str, int]:
    """Block emission when an approved observation was reviewed by its own annotator.

    Attribution is not review: for every approved observation the ledger ``reviewer``
    must be a named person different from the workbook ``observer_id``. Rows that carry
    no observation, no sealed-mapping entry, or no ledger entry are left to the other
    gates (``collate_observations`` and ``require_reviewed``); this gate only checks the
    approvals that already exist.
    """
    self_approved: list[str] = []
    anonymous: list[str] = []
    checked = 0
    for row in workbook_rows:
        observation = (row.get("observation") or "").strip()
        if not observation:
            continue
        alias = row.get("alias", "")
        sample_id = alias_map.get(alias)
        if not sample_id:
            continue
        entry = review_ledger.get(sample_id)
        if entry is None or entry.get("review_status", "").casefold() != "approved":
            continue
        reviewer = (entry.get("reviewer") or "").strip()
        if not reviewer:
            anonymous.append(alias)
            continue
        observer = (row.get("observer_id") or "").strip()
        checked += 1
        if observer and reviewer.casefold() == observer.casefold():
            self_approved.append(alias)
    if anonymous:
        raise ValueError(
            f"Approved observations need a named reviewer: {anonymous[:5]} ({len(anonymous)} total)"
        )
    if self_approved:
        raise ValueError(
            "An observation cannot be approved by its own annotator; a separate independent "
            f"reviewer is required: {self_approved[:5]} ({len(self_approved)} total)"
        )
    return {"independent_reviews": checked}


def require_instrument_version(provenance: str, instrument_version: str) -> str:
    """Ensure provenance records the elicitation instrument version used for collection.

    Provenance is the only field transmitted into training, so the instrument version must
    be embedded there; otherwise train and inference text cannot be traced to one instrument.
    """
    version = (instrument_version or "").strip()
    if not version:
        raise ValueError(
            "--instrument-version is required: pass the elicitation instrument version that was actually used"
        )
    if not re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(version)}(?![A-Za-z0-9_.-])", provenance or "", re.IGNORECASE):
        raise ValueError(
            f"provenance must record the elicitation instrument version {version!r}: "
            "add it to --provenance (for example '... instrument v1') or correct --instrument-version"
        )
    return version


def write_alias_map(alias_map: Mapping[str, str], path: Path, salt: str) -> Path:
    """Write the sealed alias map. Never share this file with annotators."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"salt": salt, "aliases": dict(alias_map)}, indent=2), encoding="utf-8")
    return path


def load_alias_map(path: Path) -> tuple[dict[str, str], str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Sealed alias map not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    salt = str(payload.get("salt", ""))
    aliases = payload.get("aliases")
    if not salt or not isinstance(aliases, dict) or not aliases:
        raise ValueError("Sealed alias map is malformed")
    return {str(alias): str(sample_id) for alias, sample_id in aliases.items()}, salt


def write_workbook(rows: Sequence[Mapping[str, str]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(WORKBOOK_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_observations_csv(rows: Sequence[Mapping[str, str]], path: Path) -> Path:
    """Write the paired CSV, replacing the destination only once the file is complete.

    A temporary file is created next to the destination and moved into place with
    ``os.replace`` so a failure can never leave a partial or misleading training CSV behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent,
        prefix=f"{path.name}.", suffix=".partial", delete=False,
    )
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=list(OBSERVATION_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return path
