#!/usr/bin/env python3
"""Populate the PlantWild fertilizer knowledge base from verified source records."""
from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
KB_ROOT = PROJECT_ROOT / "data" / "knowledge_base"
FERTILIZER_DIR = KB_ROOT / "fertilizer"
MANIFEST_PATH = FERTILIZER_DIR / "_manifest.fertilizer.json"
LAST_REVIEWED = "2026-09-22"
SUFFIX = ".fertilizer.json"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _populate_fertilizers import (
    _fertilizer_records,
    _fertilizer_records_2,
    _fertilizer_records_3,
    _fertilizer_records_4,
)


def _load_manifest() -> dict[str, dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("category") != "fertilizer":
        raise ValueError("Fertilizer manifest category must be 'fertilizer'.")

    documents: dict[str, dict[str, Any]] = {}
    for crop_key, crop_group in manifest.get("crops", {}).items():
        crop = crop_group.get("crop")
        if not isinstance(crop, str) or not crop:
            raise ValueError(f"Manifest crop value missing for key {crop_key!r}.")
        for entry in crop_group.get("documents", []):
            filename = entry.get("file")
            if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(SUFFIX):
                raise ValueError(f"Invalid manifest filename for crop {crop!r}: {filename!r}.")
            key = filename.removesuffix(SUFFIX)
            if key in documents:
                raise ValueError(f"Duplicate manifest key: {key}.")
            if "document_id" not in entry:
                raise ValueError(f"Manifest entry {key!r} is missing document_id.")
            documents[key] = {"crop": crop, **entry}

    expected_total = manifest.get("total_documents")
    if expected_total != len(documents):
        raise ValueError(
            f"Manifest total_documents is {expected_total!r}, but contains {len(documents)} records."
        )
    return documents


def _merge_source_records() -> dict[str, Any]:
    records: dict[str, Any] = {}
    parts: Iterable[dict[str, Any]] = (
        _fertilizer_records(),
        _fertilizer_records_2(),
        _fertilizer_records_3(),
        _fertilizer_records_4(),
    )
    for part in parts:
        duplicate_keys = records.keys() & part.keys()
        if duplicate_keys:
            raise ValueError(f"Duplicate source records: {', '.join(sorted(duplicate_keys))}.")
        records.update(part)
    return records


def _read_existing_document(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest document does not exist: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    metadata = document.get("metadata")
    expected = {
        "document_id": entry["document_id"],
        "crop": entry["crop"],
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise ValueError(
                f"Existing document identity mismatch in {path}: {field} is "
                f"{document.get(field)!r}, expected {value!r}."
            )
    if not isinstance(metadata, dict) or metadata.get("class_label") != entry["crop"]:
        raise ValueError(f"Existing document class_label mismatch in {path}.")
    return document


def _build_document(entry: dict[str, Any], source_record: Any, existing: dict[str, Any]) -> dict[str, Any]:
    source = asdict(source_record)
    metadata = dict(existing["metadata"])
    metadata["status"] = "sourced"
    metadata["placeholder_replaced"] = "true"
    metadata["last_reviewed"] = LAST_REVIEWED
    return {
        "schema_version": "1.0.0",
        "document_id": entry["document_id"],
        "category": "fertilizer",
        "source": source["source"],
        "title": source["title"],
        "crop": entry["crop"],
        "disease": None,
        "content": source["content"],
        "url": source["url"],
        "author": source["author"],
        "date": source["date"],
        "region": source["region"],
        "reliability": source["reliability"],
        "language": "en",
        "tags": source["tags"],
        "metadata": metadata,
    }


def _update_manifest(manifest_records: dict[str, dict[str, Any]]) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for crop_group in manifest.get("crops", {}).values():
        for entry in crop_group.get("documents", []):
            entry["status"] = "sourced"
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def populate() -> list[Path]:
    manifest_records = _load_manifest()
    source_records = _merge_source_records()
    missing_sources = manifest_records.keys() - source_records.keys()
    unexpected_sources = source_records.keys() - manifest_records.keys()
    if missing_sources or unexpected_sources:
        details = []
        if missing_sources:
            details.append(f"missing sources: {', '.join(sorted(missing_sources))}")
        if unexpected_sources:
            details.append(f"unexpected sources: {', '.join(sorted(unexpected_sources))}")
        raise ValueError("Source/manifest coverage mismatch: " + "; ".join(details))

    written: list[Path] = []
    for key, entry in sorted(manifest_records.items()):
        path = FERTILIZER_DIR / key / entry["file"]
        existing = _read_existing_document(path, entry)
        document = _build_document(entry, source_records[key], existing)
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    _update_manifest(manifest_records)
    return written


def main() -> None:
    written = populate()
    print(f"Wrote {len(written)} sourced fertilizer documents.")


if __name__ == "__main__":
    main()
