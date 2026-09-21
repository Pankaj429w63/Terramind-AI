#!/usr/bin/env python3
"""Populate the PlantWild disease knowledge base from verified source records."""
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
DISEASES_DIR = KB_ROOT / "diseases"
MANIFEST_PATH = DISEASES_DIR / "_manifest.diseases.json"
LAST_REVIEWED = "2026-09-21"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _populate_diseases import _disease_records
from _populate_diseases_2 import _disease_records_2
from _populate_diseases_3 import _disease_records_3
from _populate_diseases_4 import _disease_records_4


def _load_manifest() -> dict[str, dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("category") != "diseases":
        raise ValueError("Disease manifest category must be 'diseases'.")

    documents: dict[str, dict[str, Any]] = {}
    for crop_key, crop_group in manifest.get("crops", {}).items():
        crop = crop_group.get("crop")
        if crop != crop_key:
            raise ValueError(f"Manifest crop mismatch for {crop_key!r}: {crop!r}.")
        for entry in crop_group.get("documents", []):
            filename = entry.get("file")
            suffix = ".diseases.json"
            if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(suffix):
                raise ValueError(f"Invalid manifest filename for crop {crop!r}: {filename!r}.")
            key = filename.removesuffix(suffix)
            if key in documents:
                raise ValueError(f"Duplicate manifest key: {key}.")
            required = ("document_id", "class_index", "class_label", "disease")
            missing = [field for field in required if field not in entry]
            if missing:
                raise ValueError(f"Manifest entry {key!r} is missing {', '.join(missing)}.")
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
        _disease_records(),
        _disease_records_2(),
        _disease_records_3(),
        _disease_records_4(),
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
        "disease": entry["disease"],
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise ValueError(
                f"Existing document identity mismatch in {path}: {field} is "
                f"{document.get(field)!r}, expected {value!r}."
            )
    if not isinstance(metadata, dict) or metadata.get("class_label") != entry["class_label"]:
        raise ValueError(f"Existing document class_label mismatch in {path}.")
    return document


def _build_document(entry: dict[str, Any], source_record: Any, existing: dict[str, Any]) -> dict[str, Any]:
    source = asdict(source_record)
    metadata = dict(existing["metadata"])
    metadata.update(
        {
            "class_index": entry["class_index"],
            "class_label": entry["class_label"],
            "crop_slug": entry["crop"],
            "disease_slug": entry["file"].removesuffix(".diseases.json").removeprefix(f"{entry['crop']}_"),
            "status": "sourced",
            "placeholder_replaced": "true",
            "last_reviewed": LAST_REVIEWED,
        }
    )
    return {
        "schema_version": "1.0.0",
        "document_id": entry["document_id"],
        "category": "diseases",
        "source": source["source"],
        "title": source["title"],
        "crop": entry["crop"],
        "disease": entry["disease"],
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
    for key, entry in sorted(manifest_records.items(), key=lambda item: item[1]["class_index"]):
        path = DISEASES_DIR / entry["crop"] / entry["file"]
        existing = _read_existing_document(path, entry)
        document = _build_document(entry, source_records[key], existing)
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    written = populate()
    print(f"Wrote {len(written)} sourced disease documents.")


if __name__ == "__main__":
    main()
