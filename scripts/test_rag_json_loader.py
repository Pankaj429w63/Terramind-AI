"""Focused tests for KB JSON loading in rag/ingestion/loader.py.

Verifies the folder_of_json manifest contract:
  - JSON KB documents load with `content` folded into Document.text
  - title / crop / disease are concatenated for retrieval context
  - metadata (document_id, title, crop, category, source, url, reliability, ...)
    is preserved onto Document.metadata
  - underscore-prefixed files (manifests/schemas) are skipped
  - TXT/MD loading still works unchanged
  - broken JSON surfaces a clear error

Run: .venv\\Scripts\\python.exe scripts/test_rag_json_loader.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from rag.ingestion.loader import Document, load_path, load_sources  # noqa: E402

KB_ROOT = PROJECT / "data" / "knowledge_base"
CATEGORY_DIRS = {
    "diseases": "diseases",
    "treatment": "treatment",
    "fertilizer": "fertilizer",
    "plant care": "plant_care",
    "agriculture": "agriculture",
}

CHECK_COUNT = 0


def check(condition: bool, label: str) -> None:
    global CHECK_COUNT
    CHECK_COUNT += 1
    if not condition:
        raise AssertionError(f"FAILED: {label}")
    print(f"  ok  {label}")


# --------------------------------------------------------------------------
# 1. Unit tests on a synthetic KB document
# --------------------------------------------------------------------------
def test_synthetic_document() -> None:
    print("== synthetic JSON document ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        doc = {
            "schema_version": "1.0.0",
            "document_id": "11111111-2222-3333-4444-555555555555",
            "category": "fertilizer",
            "source": "Example Extension Service",
            "title": "Tomato Nutrition Guide",
            "crop": "tomato",
            "disease": None,
            "content": "Apply nitrogen in split doses through the season.",
            "url": "https://example.edu/tomato-nutrition",
            "author": "Example Author",
            "date": "2023-01-01",
            "region": "north_america",
            "reliability": "extension",
            "language": "en",
            "metadata": {"class_label": "tomato", "status": "sourced"},
        }
        (root / "tomato.fertilizer.json").write_text(json.dumps(doc), encoding="utf-8")

        docs = load_path(root, "fertilizer")
        check(len(docs) == 1, "one JSON document loaded")
        d = docs[0]
        check(isinstance(d, Document), "returns Document instances")
        check("Apply nitrogen in split doses" in d.text, "content folded into text")
        check("Tomato Nutrition Guide" in d.text, "title folded into text")
        check("Crop: tomato" in d.text, "crop folded into text")
        check("Disease:" not in d.text, "null disease omitted from text")
        check(d.category == "fertilizer", "category preserved")
        check(d.source.endswith("tomato.fertilizer.json"), "Document.source is unique per document")
        check(d.metadata["source"] == "Example Extension Service", "publisher preserved in metadata[source]")
        check(d.title == "Tomato Nutrition Guide", "title preserved")
        check(d.metadata["document_id"] == "11111111-2222-3333-4444-555555555555", "document_id in metadata")
        check(d.metadata["crop"] == "tomato", "crop in metadata")
        check(d.metadata["url"] == "https://example.edu/tomato-nutrition", "url in metadata")
        check(d.metadata["reliability"] == "extension", "reliability in metadata")
        check(d.metadata["category"] == "fertilizer", "category in metadata")
        check("path" in d.metadata, "path recorded in metadata")
    print("OK synthetic\n")


def test_disease_document_has_disease_field() -> None:
    print("== disease JSON document ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        doc = {
            "document_id": "aaaa",
            "category": "treatment",
            "source": "Cornell",
            "title": "Early Blight Management",
            "crop": "tomato",
            "disease": "early blight",
            "content": "Rotate away from solanaceous crops.",
            "url": "https://example.edu/eb",
            "reliability": "extension",
        }
        (root / "tomato_early_blight.treatment.json").write_text(json.dumps(doc), encoding="utf-8")
        docs = load_path(root, "treatment")
        check(len(docs) == 1, "one treatment document loaded")
        check("Disease: early blight" in docs[0].text, "disease folded into text when present")
        check(docs[0].metadata["disease"] == "early blight", "disease preserved in metadata")
    print("OK disease\n")


def test_underscore_files_skipped() -> None:
    print("== underscore-prefixed files skipped ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = {"document_id": "x", "category": "agriculture", "source": "s", "content": "body text here",
                "title": "t", "url": "https://example.edu/x", "reliability": "extension"}
        (root / "apple.agriculture.json").write_text(json.dumps(good), encoding="utf-8")
        (root / "_manifest.agriculture.json").write_text(json.dumps({"category": "agriculture"}), encoding="utf-8")
        (root / "_schemas.json").write_text(json.dumps({"schema": 1}), encoding="utf-8")
        docs = load_path(root, "agriculture")
        check(len(docs) == 1, "manifest/schema files are not ingested as documents")
    print("OK underscore\n")


def test_txt_md_still_loads() -> None:
    print("== TXT/MD loading preserved ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "note.txt").write_text("plain text knowledge", encoding="utf-8")
        (root / "guide.md").write_text("# markdown knowledge", encoding="utf-8")
        docs = load_path(root, "diseases")
        check(len(docs) == 2, "both .txt and .md loaded")
        titles = {d.title for d in docs}
        check(titles == {"note", "guide"}, f"txt/md titles from stems: {titles}")
        check(all(d.category == "diseases" for d in docs), "txt/md keep manifest category")
        check(all(d.metadata.get("path") for d in docs), "txt/md keep path metadata")
    print("OK txt/md\n")


def test_mixed_directory() -> None:
    print("== mixed JSON + TXT + MD ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.txt").write_text("alpha", encoding="utf-8")
        (root / "b.md").write_text("beta", encoding="utf-8")
        doc = {"document_id": "y", "category": "plant care", "source": "s", "content": "gamma",
               "title": "Gamma", "url": "https://example.edu/y", "reliability": "extension"}
        (root / "c.plant_care.json").write_text(json.dumps(doc), encoding="utf-8")
        docs = load_path(root, "plant care")
        check(len(docs) == 3, "all three formats loaded together")
    print("OK mixed\n")


def test_broken_json_raises() -> None:
    print("== broken JSON raises clear error ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "broken.diseases.json").write_text("{not valid json", encoding="utf-8")
        raised = False
        try:
            load_path(root, "diseases")
        except ValueError as exc:
            raised = "Invalid JSON" in str(exc)
        check(raised, "malformed JSON raises ValueError naming the file")
    print("OK broken\n")


def test_missing_content_raises() -> None:
    print("== JSON without content raises ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "empty.treatment.json").write_text(
            json.dumps({"document_id": "z", "category": "treatment", "title": "no body"}),
            encoding="utf-8",
        )
        raised = False
        try:
            load_path(root, "treatment")
        except ValueError as exc:
            raised = "content" in str(exc)
        check(raised, "document with no usable content field raises")
    print("OK missing content\n")


# --------------------------------------------------------------------------
# 2. Integration test against the real KB
# --------------------------------------------------------------------------
def test_real_kb_counts() -> None:
    print("== real knowledge base ==")
    total = 0
    by_category: dict[str, int] = {}
    for category, folder in CATEGORY_DIRS.items():
        docs = load_path(KB_ROOT / folder, category)
        by_category[category] = len(docs)
        total += len(docs)
        print(f"  {category:<12} {len(docs)} documents")
    check(total == 218, f"218 KB documents load (got {total})")
    for category, expected in {"diseases": 58, "treatment": 58, "fertilizer": 34,
                               "plant care": 34, "agriculture": 34}.items():
        check(by_category[category] == expected, f"{category} == {expected}")

    # spot-check metadata fidelity on a real document
    docs = load_path(KB_ROOT / "fertilizer", "fertilizer")
    apple = next(d for d in docs if d.metadata.get("crop") == "apple")
    check(apple.metadata.get("document_id") == "5ba13e3c-0612-5d0d-be0e-bb54bb4c05c1",
          "real apple document_id preserved")
    check(apple.metadata.get("reliability") == "extension", "real reliability preserved")
    check(apple.metadata.get("url", "").startswith("https://"), "real url preserved")
    check("Crop: apple" in apple.text, "real content context built")

    # load_sources (the path the pipeline uses) must also work
    sources = [(str(KB_ROOT / folder), category) for category, folder in CATEGORY_DIRS.items()]
    via_sources = load_sources(sources)
    check(len(via_sources) == 218, f"load_sources returns 218 documents (got {len(via_sources)})")

    # Regression guard: chunk_id is derived from Document.source by the chunker,
    # so every KB document must contribute a unique source to avoid silent
    # chunk collisions/overwrites in the vector store.
    from rag.chunking.text import chunk_documents  # noqa: PLC0415

    chunks = chunk_documents(via_sources)
    ids = [c.chunk_id for c in chunks]
    check(len(set(ids)) == len(ids), f"all {len(ids)} chunk_ids unique (got {len(set(ids))} unique)")
    print("OK real kb\n")


def main() -> int:
    test_synthetic_document()
    test_disease_document_has_disease_field()
    test_underscore_files_skipped()
    test_txt_md_still_loads()
    test_mixed_directory()
    test_broken_json_raises()
    test_missing_content_raises()
    test_real_kb_counts()
    print(f"ALL TESTS PASSED ({CHECK_COUNT} assertions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
