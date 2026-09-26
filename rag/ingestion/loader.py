from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests


ALLOWED_CATEGORIES = {"agriculture", "diseases", "treatment", "fertilizer", "plant care"}

TEXT_SUFFIXES = {".txt", ".md"}
JSON_SUFFIXES = {".json"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | JSON_SUFFIXES

# Fields lifted from a KB JSON document into Document.metadata, per the
# folder_of_json manifest contract (see data/knowledge_base/rag_ingestion.manifest.json).
KB_METADATA_FIELDS = (
    "document_id",
    "title",
    "category",
    "source",
    "url",
    "reliability",
    "crop",
    "disease",
    "author",
    "date",
    "region",
    "language",
)


@dataclass(frozen=True)
class Document:
    text: str
    source: str
    title: str
    category: str
    metadata: dict[str, str]


def _category(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in ALLOWED_CATEGORIES:
        raise ValueError(f"Unsupported category '{value}'. Use one of {sorted(ALLOWED_CATEGORIES)}.")
    return normalized


def _is_kb_document(path: Path) -> bool:
    """Underscore-prefixed files (manifests, schemas) are not KB documents."""
    return not path.name.startswith("_")


def _document_text(doc: dict[str, Any], path: Path) -> str:
    """Build the retrievable text for a KB JSON document.

    The folder_of_json manifest contract requires the `content` field to be
    concatenated with identifying context (title / crop / disease) so the chunk
    is self-describing for retrieval. Falls back to `summary`/`description` if a
    document lacks `content`, preserving the schema's intent that content is the
    authoritative body.
    """
    content = doc.get("content")
    if not isinstance(content, str) or not content.strip():
        for alt in ("summary", "description", "text"):
            candidate = doc.get(alt)
            if isinstance(candidate, str) and candidate.strip():
                content = candidate
                break
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"KB document has no usable content field: {path}")

    parts: list[str] = []
    title = doc.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())
    crop = doc.get("crop")
    if isinstance(crop, str) and crop.strip():
        parts.append(f"Crop: {crop.strip()}")
    disease = doc.get("disease")
    if isinstance(disease, str) and disease.strip():
        parts.append(f"Disease: {disease.strip()}")
    parts.append(content.strip())
    return "\n".join(parts)


def _load_json(path: Path, category: str) -> Document | None:
    """Load one KB JSON document into a Document, or None if it is not a KB doc.

    Raises ValueError only for files that claim to be KB documents but cannot be
    parsed into usable text, so ingestion surfaces genuinely broken documents.
    """
    if not _is_kb_document(path):
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in KB document {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"KB document is not a JSON object: {path}")

    resolved = str(path.resolve())
    text = _document_text(doc, path)

    doc_category = doc.get("category")
    if isinstance(doc_category, str) and doc_category.strip():
        effective_category = _category(doc_category)
    else:
        effective_category = _category(category)

    title = doc.get("title")
    if not isinstance(title, str) or not title.strip():
        title = path.stem

    # Document.source must be unique per document: the chunker derives chunk_id
    # from it (f"{source}#chunk-{n}"), so sharing a publisher string across many
    # KB documents would make their chunks collide and overwrite in the store.
    # The human-readable publisher is preserved in metadata["source"].
    source = resolved

    metadata: dict[str, str] = {"path": resolved}
    for field in KB_METADATA_FIELDS:
        value = doc.get(field)
        if value is None:
            continue
        metadata[field] = value if isinstance(value, str) else str(value)
    metadata["path"] = resolved
    publisher = metadata.get("source")
    if not isinstance(publisher, str) or not publisher.strip():
        metadata["source"] = source

    return Document(text, source, title, effective_category, metadata)


def load_path(path: str | Path, category: str) -> list[Document]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        files = [root]
    else:
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    result: list[Document] = []
    for file in files:
        if file.suffix.lower() in JSON_SUFFIXES:
            document = _load_json(file, category)
            if document is not None:
                result.append(document)
            continue
        text = file.read_text(encoding="utf-8").strip()
        if text:
            result.append(Document(text, str(file.resolve()), file.stem, _category(category), {"path": str(file.resolve())}))
    return result


def load_url(url: str, category: str, timeout: int = 20) -> Document:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Document URL must use http or https.")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    title = parsed.path.rstrip("/").split("/")[-1] or parsed.netloc
    return Document(response.text.strip(), url, title, _category(category), {"url": url, "http_status": str(response.status_code)})


def load_sources(sources: Iterable[tuple[str, str]]) -> list[Document]:
    documents: list[Document] = []
    for source, category in sources:
        if urlparse(source).scheme in {"http", "https"}:
            documents.append(load_url(source, category))
        else:
            documents.extend(load_path(source, category))
    return documents
