from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests


ALLOWED_CATEGORIES = {"agriculture", "diseases", "treatment", "fertilizer", "plant care"}


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


def load_path(path: str | Path, category: str) -> list[Document]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in {".txt", ".md"})
    result: list[Document] = []
    for file in files:
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
