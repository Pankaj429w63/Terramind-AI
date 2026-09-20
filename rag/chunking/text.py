from __future__ import annotations

from dataclasses import dataclass

from rag.ingestion.loader import Document


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_id: str
    source: str
    title: str
    category: str
    metadata: dict[str, str]


def chunk_documents(documents: list[Document], chunk_size: int = 900, overlap: int = 120) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size.")
    chunks: list[Chunk] = []
    for document in documents:
        words = document.text.split()
        start = 0
        index = 0
        while start < len(words):
            text = " ".join(words[start : start + chunk_size]).strip()
            if text:
                chunks.append(Chunk(text, f"{document.source}#chunk-{index}", document.source, document.title, document.category, document.metadata))
            index += 1
            start += chunk_size - overlap
    return chunks
