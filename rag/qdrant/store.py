from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from rag.chunking.text import Chunk

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_QDRANT_PATH = _PROJECT_ROOT / "outputs" / "qdrant_data"
_LOCAL_PERSIST_PATH = _PROJECT_ROOT / "outputs" / "rag" / "vector_store_memory.json"
_LOCAL_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)


def _default_qdrant_path() -> str | None:
    explicit = os.getenv("QDRANT_PATH")
    if explicit:
        return explicit
    if not os.getenv("QDRANT_URL"):
        _DEFAULT_QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        return str(_DEFAULT_QDRANT_PATH)
    return None


class VectorStore:
    def __init__(self, collection: str = "terramind_knowledge") -> None:
        self.collection = collection
        self.client = None
        self._points: list[dict[str, Any]] = []
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except ImportError:
            QdrantClient = None
        self._models = locals()
        connected = False
        if QdrantClient:
            url = os.getenv("QDRANT_URL")
            api_key = os.getenv("QDRANT_API_KEY")
            path = _default_qdrant_path()
            try:
                if url:
                    self.client = QdrantClient(url=url, api_key=api_key)
                elif path:
                    self.client = QdrantClient(path=path)
                connected = self.client is not None
            except Exception:
                self.client = None
                connected = False
        if not connected:
            self._load_local()

    @property
    def backend(self) -> str:
        return "qdrant" if self.client else "local-memory"

    # ---------------------------
    # Local persistence (fallback)
    # ---------------------------
    def _load_local(self) -> None:
        try:
            if _LOCAL_PERSIST_PATH.exists():
                self._points = json.loads(_LOCAL_PERSIST_PATH.read_text(encoding="utf-8"))
                if not isinstance(self._points, list):
                    self._points = []
        except Exception:
            self._points = []

    def _save_local(self) -> None:
        try:
            _LOCAL_PERSIST_PATH.write_text(json.dumps(self._points, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # ---------------------------
    # Shared chunk ID helpers
    # ---------------------------
    @staticmethod
    def point_id_for_chunk(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    # ---------------------------
    # Enumerate / mutate all points (needed for vocabulary refits)
    # ---------------------------
    def enumerate_payloads(self) -> list[dict[str, Any]]:
        if self.client:
            payloads: list[dict[str, Any]] = []
            try:
                if self.client.collection_exists(self.collection):
                    total = None
                    try:
                        info = self.client.get_collection(self.collection)
                        total = getattr(info, "points_count", None)
                    except Exception:
                        total = None
                    scroll_limit = min(10000, max(1000, (total or 0) + 1000))
                    offset: Any = None
                    while True:
                        try:
                            res = self.client.scroll(
                                self.collection,
                                limit=scroll_limit,
                                offset=offset,
                                with_vectors=False,
                                with_payload=True,
                            )
                        except TypeError:
                            res = self.client.scroll(
                                self.collection,
                                limit=scroll_limit,
                                with_vectors=False,
                                with_payload=True,
                            )
                        if isinstance(res, tuple):
                            batch, next_offset = res
                        else:
                            batch = res
                            next_offset = None
                        for p in batch:
                            payload = getattr(p, "payload", None) or {}
                            if isinstance(payload, dict):
                                payloads.append(payload)
                        if not batch or not next_offset:
                            break
                        offset = next_offset
                        if len(payloads) > 1_000_000:
                            break
            except Exception:
                pass
            return payloads
        return [dict(p.get("payload") or {}) for p in self._points]

    def clear(self) -> None:
        if self.client:
            try:
                if self.client.collection_exists(self.collection):
                    self.client.delete_collection(self.collection)
            except Exception:
                pass
        self._points = []
        self._save_local()

    # ---------------------------
    # Idempotent upsert (deduplicate by chunk_id)
    # ---------------------------
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors) or not chunks:
            raise ValueError("Chunks and vectors must be non-empty and have equal length.")
        if self.client:
            models = self._models
            size = len(vectors[0])
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    self.collection,
                    vectors_config=models["VectorParams"](size=size, distance=models["Distance"].COSINE),
                )
            points = [
                models["PointStruct"](
                    id=self.point_id_for_chunk(chunk.chunk_id),
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "title": chunk.title,
                        "category": chunk.category,
                        "metadata": chunk.metadata,
                    },
                )
                for chunk, vector in zip(chunks, vectors)
            ]
            self.client.upsert(self.collection, points=points, wait=True)
            return len(chunks)
        # local-memory: deduplicate in-place by chunk_id (update existing or append)
        by_id: dict[str, int] = {}
        for i, p in enumerate(self._points):
            cid = (p.get("payload") or {}).get("chunk_id")
            if isinstance(cid, str):
                by_id[cid] = i
        for chunk, vector in zip(chunks, vectors):
            record = {
                "vector": vector,
                "payload": {
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "title": chunk.title,
                    "category": chunk.category,
                    "metadata": chunk.metadata,
                },
            }
            existing = by_id.get(chunk.chunk_id)
            if existing is not None:
                self._points[existing] = record
            else:
                by_id[chunk.chunk_id] = len(self._points)
                self._points.append(record)
        self._save_local()
        return len(chunks)

    # ---------------------------
    # Search / stats
    # ---------------------------
    def search(self, vector: list[float], limit: int = 8) -> list[dict[str, Any]]:
        if self.client:
            try:
                return [
                    {"score": getattr(item, "score", 0.0), **(getattr(item, "payload", None) or {})}
                    for item in self.client.search(self.collection, query_vector=vector, limit=limit)
                ]
            except Exception:
                return []
        import numpy as np
        query = np.asarray(vector, dtype=float)
        scored: list[dict[str, Any]] = []
        for point in self._points:
            candidate = np.asarray(point.get("vector", []), dtype=float)
            if candidate.size == 0 or query.size != candidate.size:
                continue
            denom = np.linalg.norm(query) * np.linalg.norm(candidate)
            score = float(np.dot(query, candidate) / denom) if denom else 0.0
            scored.append({"score": score, **(point.get("payload") or {})})
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]

    def stats(self) -> dict[str, Any]:
        if self.client:
            try:
                info = None
                try:
                    if self.client.collection_exists(self.collection):
                        info = self.client.get_collection(self.collection)
                except Exception:
                    info = None
                total = getattr(info, "points_count", None) if info is not None else None
                payloads = self.enumerate_payloads()
                by_category: dict[str, int] = {}
                by_source: dict[str, int] = {}
                for payload in payloads:
                    if not isinstance(payload, dict):
                        continue
                    cat = str(payload.get("category") or "uncategorized")
                    by_category[cat] = by_category.get(cat, 0) + 1
                    src = str(payload.get("source") or "unknown")
                    by_source[src] = by_source.get(src, 0) + 1
                total_points = int(total) if isinstance(total, int) else len(payloads)
                return {
                    "backend": self.backend,
                    "collection": self.collection,
                    "total_points": total_points,
                    "by_category": by_category,
                    "by_source_count": len(by_source),
                }
            except Exception:
                return {
                    "backend": self.backend,
                    "collection": self.collection,
                    "total_points": 0,
                    "by_category": {},
                    "by_source_count": 0,
                }
        by_category: dict[str, int] = {}
        by_source: set[str] = set()
        for point in self._points:
            payload = point.get("payload") or {}
            cat = str(payload.get("category") or "uncategorized")
            by_category[cat] = by_category.get(cat, 0) + 1
            src = payload.get("source")
            if isinstance(src, str):
                by_source.add(src)
        return {
            "backend": self.backend,
            "collection": self.collection,
            "total_points": len(self._points),
            "by_category": by_category,
            "by_source_count": len(by_source),
        }
