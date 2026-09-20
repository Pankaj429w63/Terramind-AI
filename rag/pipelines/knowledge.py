from __future__ import annotations

from typing import Any

from rag.chunking.text import Chunk, chunk_documents
from rag.embeddings.tfidf import TfidfEmbeddings
from rag.ingestion.loader import Document, load_sources
from rag.qdrant.store import VectorStore
from rag.reranking.lexical import rerank


class KnowledgePipeline:
    def __init__(self, embeddings: TfidfEmbeddings | None = None, store: VectorStore | None = None) -> None:
        self.embeddings = embeddings or TfidfEmbeddings()
        self.store = store or VectorStore()
        self.indexed = self.embeddings.is_fitted and self.store.stats()["total_points"] > 0

    def ingest(self, sources: list[dict[str, str]]) -> dict[str, Any]:
        if not sources:
            raise ValueError("No ingestion sources provided.")
        documents: list[Document] = []
        errors: list[dict[str, Any]] = []
        for item in sources:
            src = item.get("source")
            category = item.get("category")
            if not isinstance(src, str) or not src:
                errors.append({"source": src, "category": category, "error": "missing source"})
                continue
            if not isinstance(category, str) or not category:
                errors.append({"source": src, "category": category, "error": "missing category"})
                continue
            try:
                loaded = load_sources([(src, category)])
                documents.extend(loaded)
            except Exception as exc:
                errors.append({"source": src, "category": category, "error": str(exc)})
        if not documents:
            msg = "No documents could be loaded."
            if errors:
                msg += " Errors: " + "; ".join(e["error"] for e in errors)
            raise RuntimeError(msg)
        chunks: list[Chunk] = chunk_documents(documents)
        new_texts = [chunk.text for chunk in chunks]
        new_corpus_ids = [chunk.chunk_id for chunk in chunks]

        refit_required = self.embeddings.should_refit(new_texts)

        if refit_required:
            existing_payloads = self.store.enumerate_payloads()
            existing_texts_map: dict[str, str] = {}
            existing_chunks_map: dict[str, Chunk] = {}
            for p in existing_payloads:
                cid = p.get("chunk_id")
                text = p.get("text")
                if not isinstance(cid, str) or not isinstance(text, str):
                    continue
                existing_texts_map[cid] = text
                existing_chunks_map[cid] = Chunk(
                    text=text,
                    chunk_id=cid,
                    source=p.get("source") or "unknown",
                    title=p.get("title") or cid,
                    category=p.get("category") or "uncategorized",
                    metadata=p.get("metadata") or {},
                )
            combined_ids = list(existing_texts_map.keys()) + new_corpus_ids
            combined_texts = [existing_texts_map[cid] for cid in existing_texts_map.keys()] + new_texts
            self.embeddings.fit(combined_texts, corpus_ids=combined_ids)
            if existing_chunks_map:
                re_ids = list(existing_chunks_map.keys())
                re_chunks = [existing_chunks_map[cid] for cid in re_ids]
                re_texts = [existing_chunks_map[cid].text for cid in re_ids]
                re_vectors = self.embeddings.encode(re_texts)
                self.store.upsert(re_chunks, re_vectors)
        else:
            self.embeddings.update_corpus(dict(zip(new_corpus_ids, new_texts, strict=False)))

        new_vectors = self.embeddings.encode(new_texts)
        inserted = self.store.upsert(chunks, new_vectors)
        self.indexed = True

        return {
            "documents": len(documents),
            "chunks": inserted,
            "backend": self.store.backend,
            "refit_performed": refit_required,
            "sources": sorted({doc.source for doc in documents}),
            "errors": errors,
            "stats": self.store.stats(),
            "embeddings": {
                "vector_size": self.embeddings.vector_size,
                "vocabulary_size": self.embeddings.vocabulary_size,
                "corpus_size": len(self.embeddings.corpus),
            },
        }

    def retrieve(self, query: str, limit: int = 5, category: str | None = None) -> dict[str, Any]:
        grounded: list[dict[str, Any]] = []
        retrieval_error: str | None = None
        if not self.indexed:
            retrieval_error = "Knowledge base is empty; ingest agricultural documents to enable retrieval."
        else:
            try:
                vector = self.embeddings.encode([query])[0]
                results = self.store.search(vector, limit=limit * 3)
                if category:
                    results = [r for r in results if r.get("category") == category]
                ranked = rerank(query, results, limit)
                for item in ranked:
                    if float(item.get("retrieval_score") or 0.0) <= 0:
                        continue
                    excerpt = item.get("text") or ""
                    grounded.append({
                        "text": item.get("text") or "",
                        "excerpt": (excerpt[:500].rstrip() + "…") if len(excerpt) > 500 else excerpt,
                        "source": item.get("source"),
                        "title": item.get("title"),
                        "category": item.get("category"),
                        "score": item.get("retrieval_score") or item.get("score"),
                        "chunk_id": item.get("chunk_id"),
                        "raw_score": item.get("score"),
                        "relevance_rank": item.get("relevance_rank"),
                    })
            except Exception as error:
                retrieval_error = str(error)
        return {
            "retrieved": len(grounded),
            "sources": grounded,
            "error": retrieval_error,
            "query": query,
            "category": category,
            "grounded": len(grounded) > 0,
        }

    def stats(self) -> dict[str, Any]:
        store_stats = self.store.stats()
        corpus_size = len(self.embeddings.corpus)
        total_points = int(store_stats.get("total_points") or 0)
        try:
            model_size = __import__("os").path.getsize(self.embeddings.model_path) if __import__("os").path.exists(str(self.embeddings.model_path)) else 0
        except Exception:
            model_size = 0
        return {
            **store_stats,
            "indexed": self.indexed or (model_size > 0 and total_points > 0),
            "embeddings": {
                "backend": "tfidf",
                "vector_size": self.embeddings.vector_size,
                "vocabulary_size": self.embeddings.vocabulary_size,
                "corpus_chunks_tracked": corpus_size,
                "model_size_bytes": model_size,
                "fitted": self.embeddings.is_fitted,
            },
        }


pipeline = KnowledgePipeline()

