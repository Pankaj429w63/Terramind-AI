from __future__ import annotations

import json

from agents.generation import AnswerGenerator
from agents.workflow import ChatSupervisorAgent
from rag.chunking.text import Chunk
from rag.qdrant.store import VectorStore


class FakePipeline:
    def __init__(self, sources=None, error=None):
        self.sources = sources or []
        self.error = error
        self.queries = []

    def retrieve(self, query, limit=5, category=None):
        self.queries.append((query, limit, category))
        selected = [s for s in self.sources if category is None or s["category"] == category]
        return {"query": query, "retrieved": len(selected), "sources": selected, "error": self.error, "grounded": bool(selected)}


SOURCES = [
    {"text": f"Evidence for {category} crops.", "excerpt": f"Evidence for {category} crops.", "source": f"{category}.json", "title": f"{category.title()} guide", "category": category, "score": 0.9, "chunk_id": f"{category}-1"}
    for category in ("diseases", "treatment", "fertilizer", "plant care", "agriculture")
]


def test_chat_routes_each_category_and_preserves_citations(monkeypatch):
    monkeypatch.delenv("TERRAMIND_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TERRAMIND_LLM_MODEL", raising=False)
    routes = {
        "How do I identify disease spots?": "Diagnosis Agent",
        "How should I treat this?": "Treatment Agent",
        "What fertilizer and nitrogen?": "Fertilizer Agent",
        "How often should I water?": "Care Agent",
        "How does agriculture work?": "Research Agent",
    }
    for query, expected in routes.items():
        pipeline = FakePipeline(SOURCES)
        ctx = ChatSupervisorAgent(pipeline).run(query)
        assert ctx.specialist == expected
        assert ctx.generation["mode"] == "evidence_template"
        assert "[S1]" in ctx.response
        assert ctx.retrieval["sources"][0]["citation_id"] == "S1"
        assert any(e["agent_name"] == expected and e["status"] == "completed" for e in ctx.events)


def test_low_confidence_context_is_not_confirmed_and_is_retrieval_context():
    pipeline = FakePipeline(SOURCES)
    ctx = ChatSupervisorAgent(pipeline).run(
        "What should I do?", {"predicted_class": {"label": "apple_scab"}, "low_confidence": True}
    )
    assert pipeline.queries[0][0].startswith("apple_scab ")
    assert "not a confirmed diagnosis" in ctx.response
    assert "expert review" in ctx.response


def test_empty_and_unavailable_retrieval_are_honest():
    empty = ChatSupervisorAgent(FakePipeline()).run("What is the best approach?")
    assert empty.generation == {"text": empty.response, "mode": "no_evidence", "status": "no_retrieval_results"}
    unavailable = ChatSupervisorAgent(FakePipeline(error="store offline")).run("What is the best approach?")
    assert any(e["agent_name"] == "research_rag" and e["status"] == "failed" for e in unavailable.events)
    assert "could not find supporting" in unavailable.response


def test_specialist_and_rag_can_be_disabled_without_fake_events():
    pipeline = FakePipeline(SOURCES)
    ctx = ChatSupervisorAgent(pipeline).run("What is crop care?", enable_specialist=False, enable_rag=False)
    assert pipeline.queries == []
    assert ctx.specialist == ""
    assert not any(e["agent_name"] == "Care Agent" for e in ctx.events)
    assert next(e for e in ctx.events if e["agent_name"] == "research_rag")["status"] == "skipped"
    assert ctx.generation["mode"] == "no_evidence"


def test_configured_provider_failure_falls_back_and_reports_error(monkeypatch):
    monkeypatch.setenv("TERRAMIND_LLM_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("TERRAMIND_LLM_MODEL", "test-model")
    generator = AnswerGenerator(timeout_seconds=0.01)
    monkeypatch.setattr(generator, "_remote", lambda *_: (_ for _ in ()).throw(TimeoutError()))
    result = generator.generate("question", SOURCES[:1])
    assert result["mode"] == "evidence_template"
    assert result["status"] == "provider_error: TimeoutError"
    assert "[S1]" in result["text"]


def test_local_vector_store_persists_and_reloads(tmp_path, monkeypatch):
    import rag.qdrant.store as store_module

    path = tmp_path / "index.json"
    monkeypatch.setattr(store_module, "_LOCAL_PERSIST_PATH", path)
    store = VectorStore.__new__(VectorStore)
    store.client = None
    store._points = []
    store.collection = "test"
    chunk = Chunk("evidence", "id-1", "source.json", "Source title", "diseases", {})
    assert store.upsert([chunk], [[1.0, 0.0]]) == 1
    assert json.loads(path.read_text(encoding="utf-8"))[0]["payload"]["title"] == "Source title"
    restored = VectorStore.__new__(VectorStore)
    restored.client = None
    restored.collection = "test"
    restored._points = []
    restored._load_local()
    assert restored.stats()["total_points"] == 1
    assert restored.enumerate_payloads()[0]["chunk_id"] == "id-1"
