from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from backend.repositories import repository
from ml.serving.predictor import InferenceResult, Predictor
from rag.pipelines.knowledge import KnowledgePipeline


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WorkflowContext:
    workflow_id: str
    image_bytes: bytes
    filename: str
    user_id: str | None = None
    diagnosis_id: str | None = None
    query: str = ""
    prediction: InferenceResult | None = None
    retrieval: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    persist_events: bool = False


class Agent:
    name = "agent"

    def run(self, context: WorkflowContext) -> dict[str, Any]:
        started = time.perf_counter()
        event: dict[str, Any] = {
            "agent_name": self.name,
            "status": "running",
            "started_at": now(),
            "input": {"workflow_id": context.workflow_id},
        }
        context.events.append(event)
        try:
            output = self.execute(context)
            event.update({"status": "completed", "output": output, "completed_at": now()})
            self._persist(context, event)
            return output
        except Exception as error:
            event.update({"status": "failed", "error": str(error), "completed_at": now()})
            self._persist(context, event)
            raise
        finally:
            event["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)

    def execute(self, context: WorkflowContext) -> dict[str, Any]:
        raise NotImplementedError

    def _persist(self, context: WorkflowContext, event: dict[str, Any]) -> None:
        if not context.persist_events:
            return
        repository.record_agent_execution({
            "diagnosis_id": context.diagnosis_id,
            "user_id": context.user_id,
            "agent_name": event["agent_name"],
            "status": event["status"],
            "input": event["input"],
            "output": event.get("output", {}),
            "error": event.get("error"),
            "started_at": event["started_at"],
            "completed_at": event.get("completed_at"),
        })


class VisionAgent(Agent):
    name = "vision"

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def execute(self, context: WorkflowContext) -> dict[str, Any]:
        context.prediction = self.predictor.predict(context.image_bytes, top_k=5)
        return context.prediction.to_dict()


class DiagnosisAgent(Agent):
    name = "diagnosis"

    def execute(self, context: WorkflowContext) -> dict[str, Any]:
        if context.prediction is None:
            raise RuntimeError("Vision agent must complete before diagnosis.")
        result = {
            "label": context.prediction.predicted_class.label,
            "confidence": context.prediction.predicted_class.confidence,
            "top5": [item.__dict__ for item in context.prediction.top5],
        }
        context.outputs[self.name] = result
        return result


class ResearchRagAgent(Agent):
    name = "research_rag"

    def __init__(self, pipeline: KnowledgePipeline) -> None:
        self.pipeline = pipeline

    def execute(self, context: WorkflowContext) -> dict[str, Any]:
        if context.prediction is None:
            raise RuntimeError("Diagnosis is required before research.")
        query = context.query or f"{context.prediction.predicted_class.label} treatment fertilizer plant care"
        context.retrieval = self.pipeline.retrieve(query, limit=5)
        result = {
            "query": query,
            "retrieved": context.retrieval["retrieved"],
            "sources": context.retrieval["sources"],
        }
        context.outputs[self.name] = result
        return result


class SourceRecommendationAgent(Agent):
    focus: str = ""

    def execute(self, context: WorkflowContext) -> dict[str, Any]:
        sources = context.retrieval.get("sources", [])
        result = {
            "focus": self.focus,
            "available": bool(sources),
            "sources": [
                {"title": item.get("title"), "source": item.get("source"), "excerpt": item.get("text", "")[:500]}
                for item in sources
            ],
        }
        context.outputs[self.name] = result
        return result


class TreatmentAgent(SourceRecommendationAgent):
    name = "treatment"
    focus = "treatment"


class FertilizerAgent(SourceRecommendationAgent):
    name = "fertilizer"
    focus = "fertilizer"


class CareAgent(SourceRecommendationAgent):
    name = "care"
    focus = "plant care"


class ReportAgent(Agent):
    name = "report"

    def execute(self, context: WorkflowContext) -> dict[str, Any]:
        if context.prediction is None:
            raise RuntimeError("Diagnosis is required before report generation.")
        report = {
            "workflow_id": context.workflow_id,
            "diagnosis_id": context.diagnosis_id,
            "generated_at": now(),
            "diagnosis": context.outputs.get("diagnosis"),
            "research": context.outputs.get("research_rag"),
            "treatment": context.outputs.get("treatment"),
            "fertilizer": context.outputs.get("fertilizer"),
            "care": context.outputs.get("care"),
            "grounded": bool(context.retrieval.get("sources")),
        }
        context.outputs[self.name] = report
        return report


class SupervisorAgent:
    name = "supervisor"

    def __init__(self, predictor: Predictor, pipeline: KnowledgePipeline) -> None:
        self.agents: list[Agent] = [
            VisionAgent(predictor),
            DiagnosisAgent(),
            ResearchRagAgent(pipeline),
            TreatmentAgent(),
            FertilizerAgent(),
            CareAgent(),
            ReportAgent(),
        ]

    def run(
        self,
        image_bytes: bytes,
        filename: str,
        user_id: str | None = None,
        query: str = "",
        diagnosis_id: str | None = None,
    ) -> WorkflowContext:
        if not image_bytes:
            raise ValueError("Workflow requires non-empty image bytes.")
        context = WorkflowContext(str(uuid.uuid4()), image_bytes, filename, user_id=user_id, query=query, diagnosis_id=diagnosis_id)
        for agent in self.agents:
            agent.run(context)
        return context


@dataclass
class ChatWorkflowContext:
    workflow_id: str
    query: str
    diagnosis_context: dict[str, Any] | None = None
    retrieval: dict[str, Any] = field(default_factory=dict)
    specialist: str = ""
    response: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


class ChatSupervisorAgent:
    """Orchestrates source-grounded chat without inventing an LLM response."""

    def __init__(self, pipeline: KnowledgePipeline) -> None:
        self.pipeline = pipeline

    @staticmethod
    def _specialist(query: str) -> tuple[str, str]:
        text = query.lower()
        if any(word in text for word in ("fertilizer", "nutrient", "nitrogen", "npk")):
            return "fertilizer", "Fertilizer Agent"
        if any(word in text for word in ("treat", "medicine", "fungicide", "pesticide", "control")):
            return "treatment", "Treatment Agent"
        if any(word in text for word in ("care", "water", "watering", "prune", "sunlight", "soil")):
            return "plant care", "Care Agent"
        if any(word in text for word in ("disease", "symptom", "spot", "blight", "infect")):
            return "diseases", "Diagnosis Agent"
        return "agriculture", "Research Agent"

    @staticmethod
    def _event(workflow_id: str, name: str, status: str, input_data: dict[str, Any], output: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
        event = {
            "agent_name": name,
            "status": status,
            "workflow_id": workflow_id,
            "started_at": now(),
            "input": input_data,
        }
        if output is not None:
            event["output"] = output
        if error is not None:
            event["error"] = error
        event["completed_at"] = now()
        return event

    def run(self, query: str, diagnosis_context: dict[str, Any] | None = None) -> ChatWorkflowContext:
        if not query.strip():
            raise ValueError("Chat requires a non-empty user question.")
        context = ChatWorkflowContext(str(uuid.uuid4()), query.strip(), diagnosis_context)
        context.events.append(self._event(context.workflow_id, "supervisor", "completed", {"query": context.query}, {"route": "research_rag"}))
        specialist_key, specialist_name = self._specialist(context.query)
        context.specialist = specialist_name
        retrieval_query = context.query
        if diagnosis_context:
            predicted = diagnosis_context.get("predicted_class", {}).get("label")
            if predicted:
                retrieval_query = f"{predicted} {context.query}"
        try:
            context.retrieval = self.pipeline.retrieve(retrieval_query, limit=5)
            context.events.append(self._event(
                context.workflow_id,
                "research_rag",
                "completed",
                {"query": retrieval_query},
                {"retrieved": context.retrieval["retrieved"], "source_count": len(context.retrieval["sources"])},
            ))
            sources = context.retrieval.get("sources", [])
            focused = [item for item in sources if specialist_key in str(item.get("category", "")).lower()]
            selected = focused or sources
            context.events.append(self._event(
                context.workflow_id,
                specialist_name,
                "completed",
                {"focus": specialist_key, "source_count": len(selected)},
                {"sources": [{"title": item.get("title"), "source": item.get("source")} for item in selected]},
            ))
            if selected:
                excerpts = "\n\n".join(
                    f"[{item.get('title', 'Source')}] {item.get('text', '')[:700]}"
                    for item in selected
                )
                context.response = f"Based on the retrieved TerraMind knowledge sources:\n\n{excerpts}"
            else:
                context.response = None
            context.events.append(self._event(
                context.workflow_id,
                "response",
                "completed",
                {"source_count": len(selected)},
                {"grounded": bool(selected)},
            ))
            return context
        except Exception as error:
            context.events.append(self._event(context.workflow_id, "research_rag", "failed", {"query": retrieval_query}, error=str(error)))
            raise
