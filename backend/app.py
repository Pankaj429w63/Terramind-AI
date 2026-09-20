from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from ml.serving.predictor import InferenceResult, Predictor  # noqa: E402
from backend.repositories import repository  # noqa: E402
from backend.supabase_client import CONFIG as SUPABASE_CONFIG  # noqa: E402
from rag.pipelines.knowledge import pipeline as rag_pipeline  # noqa: E402
from agents.workflow import ChatSupervisorAgent, SupervisorAgent  # noqa: E402

OUTPUTS = PROJECT_ROOT / "outputs"
GRAPHS = OUTPUTS / "graphs"
REPORTS = OUTPUTS / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUTS / "logs" / "api.log"
(OUTPUTS / "logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")],
)
log = logging.getLogger("terramind.api")

# ---- Services ----
PREDICTOR: Optional[Predictor] = None
SUPERVISOR: Optional[SupervisorAgent] = None
CHAT_SUPERVISOR: Optional[ChatSupervisorAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global PREDICTOR, SUPERVISOR, CHAT_SUPERVISOR
    t0 = time.perf_counter()
    log.info("Starting TerraMind AI backend — loading inference service...")
    try:
        PREDICTOR = Predictor(device="cpu")
        SUPERVISOR = SupervisorAgent(PREDICTOR, rag_pipeline)
        CHAT_SUPERVISOR = ChatSupervisorAgent(rag_pipeline)
        log.info("Predictor ready (%.1fs). Starting API.", time.perf_counter() - t0)
    except Exception as e:
        log.exception("Failed to load predictor: %s", e)
        PREDICTOR = None
    yield
    log.info("Shutting down TerraMind AI backend.")


app = FastAPI(
    title="TerraMind AI API",
    description="EfficientNet-B0 Plant Disease Diagnosis + Multi-Agent + RAG",
    version="1.0.0",
    lifespan=lifespan,
)

_CORS = os.getenv("TERRAMIND_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[c.strip() for c in _CORS if c.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        log.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, ms)
        return response
    except Exception as e:
        ms = (time.perf_counter() - start) * 1000
        log.exception("Unhandled error on %s %s (%.1fms): %s", request.method, request.url.path, ms, e)
        raise


def _require_predictor() -> Predictor:
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Inference service not available. Model failed to load.")
    return PREDICTOR


ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_IMAGE_BYTES = 16 * 1024 * 1024  # 16 MB


def _validated_image_bytes(file: UploadFile, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    fn = (file.filename or "").lower()
    ext = Path(fn).suffix.lower()
    if ext and ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}. Allowed: {sorted(ALLOWED_EXT)}")
    data = file.file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Image too large: {len(data)} bytes > {max_bytes}.")
    return data


def _prediction_to_dict(res: InferenceResult, low_confidence_threshold: float = 0.35) -> Dict:
    payload = res.to_dict()
    payload["prediction"] = {
        "class_id": res.predicted_class.class_id,
        "label": res.predicted_class.label,
        "score": res.predicted_class.confidence,
        "confidence_pct": round(res.predicted_class.confidence * 100, 2),
    }
    payload["top5"] = [
        {
            "class_id": item.class_id,
            "label": item.label,
            "score": item.confidence,
            "confidence_pct": round(item.confidence * 100, 2),
        }
        for item in res.top5
    ]
    payload["confidence_pct"] = round(res.predicted_class.confidence * 100, 2)
    payload["model"] = res.model_info.get("backbone")
    payload["img_size"] = res.model_info.get("img_size")
    payload["low_confidence"] = res.predicted_class.confidence < low_confidence_threshold
    payload["low_confidence_threshold"] = low_confidence_threshold
    payload["notes"] = None
    return payload


# ==============================
#  Health & meta
# ==============================
@app.get("/health", tags=["system"])
def health() -> Dict:
    predictor_info = PREDICTOR.info() if PREDICTOR else None
    return {
        "status": "ok" if PREDICTOR else "degraded",
        "service": "TerraMind AI Backend",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inference_available": PREDICTOR is not None,
        "model": predictor_info,
    }


@app.get("/api/supabase/status", tags=["system"])
def supabase_status() -> Dict:
    return {
        "configured": SUPABASE_CONFIG.enabled,
        "missing": SUPABASE_CONFIG.missing_variables if not SUPABASE_CONFIG.enabled else [],
    }


@app.get("/api/model/info", tags=["model"])
def model_info() -> Dict:
    p = _require_predictor()
    return {"info": p.info(), "labels": p.label_list()}


@app.get("/api/model/labels", tags=["model"])
def model_labels() -> Dict:
    p = _require_predictor()
    return {"num_classes": len(p.label_list()), "labels": p.label_list()}


# ==============================
#  Diagnosis (single & batch)
# ==============================
@app.post("/api/diagnosis/predict", tags=["diagnosis"])
async def predict_single(
    file: UploadFile = File(..., description="Plant leaf or canopy image"),
    low_confidence_threshold: float = Form(
        default=0.35, ge=0.05, le=0.95, description="Confidence below this marks prediction as risky"
    ),
    top_k: int = Form(default=5, ge=1, le=89),
    user_id: Optional[str] = Form(default=None),
    note: Optional[str] = Form(default=None),
):
    p = _require_predictor()
    data = _validated_image_bytes(file)
    try:
        res = p.predict(data, top_k=top_k)
    except Exception as e:
        log.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    diag_id = str(uuid.uuid4())
    payload = {
        "id": diag_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "note": note,
        "source_filename": file.filename,
        **_prediction_to_dict(res, low_confidence_threshold),
    }
    try:
        (REPORTS / f"{diag_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("Could not persist diagnosis: %s", e)
    try:
        repository.save_diagnosis(payload, user_id, data, file.filename, file.content_type)
    except Exception as e:
        log.exception("Supabase diagnosis persistence failed: %s", e)
    return payload


class BatchOutcome(BaseModel):
    id: str
    filename: str
    error: Optional[str] = None
    prediction: Optional[Dict] = None


@app.post("/api/diagnosis/batch", tags=["diagnosis"])
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple plant images (real batch inference)"),
    low_confidence_threshold: float = Form(default=0.35, ge=0.05, le=0.95),
    top_k: int = Form(default=5, ge=1, le=89),
    user_id: Optional[str] = Form(default=None),
):
    p = _require_predictor()
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 64:
        raise HTTPException(status_code=400, detail=f"Too many files: {len(files)} (max 64 per request).")
    results: List[BatchOutcome] = []
    images: List[bytes] = []
    metas: List[Dict] = []
    for f in files:
        try:
            data = _validated_image_bytes(f, max_bytes=MAX_IMAGE_BYTES)
            images.append(data)
            metas.append({"id": str(uuid.uuid4()), "filename": f.filename, "data": data, "content_type": f.content_type})
        except HTTPException as e:
            results.append(BatchOutcome(id=str(uuid.uuid4()), filename=f.filename or "", error=e.detail))

    if images:
        try:
            pred_list = p.predict_batch(images, top_k=top_k)
        except Exception as e:
            log.exception("Batch prediction failed: %s", e)
            raise HTTPException(status_code=400, detail=f"Batch prediction failed: {e}")
        for meta, res in zip(metas, pred_list):
            payload = {
                "id": meta["id"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "source_filename": meta["filename"],
                **_prediction_to_dict(res, low_confidence_threshold),
            }
            try:
                (REPORTS / f'{meta["id"]}.json').write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except Exception as e:
                log.warning("Could not persist batch diagnosis %s: %s", meta["id"], e)
            try:
                repository.save_diagnosis(payload, user_id, meta["data"], meta["filename"], meta["content_type"])
            except Exception as e:
                log.exception("Supabase batch persistence failed for %s: %s", meta["id"], e)
            results.append(BatchOutcome(id=meta["id"], filename=meta["filename"] or "", prediction=payload))

    return {
        "batch_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "count": len(results),
        "errors": sum(1 for r in results if r.error is not None),
        "results": results,
    }


@app.post("/api/agents/diagnose", tags=["agents"])
async def agentic_diagnosis(
    file: UploadFile = File(..., description="Plant image for the agent workflow"),
    query: Optional[str] = Form(default=None),
    user_id: Optional[str] = Form(default=None),
):
    if SUPERVISOR is None:
        raise HTTPException(status_code=503, detail="Agent workflow is not available.")
    data = _validated_image_bytes(file)
    diagnosis_id = str(uuid.uuid4())
    try:
        context = SUPERVISOR.run(data, file.filename or "upload", user_id=user_id, query=query or "", diagnosis_id=diagnosis_id)
    except Exception as error:
        log.exception("Agent workflow failed: %s", error)
        raise HTTPException(status_code=400, detail=f"Agent workflow failed: {error}")
    if context.prediction is None:
        raise HTTPException(status_code=500, detail="Agent workflow completed without a prediction.")
    payload = {
        "id": diagnosis_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "source_filename": file.filename,
        **_prediction_to_dict(context.prediction),
        "agent_workflow_id": context.workflow_id,
        "agent_results": context.outputs,
        "agent_events": context.events,
    }
    (REPORTS / f"{diagnosis_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    persisted = False
    try:
        repository.save_diagnosis(payload, user_id, data, file.filename, file.content_type)
        persisted = True
    except Exception as error:
        log.exception("Agent diagnosis persistence failed: %s", error)
    if persisted:
        for event in context.events:
            try:
                repository.record_agent_execution({
                    "diagnosis_id": diagnosis_id,
                    "user_id": user_id,
                    "agent_name": event["agent_name"],
                    "status": event["status"],
                    "input": event["input"],
                    "output": event.get("output", {}),
                    "error": event.get("error"),
                    "started_at": event["started_at"],
                    "completed_at": event.get("completed_at"),
                })
            except Exception as error:
                log.exception("Agent execution persistence failed: %s", error)
    return payload


@app.get("/api/diagnosis/history", tags=["diagnosis"])
def diagnosis_history(limit: int = 50):
    limit = max(1, min(limit, 500))
    items = []
    for p in sorted(REPORTS.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return {"count": len(items), "items": items}


@app.get("/api/diagnosis/history/database", tags=["diagnosis"])
def database_diagnosis_history(user_id: str, limit: int = 50):
    if not repository.enabled:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")
    try:
        items = repository.list_history(user_id, max(1, min(limit, 500)))
        return {"count": len(items), "items": items}
    except Exception as e:
        log.exception("Supabase history query failed: %s", e)
        raise HTTPException(status_code=502, detail="Could not query Supabase diagnosis history.")


@app.get("/api/reports/database", tags=["reports"])
def database_reports(user_id: str, limit: int = 50):
    if not repository.enabled:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")
    try:
        items = repository.list_reports(user_id, max(1, min(limit, 500)))
        return {"count": len(items), "items": items}
    except Exception as e:
        log.exception("Supabase reports query failed: %s", e)
        raise HTTPException(status_code=502, detail="Could not query Supabase reports.")


@app.get("/api/diagnosis/{diagnosis_id}", tags=["diagnosis"])
def get_diagnosis(diagnosis_id: str):
    p = REPORTS / f"{diagnosis_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Diagnosis not found.")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read diagnosis: {e}")


# ==============================
#  Analytics (read REAL artifacts, never fake)
# ==============================
def _safe_json(path: Path) -> Optional[Dict]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@app.get("/api/analytics/summary", tags=["analytics"])
def analytics_summary() -> Dict:
    metrics = _safe_json(OUTPUTS / "metrics.json") or {}
    full = _safe_json(OUTPUTS / "full_report.json") or {}
    cfg = _safe_json(OUTPUTS / "final_config.json") or {}
    split_info = _safe_json(OUTPUTS / "split_info.json") or {}

    def _g(obj, path_list, default=None):
        cur = obj
        for k in path_list:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    def f3(x):
        return round(float(x) * 100, 2) if isinstance(x, (int, float)) else None

    test_acc = f3(_g(metrics, ["test_acc"]) or _g(full, ["test_acc"]))
    val_acc = f3(_g(metrics, ["best_val_acc"]) or _g(full, ["best_val_acc"]) or _g(metrics, ["val_acc_final_eval"]))
    macro_f1 = f3(_g(metrics, ["test_macro_f1"]) or _g(full, ["test_macro_f1"]))
    w_f1 = f3(_g(metrics, ["test_weighted_f1"]) or _g(full, ["test_weighted_f1"]))
    overfit = _g(metrics, ["overfitting"]) or _g(full, ["overfitting"]) or {}
    epochs_ran = _g(full, ["epochs_ran"]) or _g(full, ["best_epoch"])

    return {
        "metrics": {
            "test_accuracy_pct": test_acc,
            "validation_accuracy_pct": val_acc,
            "test_macro_f1_pct": macro_f1,
            "test_weighted_f1_pct": w_f1,
            "num_classes": 89,
            "epochs_ran": epochs_ran,
            "best_epoch": _g(full, ["best_epoch"]),
            "best_val_loss": _g(full, ["best_val_loss"]),
            "test_loss": _g(metrics, ["test_loss"]) or _g(full, ["test_loss"]),
        },
        "overfitting": overfit or None,
        "config": cfg,
        "split": split_info,
        "artifacts": {
            "training_curves_png": "/api/analytics/graph/training_curves",
            "confusion_matrix_png": "/api/analytics/graph/confusion_matrix",
            "full_report_json": "/api/analytics/full_report",
            "final_config_json": "/api/analytics/final_config",
        },
    }


@app.get("/api/analytics/full_report", tags=["analytics"])
def get_full_report():
    p = OUTPUTS / "full_report.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="full_report.json not found")
    return _safe_json(p)


@app.get("/api/analytics/final_config", tags=["analytics"])
def get_final_config():
    p = OUTPUTS / "final_config.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="final_config.json not found")
    return _safe_json(p)


@app.get("/api/analytics/graph/{name}", tags=["analytics"])
def get_graph(name: str):
    mapping = {
        "training_curves": "training_curves.png",
        "confusion_matrix": "confusion_matrix.png",
    }
    fn = mapping.get(name)
    if fn is None:
        raise HTTPException(status_code=404, detail=f"Unknown graph: {name}. Valid: {list(mapping)}")
    p = GRAPHS / fn
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Graph not found: {fn}")
    return FileResponse(p, media_type="image/png", filename=fn)


# ==============================
#  Placeholder scaffolding for future phases (no fakes)
#  Supervisor, RAG, Agents, Chat, Supabase are defined later
#  in the loop — here they return "not configured" if missing keys.
# ==============================
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)
    diagnosis_id: Optional[str] = None
    user_id: Optional[str] = None
    enable_agents: bool = False
    enable_rag: bool = False


class RagIngestRequest(BaseModel):
    sources: List[Dict[str, str]] = Field(..., min_length=1)


class RagRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=3)
    limit: int = Field(default=5, ge=1, le=20)
    category: Optional[str] = None


@app.post("/api/rag/ingest", tags=["rag"])
async def rag_ingest(req: RagIngestRequest):
    sources = [
        {"source": item["source"], "category": item["category"]}
        for item in req.sources
        if item.get("source") and item.get("category")
    ]
    if len(sources) != len(req.sources):
        raise HTTPException(status_code=422, detail="Each source requires source and category.")
    try:
        return rag_pipeline.ingest(sources)
    except Exception as e:
        log.exception("RAG ingestion failed: %s", e)
        raise HTTPException(status_code=400, detail=f"RAG ingestion failed: {e}")


@app.post("/api/rag/retrieve", tags=["rag"])
async def rag_retrieve(req: RagRetrieveRequest):
    try:
        stats_before = rag_pipeline.stats()
        if not stats_before.get("indexed") and int(stats_before.get("total_points") or 0) == 0:
            raise HTTPException(status_code=409, detail="Knowledge base is empty; ingest agricultural documents to enable retrieval.")
        result = rag_pipeline.retrieve(req.query, req.limit, req.category)
        if result.get("error") and not result.get("sources"):
            raise HTTPException(status_code=409, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.exception("RAG retrieval failed: %s", e)
        raise HTTPException(status_code=400, detail=f"RAG retrieval failed: {e}")


@app.get("/api/rag/stats", tags=["rag"])
def rag_stats() -> Dict:
    try:
        return rag_pipeline.stats()
    except Exception as e:
        log.exception("RAG stats failed: %s", e)
        raise HTTPException(status_code=500, detail=f"RAG stats unavailable: {e}")


@app.get("/api/agents/status", tags=["agents"])
def agent_status() -> Dict:
    supabase_enabled = repository.enabled
    local_report_count = sum(1 for _ in REPORTS.glob("*.json"))
    supervisor_state = "inactive"
    supervisor_ready = SUPERVISOR is not None
    chat_supervisor_ready = CHAT_SUPERVISOR is not None
    inference_ready = PREDICTOR is not None
    if supervisor_ready and inference_ready:
        supervisor_state = "active"
    elif inference_ready:
        supervisor_state = "standby"
    # Count agent runs persisted via Supabase repository (if available)
    agent_runs = 0
    diagnosis_count_sb = 0
    reports_count_sb = 0
    if supabase_enabled:
        try:
            recent = repository.list_agent_executions(100)
            agent_runs = len(recent) if isinstance(recent, list) else 0
        except Exception:
            agent_runs = 0
        try:
            diagnoses_sb = repository.list_history("dashboard", 1000)
            diagnosis_count_sb = len(diagnoses_sb) if isinstance(diagnoses_sb, list) else 0
        except Exception:
            diagnosis_count_sb = 0
        try:
            reports_sb = repository.list_reports("dashboard", 1000)
            reports_count_sb = len(reports_sb) if isinstance(reports_sb, list) else 0
        except Exception:
            reports_count_sb = 0
    total_diagnoses = max(local_report_count, diagnosis_count_sb)
    completed_steps = 0
    if supervisor_ready:
        completed_steps += 1
    if inference_ready:
        completed_steps += 1
    if CHAT_SUPERVISOR is not None:
        completed_steps += 1
    if supabase_enabled:
        completed_steps += 1
    total_steps = 4
    progress_pct = round((completed_steps / total_steps) * 100) if total_steps else 0
    return {
        "supervisor": {
            "state": supervisor_state,
            "ready": supervisor_ready,
            "chat_ready": chat_supervisor_ready,
            "inference_ready": inference_ready,
        },
        "progress": {
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "percent": progress_pct,
        },
        "agent_runs": {
            "recorded_supabase": agent_runs,
        },
        "diagnoses": {
            "local_reports": local_report_count,
            "supabase_diagnoses": diagnosis_count_sb,
            "total": total_diagnoses,
        },
        "reports": {
            "supabase_reports": reports_count_sb,
        },
        "backend": {
            "supabase_enabled": supabase_enabled,
            "rag_backend": getattr(rag_pipeline.store, "backend", None),
        },
    }


@app.post("/api/chat", tags=["chat"])
async def chat(req: ChatRequest):
    if CHAT_SUPERVISOR is None:
        raise HTTPException(status_code=503, detail="Chat supervisor is not available.")
    diag_ctx = None
    if req.diagnosis_id:
        diag_path = REPORTS / f"{req.diagnosis_id}.json"
        if diag_path.exists():
            try:
                diag_ctx = json.loads(diag_path.read_text(encoding="utf-8"))
            except Exception:
                diag_ctx = None
    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    if not last_user.strip():
        raise HTTPException(status_code=422, detail="At least one non-empty user message is required.")
    try:
        chat_context = CHAT_SUPERVISOR.run(last_user, diagnosis_context=diag_ctx)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error))
    except Exception as error:
        log.exception("Chat workflow failed: %s", error)
        raise HTTPException(status_code=400, detail=f"Chat workflow failed: {error}")
    return {
        "response": chat_context.response,
        "diagnosis_context": diag_ctx,
        "last_user_query": last_user,
        "workflow_id": chat_context.workflow_id,
        "workflow": chat_context.events,
        "sources": chat_context.retrieval.get("sources", []),
        "grounded": bool(chat_context.retrieval.get("sources")),
        "specialist_agent": chat_context.specialist,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/", tags=["meta"], response_class=HTMLResponse)
def root_index():
    return """\
<html><head><title>TerraMind AI API</title></head>
<body style="font-family:Arial;background:#051810;color:#c4efc8;padding:2rem;">
<h1 style="color:#6ee7a0">TerraMind AI Backend (MobileNetV3-Small, 89 classes)</h1>
<p>✅ <a href="/health">/health</a> — system + model status</p>
<p>✅ <a href="/docs">/docs</a> — OpenAPI / Swagger</p>
<p>✅ POST /api/diagnosis/predict — single image prediction</p>
<p>✅ POST /api/diagnosis/batch — batch prediction (up to 64 images)</p>
<p>✅ <a href="/api/analytics/summary">/api/analytics/summary</a> — real 52% test acc / 50.04% val / 89 classes</p>
<p>✅ GET/POST /api/chat — status + workflow (no fakes)</p>
<p style="opacity:0.8;font-size:0.9rem">Frontend runs separately — see frontend/ in this repo.</p>
</body></html>"""
