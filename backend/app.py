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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from ml.serving.predictor import InferenceResult, Predictor  # noqa: E402
from ml.multimodal.runtime import (  # noqa: E402
    DEFAULT_MULTIMODAL_CHECKPOINT,
    MultimodalRuntime,
    load_runtime_if_available,
    multimodal_status,
)
from backend.repositories import repository  # noqa: E402
from backend.supabase_client import CONFIG as SUPABASE_CONFIG  # noqa: E402
from backend.supabase_client import get_authenticated_user  # noqa: E402
from backend.model_artifacts import resolve_model_checkpoint  # noqa: E402
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
MULTIMODAL_RUNTIME: Optional[MultimodalRuntime] = None
MULTIMODAL_ERROR: Optional[str] = None
MULTIMODAL_CHECKPOINT = Path(os.getenv("TERRAMIND_MULTIMODAL_CHECKPOINT", str(DEFAULT_MULTIMODAL_CHECKPOINT)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global PREDICTOR, SUPERVISOR, CHAT_SUPERVISOR, MULTIMODAL_RUNTIME, MULTIMODAL_ERROR
    t0 = time.perf_counter()
    log.info("Starting TerraMind AI backend — loading inference service...")
    try:
        PREDICTOR = Predictor(model_path=resolve_model_checkpoint(), device="cpu")
        SUPERVISOR = SupervisorAgent(PREDICTOR, rag_pipeline)
        CHAT_SUPERVISOR = ChatSupervisorAgent(rag_pipeline)
        log.info("Predictor ready (%.1fs). Starting API.", time.perf_counter() - t0)
    except Exception as e:
        log.exception("Failed to load predictor: %s", e)
        PREDICTOR = None
    MULTIMODAL_RUNTIME, MULTIMODAL_ERROR = load_runtime_if_available(PREDICTOR, MULTIMODAL_CHECKPOINT)
    if MULTIMODAL_RUNTIME is None:
        log.info("Multimodal module unavailable: %s", MULTIMODAL_ERROR)
    else:
        log.info("Validated multimodal checkpoint loaded (%d latent dimensions).", MULTIMODAL_RUNTIME.model.config.latent_dim)
    if os.getenv("TERRAMIND_RAG_AUTO_INGEST", "false").strip().lower() in {"1", "true", "yes"}:
        try:
            if int(rag_pipeline.stats().get("total_points") or 0) == 0:
                kb_root = PROJECT_ROOT / "data" / "knowledge_base"
                sources = [
                    {"source": str(kb_root / folder), "category": category}
                    for folder, category in (("diseases", "diseases"), ("treatment", "treatment"),
                                             ("fertilizer", "fertilizer"), ("plant_care", "plant care"),
                                             ("agriculture", "agriculture"))
                    if (kb_root / folder).is_dir()
                ]
                if sources:
                    result = rag_pipeline.ingest(sources)
                    log.info("Seeded bundled RAG knowledge base (%s documents, %s chunks).", result["documents"], result["chunks"])
        except Exception as error:
            log.exception("Bundled RAG initialization failed; service will continue with its available fallback: %s", error)
    yield
    log.info("Shutting down TerraMind AI backend.")


app = FastAPI(
    title="TerraMind AI API",
    description="EfficientNet-B0 Plant Disease Diagnosis + Multi-Agent + RAG",
    version="1.0.0",
    lifespan=lifespan,
)

_CORS = os.getenv("TERRAMIND_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[c.strip() for c in _CORS if c.strip()],
    allow_credentials=False,
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


def _current_user(authorization: Optional[str], required: bool = True) -> Optional[Dict]:
    if not authorization:
        if required:
            raise HTTPException(status_code=401, detail="Sign in to access personal data.")
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="A valid Bearer access token is required.")
    try:
        return get_authenticated_user(token)
    except Exception as error:
        log.info("Supabase token validation failed: %s", error)
        raise HTTPException(status_code=401, detail="Invalid or expired access token.") from error


def _confidence_threshold(value: Optional[float]) -> float:
    threshold = value if value is not None else float(os.getenv("TERRAMIND_LOW_CONFIDENCE_THRESHOLD", "0.35"))
    if not 0.05 <= threshold <= 0.95:
        raise HTTPException(status_code=500, detail="TERRAMIND_LOW_CONFIDENCE_THRESHOLD must be between 0.05 and 0.95.")
    return threshold


def _owned_local_report(diagnosis_id: str, user_id: str) -> Dict:
    path = REPORTS / f"{diagnosis_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Diagnosis not found.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise HTTPException(status_code=500, detail="Could not read diagnosis.") from error
    if payload.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Diagnosis not found.")
    return payload


def _owned_diagnosis(diagnosis_id: str, user_id: str) -> Dict:
    try:
        return _owned_local_report(diagnosis_id, user_id)
    except HTTPException as local_error:
        if repository.enabled:
            try:
                row = repository.get_diagnosis(diagnosis_id, user_id)
                predictions = sorted(row.get("predictions", []), key=lambda item: item.get("rank", 999))
                top = [{"class_id": p["class_id"], "label": p["label"], "confidence": p["confidence"], "score": p["confidence"]} for p in predictions]
                return {**row, "prediction": top[0] if top else {}, "predicted_class": top[0] if top else {}, "top5": top}
            except Exception:
                pass
        raise local_error


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


@app.get("/api/multimodal/status", tags=["multimodal"])
def multimodal_module_status() -> Dict:
    status = multimodal_status(MULTIMODAL_CHECKPOINT)
    ready = MULTIMODAL_RUNTIME is not None
    status.update(
        status="ready" if ready else "unavailable",
        trained=ready,
        checkpoint_valid=ready,
        training_completed=ready,
        message=None if ready else MULTIMODAL_ERROR or status["message"],
    )
    if ready:
        status["architecture"] = MULTIMODAL_RUNTIME.model.architecture_info()
    return status


@app.post("/api/multimodal/analyze", tags=["multimodal"])
async def multimodal_analyze(file: UploadFile = File(..., description="Plant image for multimodal analysis")) -> Dict:
    if MULTIMODAL_RUNTIME is None:
        raise HTTPException(
            status_code=503,
            detail=MULTIMODAL_ERROR or "Multimodal model is not trained and validated; analysis is unavailable.",
        )
    predictor = _require_predictor()
    data = _validated_image_bytes(file)
    try:
        diagnosis = predictor.predict(data)
        image = predictor._to_pil(data)
        image_tensor = predictor.transform(image).unsqueeze(0)
        fused = MULTIMODAL_RUNTIME.encode(image_tensor, diagnosis.predicted_class.label)
        try:
            guidance = rag_pipeline.retrieve(
                f"plant disease treatment and care guidance for {diagnosis.predicted_class.label}", limit=5
            )
        except Exception as error:
            log.warning("Multimodal analysis completed but RAG guidance was unavailable: %s", error)
            guidance = {"retrieved": 0, "sources": [], "error": "RAG guidance is unavailable."}
        return {
            "diagnosis": diagnosis.to_dict(),
            "metadata_text": fused["metadata_text"],
            "fused_representation": {"dimension": fused["dimension"], "vector": fused["vector"]},
            "guidance": guidance,
            "production_classifier": "EfficientNet-B0",
            "multimodal_training_state": "completed and validation-gated checkpoint",
        }
    except HTTPException:
        raise
    except Exception as error:
        log.exception("Multimodal analysis failed: %s", error)
        raise HTTPException(status_code=500, detail=f"Multimodal analysis failed: {error}") from error


# ==============================
#  Diagnosis (single & batch)
# ==============================
@app.post("/api/diagnosis/predict", tags=["diagnosis"])
async def predict_single(
    file: UploadFile = File(..., description="Plant leaf or canopy image"),
    low_confidence_threshold: Optional[float] = Form(
        default=None, ge=0.05, le=0.95, description="Confidence below this marks prediction as risky"
    ),
    top_k: int = Form(default=5, ge=1, le=89),
    authorization: Optional[str] = Header(default=None),
    note: Optional[str] = Form(default=None),
):
    user = _current_user(authorization, required=False)
    user_id = user["id"] if user else None
    threshold = _confidence_threshold(low_confidence_threshold)
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
        **_prediction_to_dict(res, threshold),
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
    low_confidence_threshold: Optional[float] = Form(default=None, ge=0.05, le=0.95),
    top_k: int = Form(default=5, ge=1, le=89),
    authorization: Optional[str] = Header(default=None),
):
    user = _current_user(authorization, required=False)
    user_id = user["id"] if user else None
    threshold = _confidence_threshold(low_confidence_threshold)
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
                **_prediction_to_dict(res, threshold),
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
    authorization: Optional[str] = Header(default=None),
):
    user = _current_user(authorization, required=False)
    user_id = user["id"] if user else None
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
def diagnosis_history(limit: int = 50, authorization: Optional[str] = Header(default=None)):
    user = _current_user(authorization)
    limit = max(1, min(limit, 500))
    items = _local_report_items(limit=limit, user_id=user["id"])
    return {"count": len(items), "items": items}


@app.get("/api/diagnosis/history/database", tags=["diagnosis"])
def database_diagnosis_history(limit: int = 50, authorization: Optional[str] = Header(default=None)):
    user = _current_user(authorization)
    user_id = user["id"]
    limit = max(1, min(limit, 500))
    try:
        if repository.enabled:
            items = repository.list_history(user_id, limit)
            return {"count": len(items), "items": items, "source": "supabase"}
    except Exception as e:
        log.warning("Supabase history query failed; using local reports: %s", e)
    items = _local_report_items(limit=limit, user_id=user_id)
    return {"count": len(items), "items": items, "source": "local"}


@app.get("/api/reports/database", tags=["reports"])
def database_reports(limit: int = 50, authorization: Optional[str] = Header(default=None)):
    user = _current_user(authorization)
    user_id = user["id"]
    limit = max(1, min(limit, 500))
    try:
        if repository.enabled:
            items = repository.list_reports(user_id, limit)
            return {"count": len(items), "items": items, "source": "supabase"}
    except Exception as e:
        log.warning("Supabase reports query failed; using local reports: %s", e)
    items = _local_report_items(limit=limit, user_id=user_id)
    return {"count": len(items), "items": items, "source": "local"}


def _local_report_items(limit: int = 50, user_id: str | None = None) -> list[Dict]:
    items = []
    for path in sorted(REPORTS.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if user_id is not None and payload.get("user_id") != user_id:
            continue
        items.append(payload)
        if len(items) >= limit:
            break
    return items


@app.get("/api/diagnosis/{diagnosis_id}", tags=["diagnosis"])
def get_diagnosis(diagnosis_id: str, authorization: Optional[str] = Header(default=None)):
    user = _current_user(authorization)
    try:
        return _owned_diagnosis(diagnosis_id, user["id"])
    except HTTPException as local_error:
        raise local_error


class ExpertReviewRequest(BaseModel):
    diagnosis_id: str
    reviewer_decision: str = Field(..., min_length=2, max_length=100)
    corrected_disease: Optional[str] = Field(default=None, max_length=200)
    notes: Optional[str] = Field(default=None, max_length=5000)


@app.post("/api/reviews", tags=["reviews"])
def create_expert_review(req: ExpertReviewRequest, authorization: Optional[str] = Header(default=None)):
    user = _current_user(authorization)
    diagnosis = _owned_diagnosis(req.diagnosis_id, user["id"])
    if not diagnosis.get("low_confidence"):
        raise HTTPException(status_code=422, detail="Expert review is available for low-confidence diagnoses only.")
    record = {
        "diagnosis_id": req.diagnosis_id,
        "user_id": user["id"],
        "diagnosis": diagnosis.get("prediction", {}).get("label") or diagnosis.get("predicted_class", {}).get("label"),
        "confidence": diagnosis.get("prediction", {}).get("score") or diagnosis.get("predicted_class", {}).get("confidence"),
        "reviewer_decision": req.reviewer_decision,
        "corrected_disease": req.corrected_disease,
        "notes": req.notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        saved = repository.save_review(record)
    except Exception as error:
        log.warning("Expert review database write failed; preserving local review: %s", error)
        saved = record
    review_file = REPORTS / f"{req.diagnosis_id}.reviews.json"
    try:
        current = json.loads(review_file.read_text(encoding="utf-8")) if review_file.exists() else []
        current.append(saved)
        review_file.write_text(json.dumps(current, indent=2), encoding="utf-8")
    except Exception as error:
        log.exception("Could not persist local expert review: %s", error)
        raise HTTPException(status_code=500, detail="Could not save expert review locally.") from error
    return saved


@app.get("/api/reviews/{diagnosis_id}", tags=["reviews"])
def list_expert_reviews(diagnosis_id: str, authorization: Optional[str] = Header(default=None)):
    user = _current_user(authorization)
    _owned_diagnosis(diagnosis_id, user["id"])
    try:
        if repository.enabled:
            return {"items": repository.list_reviews(diagnosis_id, user["id"]), "source": "supabase"}
    except Exception as error:
        log.warning("Could not load reviews from Supabase; using local: %s", error)
    path = REPORTS / f"{diagnosis_id}.reviews.json"
    items = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    return {"items": [item for item in items if item.get("user_id") == user["id"]], "source": "local"}


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
    # The serving model's companion metrics are the source of truth for the
    # currently deployed checkpoint. OUTPUTS/metrics.json may describe an older run.
    predictor_info = _require_predictor().info()
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

    def f3(x, digits=2):
        return round(float(x) * 100, digits) if isinstance(x, (int, float)) else None

    test_acc = f3(predictor_info.get("test_acc"), 4)
    val_acc = f3(_g(metrics, ["best_val_acc"]) or _g(full, ["best_val_acc"]) or _g(metrics, ["val_acc_final_eval"]))
    macro_f1 = f3(predictor_info.get("test_macro_f1"), 4)
    w_f1 = f3(_g(metrics, ["test_weighted_f1"]) or _g(full, ["test_weighted_f1"]))
    overfit = _g(metrics, ["overfitting"]) or _g(full, ["overfitting"]) or {}
    epochs_ran = _g(full, ["epochs_ran"]) or _g(full, ["best_epoch"])

    return {
        "metrics": {
            "test_accuracy_pct": test_acc,
            "validation_accuracy_pct": val_acc,
            "test_macro_f1_pct": macro_f1,
            "test_weighted_f1_pct": w_f1,
            "num_classes": predictor_info.get("num_classes", 89),
            "epochs_ran": epochs_ran,
            "best_epoch": predictor_info.get("best_epoch"),
            "best_val_macro_f1_pct": f3(predictor_info.get("best_val_macro_f1"), 4),
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
async def chat(req: ChatRequest, authorization: Optional[str] = Header(default=None)):
    if CHAT_SUPERVISOR is None:
        raise HTTPException(status_code=503, detail="Chat supervisor is not available.")
    diag_ctx = None
    if req.diagnosis_id:
        user = _current_user(authorization)
        diag_ctx = _owned_local_report(req.diagnosis_id, user["id"])
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
    user = _current_user(authorization, required=False)
    if user:
        chat_event = {
                "user_id": user["id"], "workflow_id": chat_context.workflow_id,
                "query": last_user, "response": chat_context.response,
                "events": chat_context.events,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        try:
            repository.record_chat_event(chat_event)
        except Exception as error:
            log.warning("Chat event persistence unavailable: %s", error)
            local_events = REPORTS / "chat-events"
            try:
                local_events.mkdir(parents=True, exist_ok=True)
                event_file = local_events / f"{user['id']}.json"
                events = json.loads(event_file.read_text(encoding="utf-8")) if event_file.exists() else []
                events.append(chat_event)
                event_file.write_text(json.dumps(events, indent=2), encoding="utf-8")
            except Exception as local_error:
                log.warning("Local chat event fallback unavailable: %s", local_error)
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
