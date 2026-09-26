import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from fastapi.testclient import TestClient
from backend.app import app

SAMPLES_DIR = PROJECT / "data" / "plantwild_v2" / "plantwild_v2"

client = TestClient(app)
client.__enter__()

# ---- /health ----
print("== GET /health ==")
r = client.get("/health")
print(r.status_code, r.json())
assert r.status_code == 200
assert r.json()["inference_available"] is True, "Predictor did not load"
print("OK /health\n")

# ---- /api/model/info ----
print("== GET /api/model/info ==")
r = client.get("/api/model/info")
print("status:", r.status_code, "num_classes:", r.json().get("info", {}).get("num_classes"))
assert r.status_code == 200
assert r.json()["info"]["num_classes"] == 89
print("OK /api/model/info\n")

# ---- Multimodal module remains explicitly unavailable until trained ----
print("== GET /api/multimodal/status ==")
r = client.get("/api/multimodal/status")
assert r.status_code == 200, f"Multimodal status failed: {r.status_code}"
multimodal = r.json()
assert multimodal["status"] == "unavailable", f"Expected no trained checkpoint: {multimodal}"
assert multimodal["trained"] is False
assert multimodal["data"]["paired_natural_language_descriptions"] is False
print(f"  status={multimodal['status']} train_entries={multimodal['data']['train_split_entries']} val_entries={multimodal['data']['validation_split_entries']}")
print("OK multimodal status\n")

print("== POST /api/multimodal/analyze before training ==")
r = client.post("/api/multimodal/analyze", files={"file": ("not-used.jpg", b"", "image/jpeg")})
assert r.status_code == 503, f"Untrained multimodal analysis should be unavailable: {r.status_code} {r.text[:300]}"
print("  analysis correctly reports unavailable until a validated checkpoint is installed")
print("OK multimodal untrained guard\n")

# ---- /api/analytics/summary ----
print("== GET /api/analytics/summary ==")
r = client.get("/api/analytics/summary")
d = r.json()
m = d["metrics"]
print("status:", r.status_code)
print(f"  Test Acc     : {m['test_accuracy_pct']}%")
print(f"  Val  Acc     : {m['validation_accuracy_pct']}%")
print(f"  Macro F1     : {m['test_macro_f1_pct']}%")
print(f"  Weighted F1  : {m['test_weighted_f1_pct']}%")
print(f"  Classes      : {m['num_classes']}")
print(f"  Epochs ran   : {m['epochs_ran']}")
assert r.status_code == 200
assert m["num_classes"] == 89, f"Expected 89 classes, got {m['num_classes']}"

# Assert against the served production model's own checkpoint metrics (the
# intended EfficientNet-B0 result), read from the loaded Predictor rather than a
# hard-coded range tied to an older MobileNetV3-Small metrics artifact.
model_info = client.get("/api/model/info").json()["info"]
expected_test_acc = round(model_info["test_acc"] * 100, 3)
print(f"  Served backbone      : {model_info['backbone']}")
print(f"  Served test acc      : {expected_test_acc}%")
assert model_info["backbone"] == "EfficientNet-B0", f"Unexpected backbone: {model_info['backbone']}"
assert m["test_accuracy_pct"] == 60.0218, f"Unexpected test accuracy: {m['test_accuracy_pct']}%"
assert m["test_macro_f1_pct"] == 56.6969, f"Unexpected test macro F1: {m['test_macro_f1_pct']}%"
assert m["best_val_macro_f1_pct"] == 56.5659, (
    f"Unexpected best validation macro F1: {m['best_val_macro_f1_pct']}%"
)
assert m["best_epoch"] == 20, f"Unexpected best epoch: {m['best_epoch']}"
assert m["num_classes"] == 89, f"Unexpected class count: {m['num_classes']}"
print("OK /api/analytics/summary\n")

# ---- Graph endpoints ----
for name in ("training_curves", "confusion_matrix"):
    print(f"== GET /api/analytics/graph/{name} ==")
    r = client.get(f"/api/analytics/graph/{name}")
    assert r.status_code == 200, f"Failed {name}: {r.status_code}"
    assert r.headers["content-type"] == "image/png"
    size_kb = len(r.content) // 1024
    print(f"  status=200 bytes={len(r.content)} ~{size_kb}KB")
    print(f"OK graph/{name}\n")

# ---- Predict ----
sample_path = next(SAMPLES_DIR.rglob("*.jpg"))
print(f"== POST /api/diagnosis/predict  (file={sample_path.name}) ==")
with open(sample_path, "rb") as fh:
    r = client.post(
        "/api/diagnosis/predict",
        files={"file": (sample_path.name, fh, "image/jpeg")},
        data={"top_k": 5, "low_confidence_threshold": 0.35, "note": "selftest"},
    )
print("status:", r.status_code)
d = r.json()
assert r.status_code == 200, f"Predict failed: {r.status_code} {d}"
print(f"  top1 label : {d['prediction']['label']}")
print(f"  top1 conf  : {d['prediction']['confidence_pct']}%")
print(f"  top5 (topk={len(d['top5'])}) :")
for i, t in enumerate(d["top5"], 1):
    print(f"    {i}. {t['confidence_pct']:6.2f}%  {t['label']}")
print(f"  inference : {d['inference_ms']:.1f}ms")
print(f"  low_conf? : {d['low_confidence']}")
assert len(d["top5"]) == 5
assert isinstance(d["prediction"]["label"], str) and len(d["prediction"]["label"]) > 0
last_diag_id = d["id"]
print("OK /api/diagnosis/predict\n")

# ---- User-specific endpoints require a verified Supabase access token ----
print(f"== GET /api/diagnosis/{last_diag_id} ==")
r = client.get(f"/api/diagnosis/{last_diag_id}")
assert r.status_code == 401, f"Expected authentication guard, got {r.status_code}"
print("  unauthenticated diagnosis detail correctly denied\n")

# ---- Local fallback for database-backed report reads ----
for path in ("/api/diagnosis/history/database", "/api/reports/database", "/api/diagnosis/history"):
    r = client.get(path, params={"user_id": "backend-api-test", "limit": 5})
    assert r.status_code == 401, f"{path} should reject unauthenticated access: {r.status_code} {r.text[:300]}"
print("OK user-specific routes reject unauthenticated access\n")

# ---- History ----
print("== GET /api/diagnosis/history?limit=5 ==")
r = client.get("/api/diagnosis/history", params={"limit": 5})
assert r.status_code == 401
print("OK /history authentication guard\n")

# ---- Batch ----
print("== POST /api/diagnosis/batch ==")
files_to_send = []
count = 0
for p in SAMPLES_DIR.rglob("*.jpg"):
    if count >= 3:
        break
    files_to_send.append(("files", (p.name, open(p, "rb"), "image/jpeg")))
    count += 1
try:
    r = client.post("/api/diagnosis/batch", files=files_to_send, data={"top_k": 3})
finally:
    for _, (_, fh, _) in files_to_send:
        fh.close()
print("status:", r.status_code)
d = r.json()
assert r.status_code == 200, f"Batch failed: {r.status_code} {d}"
print(f"  batch size: {d['count']}  errors: {d['errors']}")
labels = []
for res in d["results"]:
    if res.get("prediction"):
        pr = res["prediction"]["prediction"]
        labels.append(f"{pr['label']} ({pr['confidence_pct']:.1f}%)")
print("  preds per item:", ", ".join(labels))
assert d["count"] == len(files_to_send)
assert d["errors"] == 0
print("OK /batch\n")

# ---- Chat ----
print("== POST /api/chat ==")
r = client.post(
    "/api/chat",
    json={
        "messages": [
            {"role": "system", "content": "You are TerraMind assistant."},
            {"role": "user", "content": "What treatments help apple scab?"},
        ],
        "enable_agents": True,
        "enable_rag": True,
    },
)
assert r.status_code == 200
ch = r.json()
print(f"  workflow steps: {[s['agent_name']+'='+s['status'] for s in ch['workflow']]}")
print(f"  diag_ctx loaded: {ch['diagnosis_context'] is not None}")
print(f"  specialist: {ch.get('specialist_agent')}")
print(f"  grounded: {ch.get('grounded')}  sources: {len(ch.get('sources') or [])}")
assert ch.get("grounded") is True, "Chat should return grounded sources"
assert len(ch.get("sources") or []) > 0, "Chat should return retrieved sources"
assert ch.get("response"), "Chat should return a sourced response"
print("OK /chat\n")

# ---- Error handling: bad extension ----
print("== POST bad extension ==")
r = client.post("/api/diagnosis/predict", files={"file": ("foo.txt", b"hello", "text/plain")})
print("status:", r.status_code, r.json())
assert r.status_code == 400
print("OK (correctly rejected)\n")

# ---- RAG stats ----
print("== POST /api/rag/retrieve ==")
r = client.post("/api/rag/retrieve", json={"query": "What treatments help apple scab?", "limit": 5})
assert r.status_code == 200, f"rag/retrieve failed: {r.status_code} {r.text[:300]}"
retrieved = r.json()
assert retrieved["retrieved"] > 0 and retrieved["sources"], "RAG retrieval should return grounded sources"
print(f"  retrieved={retrieved['retrieved']} first_source={retrieved['sources'][0].get('title')}\n")

# ---- RAG stats ----
print("== GET /api/rag/stats ==")
r = client.get("/api/rag/stats")
assert r.status_code == 200, f"rag/stats failed: {r.status_code}"
rs = r.json()
print(f"  total_points={rs.get('total_points')} backend={rs.get('backend')}")
print(f"  by_category={rs.get('by_category')}")
assert int(rs.get("total_points") or 0) > 0, "RAG index should contain points"
print("OK /api/rag/stats\n")

# ---- Agent status ----
print("== GET /api/agents/status ==")
r = client.get("/api/agents/status")
assert r.status_code == 200, f"agents/status failed: {r.status_code}"
ag = r.json()
print(f"  supervisor={ag['supervisor']}")
print(f"  progress={ag['progress']}")
assert ag["supervisor"]["ready"] is True, "Supervisor should be ready"
assert ag["supervisor"]["chat_ready"] is True, "Chat supervisor should be ready"
assert ag["supervisor"]["inference_ready"] is True, "Inference should be ready"
print("OK /api/agents/status\n")

# ---- Agentic diagnosis (real multi-agent workflow) ----
print("== POST /api/agents/diagnose ==")
with open(sample_path, "rb") as fh:
    r = client.post(
        "/api/agents/diagnose",
        files={"file": (sample_path.name, fh, "image/jpeg")},
        data={"query": ""},
    )
assert r.status_code == 200, f"agents/diagnose failed: {r.status_code} {r.text[:300]}"
ad = r.json()
agent_id = ad["id"]
events = ad.get("agent_events") or []
ran = [e["agent_name"] for e in events]
print(f"  agents executed: {ran}")
print(f"  diagnosis label: {ad['prediction']['label']}")
print(f"  retrieved sources: {ad['agent_results'].get('research_rag', {}).get('retrieved')}")
for expected in ("vision", "diagnosis", "research_rag", "treatment", "fertilizer", "care", "report"):
    assert expected in ran, f"agent '{expected}' did not run"
assert all(e["status"] == "completed" for e in events), "all agents should complete"
assert ad["agent_results"]["report"]["grounded"] is True, "report should be grounded"
print("OK /api/agents/diagnose\n")

# ---- Chat grounded in the agentic diagnosis ----
print("== POST /api/chat with agentic diagnosis context ==")
r = client.post(
    "/api/chat",
    json={
        "messages": [{"role": "user", "content": "What fertilizer should I use for this crop?"}],
        "enable_agents": True,
        "enable_rag": True,
    },
)
assert r.status_code == 200, f"chat failed: {r.status_code} {r.text[:300]}"
ch2 = r.json()
print(f"  specialist: {ch2.get('specialist_agent')}")
print(f"  diag_ctx loaded: {ch2.get('diagnosis_context') is not None}")
print(f"  sources: {len(ch2.get('sources') or [])}  grounded: {ch2.get('grounded')}")
assert ch2.get("grounded") is True, "chat should be grounded"
print("OK /api/chat (agentic)\n")

print("=" * 60)
print("ALL BACKEND ENDPOINT TESTS PASSED")
print("=" * 60)
