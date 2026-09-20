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
assert 48 <= (m["test_accuracy_pct"] or 0) <= 56, "Test acc should be 52%"
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

# ---- Retrieve the same diagnosis ----
print(f"== GET /api/diagnosis/{last_diag_id} ==")
r = client.get(f"/api/diagnosis/{last_diag_id}")
assert r.status_code == 200
ret = r.json()
assert ret["id"] == last_diag_id
print(f"  same id: {ret['id']}  label: {ret['prediction']['label']}  -> OK\n")

# ---- History ----
print("== GET /api/diagnosis/history?limit=5 ==")
r = client.get("/api/diagnosis/history", params={"limit": 5})
assert r.status_code == 200
d = r.json()
print(f"  count: {d['count']}  items[0] id: {d['items'][0]['id'] if d['items'] else None}")
print("OK /history\n")

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
        "diagnosis_id": last_diag_id,
        "enable_agents": True,
        "enable_rag": True,
    },
)
assert r.status_code == 200
ch = r.json()
print(f"  workflow steps: {[s['step']+'='+s['status'] for s in ch['workflow']]}")
print(f"  diag_ctx loaded: {ch['diagnosis_context'] is not None}")
print("OK /chat\n")

# ---- Error handling: bad extension ----
print("== POST bad extension ==")
r = client.post("/api/diagnosis/predict", files={"file": ("foo.txt", b"hello", "text/plain")})
print("status:", r.status_code, r.json())
assert r.status_code == 400
print("OK (correctly rejected)\n")

print("=" * 60)
print("ALL BACKEND ENDPOINT TESTS PASSED")
print("=" * 60)
