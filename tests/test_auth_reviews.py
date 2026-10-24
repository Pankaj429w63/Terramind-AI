import json
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from backend import app as backend_app
from fastapi import HTTPException


class AuthenticationAndReviewTests(unittest.TestCase):
    def test_bearer_auth_is_required_and_validated(self):
        with self.assertRaises(HTTPException) as missing:
            backend_app._current_user(None)
        self.assertEqual(missing.exception.status_code, 401)

        def verify(token):
            if token == "valid":
                return {"id": "user-1"}
            raise ValueError("invalid")

        with patch.object(backend_app, "get_authenticated_user", verify):
            self.assertEqual(backend_app._current_user("Bearer valid")["id"], "user-1")
            with self.assertRaises(HTTPException) as invalid:
                backend_app._current_user("Bearer invalid")
            self.assertEqual(invalid.exception.status_code, 401)

    def test_expert_review_requires_owned_low_confidence_and_persists_local(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            diagnosis = {"id": "diagnosis-1", "user_id": "owner", "low_confidence": True,
                         "prediction": {"label": "Tomato___Early_blight", "score": 0.42}}
            (path / "diagnosis-1.json").write_text(json.dumps(diagnosis), encoding="utf-8")
            with patch.object(backend_app, "REPORTS", path), \
                 patch.object(backend_app, "get_authenticated_user", lambda token: {"id": "owner"}), \
                 patch.object(backend_app.repository, "client", None):
                saved = backend_app.create_expert_review(
                    backend_app.ExpertReviewRequest(diagnosis_id="diagnosis-1", reviewer_decision="incorrect",
                                                    corrected_disease="Septoria leaf spot", notes="Field review"),
                    "Bearer valid",
                )
                self.assertEqual(saved["diagnosis"], "Tomato___Early_blight")
                self.assertEqual(saved["confidence"], 0.42)
                self.assertEqual(saved["reviewer_decision"], "incorrect")
                self.assertTrue(saved["created_at"])
                stored = json.loads((path / "diagnosis-1.reviews.json").read_text(encoding="utf-8"))
                self.assertEqual(stored[0]["corrected_disease"], "Septoria leaf spot")

    def test_review_rejects_other_users_and_high_confidence_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "other.json").write_text(json.dumps({"user_id": "someone-else", "low_confidence": True}), encoding="utf-8")
            (path / "certain.json").write_text(json.dumps({"user_id": "owner", "low_confidence": False}), encoding="utf-8")
            with patch.object(backend_app, "REPORTS", path), \
                 patch.object(backend_app, "get_authenticated_user", lambda token: {"id": "owner"}), \
                 patch.object(backend_app.repository, "client", None):
                req = backend_app.ExpertReviewRequest(diagnosis_id="other", reviewer_decision="confirmed")
                with self.assertRaises(HTTPException) as unauthorized:
                    backend_app.create_expert_review(req, "Bearer valid")
                self.assertEqual(unauthorized.exception.status_code, 404)
                req = backend_app.ExpertReviewRequest(diagnosis_id="certain", reviewer_decision="confirmed")
                with self.assertRaises(HTTPException) as confident:
                    backend_app.create_expert_review(req, "Bearer valid")
                self.assertEqual(confident.exception.status_code, 422)

    def test_review_retrieval_uses_owned_local_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            diagnosis = {"id": "diagnosis-2", "user_id": "owner", "low_confidence": True}
            (path / "diagnosis-2.json").write_text(json.dumps(diagnosis), encoding="utf-8")
            (path / "diagnosis-2.reviews.json").write_text(json.dumps([{
                "diagnosis_id": "diagnosis-2", "user_id": "owner", "reviewer_decision": "incorrect",
            }]), encoding="utf-8")
            with patch.object(backend_app, "REPORTS", path), \
                 patch.object(backend_app, "get_authenticated_user", lambda token: {"id": "owner"}), \
                 patch.object(backend_app.repository, "client", None):
                result = backend_app.list_expert_reviews("diagnosis-2", "Bearer valid")
                self.assertEqual(result["source"], "local")
                self.assertEqual(result["items"][0]["reviewer_decision"], "incorrect")

    def test_chat_context_uses_ownership_checked_database_fallback(self):
        captured = {}
        class FakeChatSupervisor:
            def run(self, query, diagnosis_context=None, enable_specialist=True, enable_rag=True):
                captured["diagnosis_context"] = diagnosis_context
                return type("ChatResult", (), {
                    "response": "contextual", "retrieval": {"sources": []}, "specialist": "",
                    "generation": {"mode": "no_evidence", "status": "no_retrieval_results"},
                    "workflow_id": "workflow", "events": [],
                })()

        remote_diagnosis = {"id": "remote-1", "user_id": "owner", "low_confidence": True,
                            "prediction": {"label": "apple scab"}}
        with patch.object(backend_app, "CHAT_SUPERVISOR", FakeChatSupervisor()), \
             patch.object(backend_app, "_current_user", lambda *args, **kwargs: {"id": "owner"}), \
             patch.object(backend_app, "_owned_diagnosis", lambda diagnosis_id, user_id: remote_diagnosis), \
             patch.object(backend_app.repository, "record_chat_event", lambda record: None):
            response = asyncio.run(backend_app.chat(
                backend_app.ChatRequest(
                    messages=[backend_app.ChatMessage(role="user", content="What now?")],
                    diagnosis_id="remote-1", enable_agents=False, enable_rag=False,
                ),
                "Bearer valid",
            ))
        self.assertIs(captured["diagnosis_context"], remote_diagnosis)
        self.assertEqual(response["diagnosis_context"], remote_diagnosis)

    def test_history_uses_user_scoped_local_fallback_when_database_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "mine.json").write_text(json.dumps({"id": "mine", "user_id": "owner"}), encoding="utf-8")
            (path / "theirs.json").write_text(json.dumps({"id": "theirs", "user_id": "other"}), encoding="utf-8")
            with patch.object(backend_app, "REPORTS", path), \
                 patch.object(backend_app, "_current_user", lambda authorization, **kwargs: {"id": "owner"}), \
                 patch.object(type(backend_app.repository), "enabled", new_callable=PropertyMock, return_value=True), \
                 patch.object(backend_app.repository, "list_history", side_effect=RuntimeError("database unavailable")):
                result = backend_app.database_diagnosis_history(10, "Bearer valid")
            self.assertEqual(result["source"], "local")
            self.assertEqual([item["id"] for item in result["items"]], ["mine"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
