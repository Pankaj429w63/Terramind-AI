"""Unit tests for the dedicated independent plant-observation review tool."""
from __future__ import annotations

import csv
import datetime
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from ml.annotation.independent_review_tool import (
    check_review_context,
    create_independent_review_app,
    load_independent_review_context,
)
from ml.fusion.annotation import (
    REVIEW_COLUMNS,
    alias_for,
    build_alias_map,
    load_review_ledger,
    require_independent_review,
    require_reviewed,
    write_alias_map,
    write_workbook,
)


class IndependentReviewToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.images_dir = self.root / "images"

        # Create two sample classes and images
        class1 = self.images_dir / "apple scab"
        class1.mkdir(parents=True)
        Image.new("RGB", (16, 16), (20, 120, 40)).save(class1 / "1.jpg")

        class2 = self.images_dir / "banana leaf"
        class2.mkdir(parents=True)
        Image.new("RGB", (16, 16), (40, 160, 60)).save(class2 / "1.jpg")

        self.sample_ids = ["apple scab/1.jpg", "banana leaf/1.jpg"]
        self.salt = "review-test-salt-1234"
        self.alias_map = build_alias_map(self.sample_ids, self.salt)
        self.alias_map_path = self.root / "alias_map.json"
        write_alias_map(self.alias_map, self.alias_map_path, self.salt)

        # Map back to sample IDs
        sample_to_alias = {s: a for a, s in self.alias_map.items()}

        # Create workbook rows with distinct observers
        self.workbook_rows = [
            {
                "alias": sample_to_alias["apple scab/1.jpg"],
                "observation": "Green leaf with small dark olive-green spots and serrated margins.",
                "observer_id": "observer_alice",
                "observed_at": "2026-10-18",
            },
            {
                "alias": sample_to_alias["banana leaf/1.jpg"],
                "observation": "Elongated broad leaf with smooth parallel veins and slight edge tearing.",
                "observer_id": "observer_bob",
                "observed_at": "2026-10-19",
            },
        ]
        self.workbook_path = self.root / "annotator_workbook.csv"
        write_workbook(self.workbook_rows, self.workbook_path)

        # Create initial review ledger with pending status
        self.ledger_path = self.root / "review_ledger.csv"
        with self.ledger_path.open("w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=list(REVIEW_COLUMNS), lineterminator="\n")
            writer.writeheader()
            for s in sorted(self.sample_ids):
                writer.writerow({
                    "sample_id": s,
                    "review_status": "pending",
                    "reviewer": "",
                    "reviewed_at": "",
                    "notes": "",
                })

        self.context = load_independent_review_context(
            alias_map_path=self.alias_map_path,
            workbook_path=self.workbook_path,
            ledger_path=self.ledger_path,
            images_dir=self.images_dir,
        )
        self.app = create_independent_review_app(self.context)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_check_review_context_ready(self) -> None:
        report = check_review_context(self.context)
        self.assertEqual(report["decision"], "READY")
        self.assertEqual(report["total_items"], 2)
        self.assertEqual(report["pending_items"], 2)
        self.assertEqual(report["reviewed_items"], 0)
        self.assertEqual(report["unresolved_images"], 0)
        self.assertEqual(report["missing_observations"], 0)

    def test_get_progress_and_item(self) -> None:
        res = self.client.get("/api/progress")
        self.assertEqual(res.status_code, 200)
        prog = res.json()
        self.assertEqual(prog["total"], 2)
        self.assertEqual(prog["pending"], 2)
        self.assertEqual(prog["approved"], 0)

        item_res = self.client.get("/api/item/1")
        self.assertEqual(item_res.status_code, 200)
        item = item_res.json()
        self.assertEqual(item["index"], 1)
        self.assertIn(item["sample_id"], self.sample_ids)
        self.assertEqual(item["review_status"], "pending")
        self.assertTrue(len(item["observation"]) > 0)

    def test_self_review_prevention(self) -> None:
        # Apple scab sample was observed by 'observer_alice'
        item = self.context.items_by_sample_id["apple scab/1.jpg"]
        self.assertEqual(item.observer_id, "observer_alice")

        # Attempt to review by 'observer_alice'
        payload = {
            "sample_id": "apple scab/1.jpg",
            "review_status": "approved",
            "reviewer": "observer_alice",
            "notes": "Looks good to me",
        }
        res = self.client.post("/api/decision", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Self-review prohibited", res.json()["error"])

        # Check ledger on disk was NOT modified
        disk_ledger = load_review_ledger(self.ledger_path)
        self.assertEqual(disk_ledger["apple scab/1.jpg"]["review_status"], "pending")

    def test_case_insensitive_self_review_prevention(self) -> None:
        payload = {
            "sample_id": "apple scab/1.jpg",
            "review_status": "approved",
            "reviewer": "OBSERVER_ALICE",
            "notes": "Attempting case-insensitive self review",
        }
        res = self.client.post("/api/decision", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Self-review prohibited", res.json()["error"])

    def test_independent_reviewer_approval(self) -> None:
        payload = {
            "sample_id": "apple scab/1.jpg",
            "review_status": "approved",
            "reviewer": "dr_independent_reviewer",
            "notes": "Factual foliar description verified against photo; no diagnostic leakage.",
        }
        res = self.client.post("/api/decision", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["review_status"], "approved")
        self.assertEqual(data["reviewer"], "dr_independent_reviewer")
        self.assertEqual(data["reviewed_at"], datetime.date.today().isoformat())

        # Check in-memory item updated
        item = self.context.items_by_sample_id["apple scab/1.jpg"]
        self.assertEqual(item.review_status, "approved")
        self.assertEqual(item.reviewer, "dr_independent_reviewer")

        # Check ledger file on disk updated atomically
        disk_ledger = load_review_ledger(self.ledger_path)
        entry = disk_ledger["apple scab/1.jpg"]
        self.assertEqual(entry["review_status"], "approved")
        self.assertEqual(entry["reviewer"], "dr_independent_reviewer")
        self.assertEqual(entry["reviewed_at"], datetime.date.today().isoformat())
        self.assertIn("Factual foliar description", entry["notes"])

    def test_rejection_requires_notes(self) -> None:
        # Rejection without notes should fail
        payload = {
            "sample_id": "apple scab/1.jpg",
            "review_status": "rejected",
            "reviewer": "dr_independent_reviewer",
            "notes": "",
        }
        res = self.client.post("/api/decision", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("requires explanatory notes", res.json()["error"])

        # Rejection with notes succeeds
        payload["notes"] = "Mismatched leaf shape; description omits significant lesions."
        res2 = self.client.post("/api/decision", json=payload)
        self.assertEqual(res2.status_code, 200)

        disk_ledger = load_review_ledger(self.ledger_path)
        self.assertEqual(disk_ledger["apple scab/1.jpg"]["review_status"], "rejected")

    def test_dispute_requires_notes(self) -> None:
        payload = {
            "sample_id": "banana leaf/1.jpg",
            "review_status": "disputed",
            "reviewer": "dr_independent_reviewer",
            "notes": "",
        }
        res = self.client.post("/api/decision", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("requires explanatory notes", res.json()["error"])

        payload["notes"] = "Ambiguous resolution makes vein structure unverified."
        res2 = self.client.post("/api/decision", json=payload)
        self.assertEqual(res2.status_code, 200)

        disk_ledger = load_review_ledger(self.ledger_path)
        self.assertEqual(disk_ledger["banana leaf/1.jpg"]["review_status"], "disputed")

    def test_empty_reviewer_rejected(self) -> None:
        payload = {
            "sample_id": "apple scab/1.jpg",
            "review_status": "approved",
            "reviewer": "   ",
            "notes": "Some notes",
        }
        res = self.client.post("/api/decision", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Reviewer identity cannot be empty", res.json()["error"])

    def test_invalid_status_rejected(self) -> None:
        payload = {
            "sample_id": "apple scab/1.jpg",
            "review_status": "auto_approved",
            "reviewer": "dr_independent_reviewer",
            "notes": "Invalid status test",
        }
        res = self.client.post("/api/decision", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid review decision", res.json()["error"])

    def test_image_serving(self) -> None:
        res = self.client.get("/image/apple scab/1.jpg")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "image/jpeg")
        self.assertTrue(len(res.content) > 0)

        # Path traversal guard
        res_bad = self.client.get("/image/../escaped.jpg")
        self.assertEqual(res_bad.status_code, 404)

    def test_validation_gate_integration(self) -> None:
        # Review both items independently
        self.client.post("/api/decision", json={
            "sample_id": "apple scab/1.jpg",
            "review_status": "approved",
            "reviewer": "independent_auditor_01",
            "notes": "Verified",
        })
        self.client.post("/api/decision", json={
            "sample_id": "banana leaf/1.jpg",
            "review_status": "approved",
            "reviewer": "independent_auditor_01",
            "notes": "Verified",
        })

        # Reload ledger from disk and test compatibility with ml.fusion.annotation gates
        disk_ledger = load_review_ledger(self.ledger_path)
        collated = [
            {"sample_id": "apple scab/1.jpg"},
            {"sample_id": "banana leaf/1.jpg"},
        ]
        reviewed_gate = require_reviewed(collated, disk_ledger)
        self.assertEqual(reviewed_gate, {"collated": 2, "approved": 2})

        indep_gate = require_independent_review(self.workbook_rows, self.alias_map, disk_ledger)
        self.assertEqual(indep_gate, {"independent_reviews": 2})


if __name__ == "__main__":
    unittest.main()
