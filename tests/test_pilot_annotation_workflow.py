"""Unit and integration tests for the class-balanced pilot annotation and review workflow."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from ml.annotation.review_tool import check_context, create_app, load_review_context
from ml.fusion.annotation import (
    BLINDING_FORBIDDEN_COLUMNS,
    WORKBOOK_COLUMNS,
    alias_for,
    build_alias_map,
    collate_observations,
    load_alias_map,
    load_annotator_workbook,
    load_review_ledger,
    new_salt,
    observation_issues,
    require_independent_review,
    require_instrument_version,
    require_reviewed,
    write_alias_map,
    write_observations_csv,
    write_workbook,
)
from scripts.build_annotation_workbook import select_balanced_pilot_samples

ROOT = Path(__file__).resolve().parents[1]
ELIGIBILITY_MANIFEST_PATH = ROOT / "outputs" / "research" / "fusion_eligibility_manifest.json"


class PilotAnnotationWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.tmp.name)
        self.images_dir = self.test_dir / "images"
        self.images_dir.mkdir(parents=True)

        self.labels = {0: "apple scab", 1: "banana panama disease"}
        for class_name in self.labels.values():
            d = self.images_dir / class_name
            d.mkdir(parents=True)
            Image.new("RGB", (16, 16), (40, 140, 50)).save(d / "sample_1.jpg")
            Image.new("RGB", (16, 16), (40, 140, 50)).save(d / "sample_2.jpg")

        self.sample_ids = [
            "apple scab/sample_1.jpg",
            "apple scab/sample_2.jpg",
            "banana panama disease/sample_1.jpg",
            "banana panama disease/sample_2.jpg",
        ]
        self.salt = "pilot-test-salt-12345"
        self.alias_map = build_alias_map(self.sample_ids, self.salt)
        self.alias_map_path = self.test_dir / "sealed" / "alias_map.json"
        write_alias_map(self.alias_map, self.alias_map_path, self.salt)

        self.workbook_path = self.test_dir / "annotator_workbook.csv"
        write_workbook(
            [{"alias": a, "observation": "", "observer_id": "", "observed_at": ""} for a in sorted(self.alias_map)],
            self.workbook_path,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_pilot_sample_selection_from_manifest(self) -> None:
        """Verify the selection function enforces balance, count, and test-split exclusion."""
        if not ELIGIBILITY_MANIFEST_PATH.is_file():
            self.skipTest("Eligibility manifest not found")
        manifest = json.loads(ELIGIBILITY_MANIFEST_PATH.read_text(encoding="utf-8"))
        selected = select_balanced_pilot_samples(manifest, seed=42, train_per_class=1, val_per_class=1)

        self.assertEqual(len(selected), 178, "Must select exactly 178 samples (2 per class for 89 classes)")
        classes = {s.split("/")[0] for s in selected}
        self.assertEqual(len(classes), 89, "All 89 classes must be represented")

        # Verify test split exclusion
        test_samples = set(manifest["eligible_by_split"].get("test", []))
        self.assertFalse(any(s in test_samples for s in selected), "Pilot must not contain test split samples")

        # Verify reproducibility
        selected_repeat = select_balanced_pilot_samples(manifest, seed=42, train_per_class=1, val_per_class=1)
        self.assertEqual(selected, selected_repeat, "Selection must be fully deterministic with seed 42")

    def test_workbook_strict_blinding(self) -> None:
        """Verify workbook exposes only opaque aliases and no forbidden target columns."""
        with self.workbook_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])

        self.assertEqual(fieldnames, list(WORKBOOK_COLUMNS))
        exposed = BLINDING_FORBIDDEN_COLUMNS.intersection({name.casefold() for name in fieldnames})
        self.assertFalse(exposed, f"Forbidden columns exposed: {exposed}")

    def test_review_tool_context_and_resolution(self) -> None:
        """Verify review context resolves every alias to disk without revealing paths to client."""
        context = load_review_context(
            self.alias_map_path, self.workbook_path, self.images_dir, labels=self.labels
        )
        report = check_context(context)
        self.assertEqual(report["decision"], "READY")
        self.assertEqual(report["resolvable_images"], 4)
        self.assertEqual(report["unresolved_aliases"], 0)
        self.assertEqual(report["filled_rows"], 0)
        self.assertEqual(report["missing_rows"], 4)

    def test_review_tool_label_leak_rejection(self) -> None:
        """Verify the review tool rejects descriptions containing exact class/disease names."""
        context = load_review_context(
            self.alias_map_path, self.workbook_path, self.images_dir, labels=self.labels
        )
        app = create_app(context)
        client = TestClient(app)

        # Set observer ID
        res = client.post("/api/observer", json={"observer_id": "observer-01"})
        self.assertEqual(res.status_code, 200)

        alias = sorted(self.alias_map)[0]

        # Submit observation with label leak
        leak_text = "leaf showing clear signs of apple scab infection"
        res = client.post("/api/observation", json={"alias": alias, "observation": leak_text})
        self.assertEqual(res.status_code, 400)
        self.assertIn("target-derived", res.json()["error"].lower())

    def test_review_tool_valid_save_and_overwrite_protection(self) -> None:
        """Verify valid clinical symptom description saves cleanly and blocks overwrite."""
        context = load_review_context(
            self.alias_map_path, self.workbook_path, self.images_dir, labels=self.labels
        )
        app = create_app(context)
        client = TestClient(app)

        client.post("/api/observer", json={"observer_id": "observer-01"})
        alias = sorted(self.alias_map)[0]

        valid_text = "circular dark brown velvety lesion on upper leaf blade with pale perimeter"
        res = client.post("/api/observation", json={"alias": alias, "observation": valid_text})
        self.assertEqual(res.status_code, 200)

        # Verify saved in workbook
        workbook_rows = load_annotator_workbook(self.workbook_path)
        matching = [r for r in workbook_rows if r["alias"] == alias]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["observation"], valid_text)
        self.assertEqual(matching[0]["observer_id"], "observer-01")

        # Attempt to overwrite must be rejected
        res = client.post("/api/observation", json={"alias": alias, "observation": "different observation"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("already has an observation", res.json()["error"].lower())

    def test_review_ledger_independent_review_gate(self) -> None:
        """Verify that an observation cannot be approved by its own annotator."""
        alias = sorted(self.alias_map)[0]
        sample_id = self.alias_map[alias]

        workbook_rows = [
            {"alias": alias, "observation": "pale chlorotic stippling along leaf margins", "observer_id": "dr_smith", "observed_at": "2026-10-18"}
        ]

        # Case 1: Self-approval (reviewer == observer_id) -> MUST FAIL
        self_review_ledger = {
            sample_id: {
                "sample_id": sample_id,
                "review_status": "approved",
                "reviewer": "dr_smith",
                "reviewed_at": "2026-10-18",
                "notes": "looks good",
            }
        }
        with self.assertRaises(ValueError) as ctx:
            require_independent_review(workbook_rows, self.alias_map, self_review_ledger)
        self.assertIn("cannot be approved by its own annotator", str(ctx.exception).lower())

        # Case 2: Independent approval (reviewer != observer_id) -> MUST PASS
        independent_ledger = {
            sample_id: {
                "sample_id": sample_id,
                "review_status": "approved",
                "reviewer": "dr_patel",
                "reviewed_at": "2026-10-18",
                "notes": "confirmed visual symptoms",
            }
        }
        gate_res = require_independent_review(workbook_rows, self.alias_map, independent_ledger)
        self.assertEqual(gate_res["independent_reviews"], 1)

    def test_review_ledger_rejection_exclusion(self) -> None:
        """Verify that rejected observations are blocked from training pairing."""
        alias = sorted(self.alias_map)[0]
        sample_id = self.alias_map[alias]

        collated = [
            {
                "sample_id": sample_id,
                "image_path": sample_id,
                "text": "pale chlorotic stippling along leaf margins",
                "text_origin": "user_observation",
                "provenance": "clinical pilot v1",
                "source_reference": "https://terramind.ai/pilot",
            }
        ]

        # Ledger with rejected status
        rejected_ledger = {
            sample_id: {
                "sample_id": sample_id,
                "review_status": "rejected",
                "reviewer": "dr_patel",
                "reviewed_at": "2026-10-18",
                "notes": "blurry photo, symptoms ambiguous",
            }
        }

        with self.assertRaises(ValueError) as ctx:
            require_reviewed(collated, rejected_ledger)
        self.assertIn("not approved by review", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
