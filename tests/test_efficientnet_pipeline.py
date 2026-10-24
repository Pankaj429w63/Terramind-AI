from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.models.class_labels import load_class_labels
from ml.models.model_loader import DEFAULT_MODEL_PATH, load_checkpoint
from mvpdr.train_efficientnet_b0 import assert_official_split, exclude_duplicate_content


class DuplicateContentFilterTests(unittest.TestCase):
    def test_official_split_guard_rejects_same_path_in_two_splits(self):
        with self.assertRaisesRegex(RuntimeError, "path leakage"):
            assert_official_split(["crop/a.jpg=0=1"], ["crop/a.jpg=0=2"], [], {"crop": 0})

    def test_removes_every_copy_of_cross_split_and_within_split_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            (images / "train.jpg").write_bytes(b"same image")
            (images / "validation.jpg").write_bytes(b"same image")
            (images / "repeat_a.jpg").write_bytes(b"repeated only within split")
            (images / "repeat_b.jpg").write_bytes(b"repeated only within split")
            (images / "unique.jpg").write_bytes(b"unique image")
            splits = {
                "train": ["train.jpg=0=1", "repeat_a.jpg=1=1", "repeat_b.jpg=1=1"],
                "validation": ["validation.jpg=0=2"],
                "test": ["unique.jpg=2=0"],
            }
            with patch("mvpdr.train_efficientnet_b0.DATASET_ROOT", root):
                filtered, audit = exclude_duplicate_content(splits)
            self.assertEqual(filtered, {"train": [], "validation": [], "test": ["unique.jpg=2=0"]})
            self.assertEqual(audit["duplicate_content_groups"], 2)
            self.assertEqual(audit["eligible_counts"], {"train": 0, "validation": 0, "test": 1})


class ProductionCheckpointCompatibilityTests(unittest.TestCase):
    def test_production_checkpoint_class_order_matches_dataset_mapping(self):
        _model, metadata = load_checkpoint(DEFAULT_MODEL_PATH, device="cpu")
        expected = load_class_labels()
        self.assertEqual(metadata["labels"], expected)
        self.assertEqual(metadata["num_classes"], 89)


if __name__ == "__main__":
    unittest.main()
