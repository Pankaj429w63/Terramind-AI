from __future__ import annotations

import tempfile
import unittest
import csv
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from ml.multimodal.data import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    PlantWildMetadataDataset,
    build_vocabulary,
    collate_metadata_pairs,
    discover_pairs,
    encode_metadata,
    metadata_tokens,
)
from ml.multimodal.model import MultimodalAutoencoder, MultimodalConfig
from ml.multimodal.runtime import multimodal_status
from ml.fusion.data import (
    build_vocabulary as build_fusion_vocabulary,
    load_paired_text_csv,
    load_official_split,
    PairedExample,
    validate_provenance,
    validate_source_reference,
    validate_text_not_label_derived,
)
from ml.fusion.eligibility import (
    build_eligibility_manifest,
    filter_paired_examples,
    load_eligibility_manifest,
    validate_eligibility_manifest,
    write_eligibility_manifest,
)
from ml.fusion.models import FusionConfig, LateFusionClassifier, build_fusion_model
from ml.fusion.registry import discover_models
from ml.fusion.text_only import CHECKPOINT_FORMAT, TextOnlyClassifier, load_text_only_model
from ml.fusion.annotation import (
    OBSERVATION_COLUMNS,
    REVIEW_COLUMNS,
    WORKBOOK_COLUMNS,
    build_alias_map,
    build_workbook_rows,
    collate_observations,
    load_alias_map,
    load_annotator_workbook,
    load_review_ledger,
    observation_issues,
    require_instrument_version,
    require_reviewed,
    validate_observed_at,
    write_alias_map,
    write_observations_csv,
    write_workbook,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = REPO_ROOT / "data" / "annotation"


class MultimodalPreparationTests(unittest.TestCase):
    def test_text_is_only_tokenized_class_metadata_and_vocabulary_is_stable(self) -> None:
        labels = {0: "Tomato early blight", 1: "Healthy tomato"}
        self.assertEqual(metadata_tokens(labels[0]), ["tomato", "early", "blight"])
        self.assertEqual(build_vocabulary(labels), build_vocabulary(labels))
        vocabulary = build_vocabulary(labels)
        encoded = encode_metadata(labels[0], vocabulary)
        self.assertEqual(encoded[0], BOS_ID)
        self.assertEqual(encoded[-1], EOS_ID)
        self.assertEqual([k for k, v in vocabulary.items() if v == PAD_ID], ["<pad>"])

    def test_split_records_pair_existing_image_paths_with_exact_class_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "tomato early blight").mkdir(parents=True)
            image = root / "images" / "tomato early blight" / "sample.jpg"
            Image.new("RGB", (32, 32), "green").save(image)
            (root / "classes.txt").write_text("0 tomato early blight\n", encoding="utf-8")
            (root / "train_split.txt").write_text("tomato early blight/sample.jpg=0=1\n", encoding="utf-8")
            records = discover_pairs(root, Path("train_split.txt"))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].metadata_text, "tomato early blight")
            dataset = PlantWildMetadataDataset(records, build_vocabulary({0: "tomato early blight"}), ToTensor())
            batch = collate_metadata_pairs([dataset[0]])
            self.assertEqual(tuple(batch["images"].shape), (1, 3, 32, 32))

    def test_split_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "classes.txt").write_text("0 tomato\n", encoding="utf-8")
            (root / "train_split.txt").write_text("../outside.jpg=0=1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                discover_pairs(root, Path("train_split.txt"))


class MultimodalArchitectureTests(unittest.TestCase):
    def test_reconstruction_cross_attention_and_fused_shared_z(self) -> None:
        torch.manual_seed(7)
        feature_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.AdaptiveAvgPool2d((7, 7)),
        )
        config = MultimodalConfig(
            latent_dim=32,
            text_embedding_dim=32,
            attention_heads=4,
            transformer_layers=1,
            dropout=0.0,
            max_image_tokens=64,
            max_text_length=8,
        )
        model = MultimodalAutoencoder(feature_encoder, vocabulary_size=12, feature_dim=16, config=config)
        images = torch.randn(2, 3, 224, 224)
        text_ids = torch.tensor([[2, 4, 5, 3], [2, 6, 3, PAD_ID]])
        class_ids = torch.tensor([1, 2])
        outputs = model(images, text_ids)
        losses = model.loss(outputs, images, text_ids, class_ids)
        self.assertEqual(tuple(outputs["image_reconstruction"].shape), tuple(images.shape))
        self.assertEqual(tuple(outputs["text_logits"].shape), (2, 3, 12))
        self.assertEqual(tuple(outputs["fused_z"].shape), (2, 32))
        self.assertEqual(tuple(outputs["image_cross_attention"].shape), (2, 49, 32))
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(model.fusion_projection[0].weight.grad)

    def test_untrained_checkpoint_is_reported_as_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status = multimodal_status(Path(directory) / "not-trained.pth")
        self.assertEqual(status["status"], "unavailable")
        self.assertFalse(status["trained"])
        self.assertFalse(status["checkpoint_available"])
        methods = {item["name"]: item for item in status["model_availability"]}
        self.assertTrue(methods["Image-only EfficientNet-B0"]["trained"])
        for name in ("Early fusion", "Intermediate fusion", "Late fusion"):
            self.assertFalse(methods[name]["trained"])
            self.assertTrue(methods[name]["requires_text"])


class SupervisedFusionTests(unittest.TestCase):
    def _fixture(self, root: Path):
        (root / "images").mkdir()
        paths = {"train": "train_a.jpg", "validation": "val_a.jpg", "test": "test_a.jpg"}
        classes = {"train": 0, "validation": 1, "test": 0}
        for split, image_name in paths.items():
            Image.new("RGB", (24, 24), (20 + len(split), 80, 30)).save(root / "images" / image_name)
            (root / f"{split}.txt").write_text(f"{image_name}={classes[split]}=1\n", encoding="utf-8")
        (root / "classes.txt").write_text("0 tomato early blight\n1 apple scab\n", encoding="utf-8")
        labels = {0: "tomato early blight", 1: "apple scab"}
        rows = [
            {"sample_id": paths[s], "image_path": paths[s], "text": f"leaf has irregular brown spots number {i}",
             "text_origin": "paired_human_observation",
             "provenance": "fixture observer notes, collected 2026-01-01",
             "source_reference": f"fixture://observer/{i}"}
            for i, s in enumerate(paths)
        ]
        csv_path = root / "pairs.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return labels, csv_path, {name: Path(f"{name}.txt") for name in paths}

    def test_all_fusion_models_forward_and_late_weight_contract(self):
        torch.manual_seed(8)
        config = FusionConfig(num_classes=3, image_feature_dim=8, text_vocab_size=16,
                             text_embedding_dim=12, text_hidden_dim=16, fusion_dim=24,
                             dropout=0.0, max_text_length=8)
        images = torch.randn(2, 3, 32, 32)
        tokens = torch.tensor([[3, 4, 2], [5, 2, 0]])
        for kind in ("early", "intermediate", "late"):
            encoder = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.AdaptiveAvgPool2d((4, 4)))
            model = build_fusion_model(kind, encoder, config)
            logits = model(images, tokens)
            self.assertEqual(tuple(logits.shape), (2, 3))
            logits.mean().backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters() if p.requires_grad))
            with self.assertRaises(ValueError):
                model(images, tokens[:1])
            if isinstance(model, LateFusionClassifier):
                with self.assertRaises(RuntimeError):
                    model.predict_proba(images, tokens)
                probs = model.predict_proba(images, tokens, image_weight=0.7)
                self.assertEqual(tuple(probs.shape), (2, 3))

    def test_pairing_uses_official_labels_and_splits_and_train_only_vocab(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, csv_path, splits = self._fixture(root)
            examples = load_paired_text_csv(csv_path, root, splits, labels)
            self.assertEqual([(e.split, e.class_id) for e in examples], [("train", 0), ("validation", 1), ("test", 0)])
            train_vocab = build_fusion_vocabulary([examples[0].text])
            self.assertNotIn("apple", train_vocab)
            official = load_official_split(root, splits)
            self.assertEqual(len(official), 3)

    def test_target_name_and_label_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "class name"):
            validate_text_not_label_derived("observed tomato early blight on leaves", {0: "tomato early blight"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, csv_path, splits = self._fixture(root)
            fields = ["sample_id", "image_path", "text", "text_origin", "provenance", "source_reference", "class_id"]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
            with self.assertRaisesRegex(ValueError, "must not include target labels"):
                load_paired_text_csv(csv_path, root, splits, labels)

    def test_pair_schema_requires_unique_ids_and_source_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, csv_path, splits = self._fixture(root)
            with csv_path.open(encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            rows[1]["sample_id"] = rows[0]["sample_id"]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
                load_paired_text_csv(csv_path, root, splits, labels)

    def test_unverified_text_origin_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, csv_path, splits = self._fixture(root)
            with csv_path.open(encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            rows[0]["text_origin"] = "class_prompt"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Unapproved text_origin"):
                load_paired_text_csv(csv_path, root, splits, labels)

    def test_duplicate_image_content_across_official_splits_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, csv_path, splits = self._fixture(root)
            (root / "images" / "test_a.jpg").write_bytes((root / "images" / "train_a.jpg").read_bytes())
            with self.assertRaisesRegex(ValueError, "Duplicate image content crosses"):
                load_paired_text_csv(csv_path, root, splits, labels)

    def test_registry_reports_checkpoint_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_file = root / "classes.txt"
            labels_file.write_text("0 tomato\n1 apple\n", encoding="utf-8")
            checkpoint = root / "fusion" / "early" / "best_model.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"not deserialized by registry")
            import json as json_module
            manifest = {"fusion_kind": "early", "trained": True,
                        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "class_labels": {"0": "tomato", "1": "apple"}, "validation_macro_f1": 0.4}
            checkpoint.with_suffix(".manifest.json").write_text(json_module.dumps(manifest), encoding="utf-8")
            registry = discover_models(root / "fusion", labels_file, root / "absent-production.pth")
            models_by_kind = {item["kind"]: item for item in registry["models"]}
            self.assertTrue(models_by_kind["early"]["available"])
            self.assertFalse(models_by_kind["early"]["loaded"])
            self.assertFalse(models_by_kind["late"]["trained"])
            self.assertFalse(registry["experimental_weights_loaded"])
            manifest["class_labels"] = {"0": "apple", "1": "tomato"}
            checkpoint.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            incompatible = discover_models(root / "fusion", labels_file, root / "absent-production.pth")
            self.assertFalse({item["kind"]: item for item in incompatible["models"]}["early"]["available"])


class DuplicateFreeEligibilityTests(unittest.TestCase):
    @staticmethod
    def _fixture(root: Path):
        images = root / "images"
        images.mkdir()
        payloads = {
            "train_a.jpg": b"AAA", "test_a.jpg": b"AAA",           # cross-split duplicate
            "train_dup_1.jpg": b"BBB", "train_dup_2.jpg": b"BBB",   # within-split duplicate copy
            "train_conflict_1.jpg": b"CCC", "train_conflict_2.jpg": b"CCC",  # within-split conflicting labels
            "val_a.jpg": b"DDD", "test_b.jpg": b"EEE",
        }
        for name, payload in payloads.items():
            (images / name).write_bytes(payload)
        (root / "classes.txt").write_text("0 tomato early blight\n1 apple scab\n", encoding="utf-8")
        (root / "train_split.txt").write_text(
            "train_a.jpg=0=1\ntrain_dup_1.jpg=0=1\ntrain_dup_2.jpg=0=1\n"
            "train_conflict_1.jpg=0=1\ntrain_conflict_2.jpg=1=1\n", encoding="utf-8")
        (root / "val_split.txt").write_text("val_a.jpg=1=1\n", encoding="utf-8")
        (root / "test_split.txt").write_text("test_a.jpg=0=1\ntest_b.jpg=0=1\n", encoding="utf-8")
        labels = {0: "tomato early blight", 1: "apple scab"}
        splits = {"train": Path("train_split.txt"), "validation": Path("val_split.txt"), "test": Path("test_split.txt")}
        return labels, splits

    def test_manifest_is_reproducible_and_excludes_duplicate_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits = self._fixture(root)
            first = build_eligibility_manifest(root, splits, labels)
            second = build_eligibility_manifest(root, splits, labels)
            self.assertEqual(first, second)
            self.assertEqual(first["eligible_rows"], 3)
            self.assertEqual(first["eligible_counts_by_split"], {"train": 1, "validation": 1, "test": 1})
            self.assertEqual(first["duplicate_groups"], {"all": 3, "cross_split": 1, "within_split": 2})
            eligible = {item["sample_id"] for item in first["eligible"]}
            self.assertEqual(eligible, {"train_dup_1.jpg", "val_a.jpg", "test_b.jpg"})
            # Every member of a cross-split group is excluded, not just the copy.
            self.assertNotIn("train_a.jpg", eligible)
            self.assertNotIn("test_a.jpg", eligible)
            # Conflicting-label copies are both excluded; the same-label copy keeps the first path only.
            self.assertNotIn("train_conflict_1.jpg", eligible)
            self.assertNotIn("train_conflict_2.jpg", eligible)
            self.assertEqual(first["eligible_by_split"]["train"], ["train_dup_1.jpg"])
            self.assertEqual(first["excluded_counts_by_reason"],
                             {"duplicate_content_crosses_official_splits": 2,
                              "duplicate_content_has_conflicting_labels": 2,
                              "duplicate_copy_within_same_split": 1})

    def test_class_alignment_and_split_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits = self._fixture(root)
            manifest = build_eligibility_manifest(root, splits, labels)
            official = load_official_split(root, splits)
            for entry in manifest["eligible"]:
                self.assertEqual(entry["class_name"], labels[entry["class_id"]])
                # Class and split come only from the untouched official split files: nothing moved.
                self.assertEqual((entry["class_id"], entry["split"]), official[entry["sample_id"]])
                self.assertIn(entry["sample_id"], manifest["eligible_by_split"][entry["split"]])
            self.assertEqual(set(manifest["eligible_by_split"]), set(splits))

    def test_manifest_round_trip_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits = self._fixture(root)
            manifest = build_eligibility_manifest(root, splits, labels)
            path = root / "eligibility.json"
            write_eligibility_manifest(manifest, path)
            reloaded = load_eligibility_manifest(path)
            eligible = validate_eligibility_manifest(reloaded, root, splits, labels)
            self.assertEqual(eligible, {"train_dup_1.jpg", "val_a.jpg", "test_b.jpg"})
            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["eligible"][0]["class_id"] = 1 - tampered["eligible"][0]["class_id"]
            with self.assertRaisesRegex(ValueError, "unsupported or altered"):
                validate_eligibility_manifest(tampered, root, splits, labels)

    def test_only_eligible_paired_examples_are_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits = self._fixture(root)
            manifest = build_eligibility_manifest(root, splits, labels)
            eligible = {item["sample_id"] for item in manifest["eligible"]}
            examples = [
                PairedExample(root / "images" / name, name, "observed leaf note", 0, "train",
                              "paired_human_observation", "fixture observer notes, collected 2026-01-01",
                              name, f"fixture://observer/{name}")
                for name in ("train_a.jpg", "train_dup_1.jpg", "val_a.jpg")
            ]
            selected = filter_paired_examples(examples, eligible)
            self.assertEqual([item.relative_path for item in selected], ["train_dup_1.jpg", "val_a.jpg"])


class TextOnlyBaselineTests(unittest.TestCase):
    def test_text_only_forward_dimensions_and_input_contract(self):
        torch.manual_seed(11)
        config = FusionConfig(num_classes=3, text_vocab_size=16, text_embedding_dim=12,
                              text_hidden_dim=16, fusion_dim=24, dropout=0.0, max_text_length=8)
        model = TextOnlyClassifier(config)
        tokens = torch.tensor([[3, 4, 2], [5, 2, 0]])
        logits = model(tokens)
        self.assertEqual(tuple(logits.shape), (2, 3))
        logits.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.text_encoder.parameters()))
        with self.assertRaises(ValueError):
            model(torch.tensor([1, 2, 3]))

    def test_text_only_checkpoint_validation_requires_matching_class_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = {0: "tomato early blight", 1: "apple scab"}
            config = FusionConfig(num_classes=2, text_vocab_size=8, text_embedding_dim=8,
                                  text_hidden_dim=8, fusion_dim=16, dropout=0.0, max_text_length=6)
            checkpoint = root / "text_only" / "best_model.pth"
            checkpoint.parent.mkdir(parents=True)
            model = TextOnlyClassifier(config)
            torch.save({
                "format": CHECKPOINT_FORMAT, "fusion_kind": "text_only",
                "model_config": config.__dict__, "class_labels": labels,
                "model_state": model.state_dict(), "vocabulary": {"<pad>": 0, "<unk>": 1},
            }, checkpoint)
            checkpoint.with_suffix(".manifest.json").write_text(json.dumps({
                "fusion_kind": "text_only", "trained": True,
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "class_labels": {str(key): value for key, value in labels.items()}, "validation_macro_f1": 0.5,
            }), encoding="utf-8")
            loaded, payload = load_text_only_model(checkpoint, labels)
            self.assertEqual(tuple(loaded(torch.tensor([[0, 2, 3]])).shape), (1, 2))
            self.assertEqual(payload["fusion_kind"], "text_only")
            with self.assertRaisesRegex(ValueError, "class ordering"):
                load_text_only_model(checkpoint, {0: "other", 1: "labels"})


class AnnotationWorkflowTests(unittest.TestCase):
    @staticmethod
    def _dataset(root: Path):
        images = root / "images"
        images.mkdir()
        for name, payload in (("train_a.jpg", b"AAA"), ("val_a.jpg", b"BBB"), ("test_a.jpg", b"CCC")):
            (images / name).write_bytes(payload)
        (root / "classes.txt").write_text("0 tomato early blight\n1 apple scab\n", encoding="utf-8")
        (root / "train_split.txt").write_text("train_a.jpg=0=1\n", encoding="utf-8")
        (root / "val_split.txt").write_text("val_a.jpg=1=1\n", encoding="utf-8")
        (root / "test_split.txt").write_text("test_a.jpg=0=1\n", encoding="utf-8")
        labels = {0: "tomato early blight", 1: "apple scab"}
        splits = {"train": Path("train_split.txt"), "validation": Path("val_split.txt"), "test": Path("test_split.txt")}
        return labels, splits

    def test_workbook_is_blinded_deterministic_and_alias_is_opaque(self) -> None:
        samples = ["apple scab/1.jpg", "tomato early blight/2.jpg", "cherry leaf/9.jpg"]
        first = build_alias_map(samples, "fixed-salt")
        self.assertEqual(first, build_alias_map(samples, "fixed-salt"))
        self.assertNotEqual(first, build_alias_map(samples, "other-salt"))
        for alias in first:
            self.assertTrue(alias.startswith("obs-"))
            for leaked in ("apple", "scab", "tomato", "cherry", "jpg", "/"):
                self.assertNotIn(leaked, alias.casefold())
        rows = build_workbook_rows(first)
        self.assertEqual(len(rows), len(samples))
        for row in rows:
            self.assertEqual(set(row), set(WORKBOOK_COLUMNS))
            self.assertEqual(row["observation"], "")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_workbook(rows, root / "workbook.csv")
            self.assertEqual(load_annotator_workbook(path), rows)

    def test_workbook_rejects_label_or_path_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for extra in ("image_path", "class_name"):
                path = root / f"leaky_{extra}.csv"
                path.write_text(f"alias,observation,observer_id,observed_at,{extra}\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "must not expose"):
                    load_annotator_workbook(path)

    def test_collation_copies_text_verbatim_and_rejects_leakage(self) -> None:
        labels = {0: "tomato early blight", 1: "apple scab"}
        alias_map = build_alias_map(["train_a.jpg"], "salt")
        alias = next(iter(alias_map))
        kwargs = {"text_origin": "paired_human_observation",
                  "provenance": "curator interview notes, collected 2026-01-01",
                  "source_reference": "https://example.org/annotations/1"}
        good = collate_observations(
            [{"alias": alias, "observation": "leaf with irregular brown spots and pale margins",
              "observer_id": "obs-1", "observed_at": "2026-01-01"}],
            alias_map, labels, **kwargs)
        self.assertEqual(good[0]["sample_id"], "train_a.jpg")
        self.assertEqual(good[0]["image_path"], "train_a.jpg")
        self.assertEqual(good[0]["text"], "leaf with irregular brown spots and pale margins")
        with self.assertRaisesRegex(ValueError, "class name"):
            collate_observations(
                [{"alias": alias, "observation": "this is tomato early blight", "observer_id": "obs-1",
                  "observed_at": "2026-01-01"}], alias_map, labels, **kwargs)
        with self.assertRaisesRegex(ValueError, "unknown alias"):
            collate_observations(
                [{"alias": "obs-unknown", "observation": "brown spots", "observer_id": "obs-1",
                  "observed_at": "2026-01-01"}], alias_map, labels, **kwargs)

    def test_review_gate_blocks_unapproved_or_unobserved(self) -> None:
        collated = [{"sample_id": "train_a.jpg"}]
        with self.assertRaisesRegex(ValueError, "lack a review-ledger entry"):
            require_reviewed(collated, {})
        with self.assertRaisesRegex(ValueError, "not approved"):
            require_reviewed(collated, {"train_a.jpg": {"review_status": "pending"}})
        self.assertEqual(require_reviewed(collated, {"train_a.jpg": {"review_status": "approved"}}),
                         {"collated": 1, "approved": 1})
        with self.assertRaisesRegex(ValueError, "unobserved samples"):
            require_reviewed(collated, {"train_a.jpg": {"review_status": "approved"},
                                        "val_a.jpg": {"review_status": "approved"}})

    def test_instrument_version_must_be_recorded_in_provenance(self) -> None:
        provenance = "user elicitation instrument v3.2, university panel, 2026-03"
        self.assertEqual(require_instrument_version(provenance, "v3.2"), "v3.2")
        with self.assertRaisesRegex(ValueError, "instrument-version is required"):
            require_instrument_version(provenance, "")
        with self.assertRaisesRegex(ValueError, "must record the elicitation instrument version"):
            require_instrument_version("user elicitation, university panel, 2026-03", "v9.9")

    def test_review_ledger_rejects_duplicates_and_unknown_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.csv"
            duplicate.write_text(
                "sample_id,review_status,reviewer,reviewed_at,notes\n"
                "train_a.jpg,approved,alice,2026-01-01,ok\ntrain_a.jpg,approved,bob,2026-01-02,ok\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
                load_review_ledger(duplicate)
            bad_status = root / "bad_status.csv"
            bad_status.write_text("sample_id,review_status,reviewer,reviewed_at,notes\ntrain_a.jpg,accepted,x,y,z\n",
                                  encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unapproved review_status"):
                load_review_ledger(bad_status)

    def test_emitted_observations_pass_the_training_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits = self._dataset(root)
            alias_map = build_alias_map(["train_a.jpg", "val_a.jpg", "test_a.jpg"], "salt")
            write_alias_map(alias_map, root / "sealed.json", "salt")
            reloaded_map, salt = load_alias_map(root / "sealed.json")
            self.assertEqual(reloaded_map, alias_map)
            self.assertEqual(salt, "salt")
            observations = {"train_a.jpg": "leaf edge necrosis with dark sunken lesions",
                            "val_a.jpg": "powdery white coating on the upper surface",
                            "test_a.jpg": "small circular spots with yellow halos"}
            rows = [{"alias": alias, "observation": observations[sample_id], "observer_id": "obs-1",
                     "observed_at": "2026-01-01"} for alias, sample_id in alias_map.items()]
            collated = collate_observations(
                rows, alias_map, labels, text_origin="paired_human_observation",
                provenance="curator interview notes, collected 2026-01-01",
                source_reference="https://example.org/annotations/set-1")
            ledger = {row["sample_id"]: {"review_status": "approved"} for row in collated}
            require_reviewed(collated, ledger)
            csv_path = write_observations_csv(collated, root / "observations.csv")
            pairs = load_paired_text_csv(csv_path, root, splits, labels)
            self.assertEqual({pair.split for pair in pairs}, {"train", "validation", "test"})
            self.assertEqual({pair.text_origin for pair in pairs}, {"paired_human_observation"})

    def test_templates_expose_only_the_required_safe_columns(self) -> None:
        expected = {
            "annotator_workbook_template.csv": WORKBOOK_COLUMNS,
            "observations_template.csv": OBSERVATION_COLUMNS,
            "review_ledger_template.csv": REVIEW_COLUMNS,
        }
        for filename, columns in expected.items():
            header = (TEMPLATE_ROOT / filename).read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(header.split(","), list(columns))
        workbook_header = (TEMPLATE_ROOT / "annotator_workbook_template.csv").read_text(encoding="utf-8")
        self.assertEqual(workbook_header.strip(), ",".join(WORKBOOK_COLUMNS))


class AnnotationValidationStage2FTests(unittest.TestCase):
    """Stage 2F regression tests: precise provenance/URI validation and atomic emission."""

    VALID_PROVENANCE = (
        "blinded image observation session; panel of 6 agronomy participants; "
        "collected 2026-02-10 to 2026-02-12; instrument v1"
    )
    VALID_SOURCE = "https://osf.io/ab12c/"

    # Documented single-line Windows CMD commands. They must stay copy-pasteable into
    # cmd.exe: no PowerShell backtick or CMD caret continuations, no angle-bracket
    # placeholders, and no unfilled ACTUAL_*/YOUR_ACTUAL_* tokens.
    WINDOWS_CMD_COMMANDS = (
        "python -c \"import csv;rows=list(csv.DictReader(open('outputs/annotation/annotator_workbook.csv',newline='',encoding='utf-8-sig')));filled=sum(1 for r in rows if (r.get('observation') or '').strip());print('rows',len(rows),'filled',filled,'missing',len(rows)-filled)\"",
        "python -c \"import csv,collections;rows=list(csv.DictReader(open('data/annotation/review_ledger.csv',newline='',encoding='utf-8-sig')));print(dict(collections.Counter((r.get('review_status') or '').strip().lower() for r in rows)))\"",
        "python -m scripts.validate_annotations --workbook \"outputs/annotation/annotator_workbook.csv\" --alias-map \"outputs/annotation/sealed/alias_map.json\" --output \"outputs/research/annotation_validation.json\"",
        "python scripts/audit_fusion_data.py",
        "python -m unittest discover -s tests -p test_multimodal.py -v",
    )

    @staticmethod
    def _dataset(root: Path):
        images = root / "images"
        images.mkdir()
        for name, payload in (("train_a.jpg", b"AAA"), ("val_a.jpg", b"BBB"), ("test_a.jpg", b"CCC")):
            (images / name).write_bytes(payload)
        (root / "classes.txt").write_text("0 tomato early blight\n1 apple scab\n", encoding="utf-8")
        (root / "train_split.txt").write_text("train_a.jpg=0=1\n", encoding="utf-8")
        (root / "val_split.txt").write_text("val_a.jpg=1=1\n", encoding="utf-8")
        (root / "test_split.txt").write_text("test_a.jpg=0=1\n", encoding="utf-8")
        labels = {0: "tomato early blight", 1: "apple scab"}
        splits = {"train": Path("train_split.txt"), "validation": Path("val_split.txt"), "test": Path("test_split.txt")}
        return labels, splits

    @staticmethod
    def _write_paired_csv(root: Path, rows: list[dict[str, str]]) -> Path:
        path = root / "paired.csv"
        columns = ["sample_id", "image_path", "text", "text_origin", "provenance", "source_reference"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _paired_row(self, **overrides: str) -> dict[str, str]:
        row = {"sample_id": "train_a.jpg", "image_path": "train_a.jpg", "text": "dark sunken lesions on the leaf edge",
               "text_origin": "user_observation", "provenance": self.VALID_PROVENANCE,
               "source_reference": self.VALID_SOURCE}
        row.update(overrides)
        return row

    def test_valid_provenance_and_instrument_version_pass(self) -> None:
        validate_provenance(self.VALID_PROVENANCE, self.VALID_SOURCE)
        validate_source_reference(self.VALID_SOURCE)
        self.assertEqual(require_instrument_version(self.VALID_PROVENANCE, "v1"), "v1")

    def test_missing_and_mismatched_instrument_version_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "instrument-version is required"):
            require_instrument_version(self.VALID_PROVENANCE, "")
        with self.assertRaisesRegex(ValueError, "must record the elicitation instrument version"):
            require_instrument_version("blinded session, collected 2026-02-10, instrument v7", "v1")
        # A version must match as a whole token, not as a prefix of a longer version string.
        with self.assertRaisesRegex(ValueError, "must record the elicitation instrument version"):
            require_instrument_version("blinded session, collected 2026-02-10, instrument v10", "v1")

    def test_placeholder_provenance_and_source_are_rejected(self) -> None:
        for placeholder in ("ACTUAL_POOL", "YOUR_ACTUAL_POOL", "<observer pool>", "actual_dates", "TODO_here"):
            with self.assertRaisesRegex(ValueError, "placeholder"):
                validate_provenance(f"blinded session collected 2026-02-10 {placeholder}", self.VALID_SOURCE)
        for placeholder in ("ACTUAL_SOURCE_REFERENCE", "<url>", "YOUR_ACTUAL_SOURCE", "n/a"):
            with self.assertRaisesRegex(ValueError, "placeholder"):
                validate_source_reference(placeholder)

    def test_provenance_requires_real_collection_detail(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing or too short"):
            validate_provenance("notes", self.VALID_SOURCE)
        with self.assertRaisesRegex(ValueError, "must record the actual collection dates"):
            validate_provenance("blinded observation session with a panel of observers", self.VALID_SOURCE)

    def test_invalid_urls_and_unsupported_schemes_are_rejected(self) -> None:
        cases = (
            ("", "required"),
            ("not a uri", "must be a URL or stable URI with a scheme"),
            ("C:\\data\\annotations.csv", "local drive path"),
            ("file:///tmp/annotations.csv", "must not be a local file"),
            ("data:text/plain,foo", "must not be a local data"),
            ("https://", "malformed or missing a host"),
            ("https://localhost", "malformed or missing a host"),
        )
        for value, pattern in cases:
            with self.assertRaisesRegex(ValueError, pattern):
                validate_source_reference(value)

    def test_collation_rejects_placeholder_provenance(self) -> None:
        labels = {0: "tomato early blight"}
        alias_map = build_alias_map(["train_a.jpg"], "salt")
        alias = next(iter(alias_map))
        rows = [{"alias": alias, "observation": "brown spots", "observer_id": "o1", "observed_at": "2026-01-01"}]
        for provenance in ("ACTUAL_POOL 2026-03-04", "<collected somewhere> 2026-03-04"):
            with self.assertRaisesRegex(ValueError, "placeholder"):
                collate_observations(rows, alias_map, labels, text_origin="user_observation",
                                     provenance=provenance, source_reference=self.VALID_SOURCE)

    def test_empty_and_unfilled_workbooks_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty.csv"
            empty.write_text("alias,observation,observer_id,observed_at\n", encoding="utf-8")
            self.assertEqual(load_annotator_workbook(empty), [])

            unfilled = root / "unfilled.csv"
            unfilled.write_text("alias,observation,observer_id,observed_at\nobs-a,,,\nobs-b,,,\n", encoding="utf-8")
            rows = load_annotator_workbook(unfilled)
            self.assertEqual(len(rows), 2)
            alias_map = build_alias_map(["train_a.jpg", "val_a.jpg"], "salt")
            # Unfilled rows are incomplete, not invalid: they must not be reported as rejected.
            self.assertEqual(observation_issues(rows, alias_map, {0: "tomato early blight"}), [])

    def test_malformed_timestamp_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ISO date"):
            validate_observed_at("01/02/2026", "obs-a")
        with self.assertRaisesRegex(ValueError, "needs an observed_at date"):
            validate_observed_at("", "obs-a")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workbook.csv"
            path.write_text("alias,observation,observer_id,observed_at\nobs-a,brown spots,o1,2026/01/01\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ISO date"):
                load_annotator_workbook(path)

    def test_observation_issues_reports_actionable_problems(self) -> None:
        labels = {0: "tomato early blight", 1: "apple scab"}
        alias_map = build_alias_map(["train_a.jpg"], "salt")
        alias = next(iter(alias_map))
        rows = [
            {"alias": alias, "observation": "no observer", "observer_id": "", "observed_at": "2026-01-01"},
            {"alias": alias, "observation": "bad date", "observer_id": "o1", "observed_at": "2026/01/01"},
            {"alias": "obs-unknown", "observation": "unmapped", "observer_id": "o1", "observed_at": "2026-01-01"},
            {"alias": alias, "observation": "tomato early blight", "observer_id": "o1", "observed_at": "2026-01-01"},
        ]
        reasons = [issue["reason"] for issue in observation_issues(rows, alias_map, labels)]
        self.assertTrue(any("missing observer_id" in reason for reason in reasons))
        self.assertTrue(any("ISO date" in reason for reason in reasons))
        self.assertTrue(any("unknown alias" in reason for reason in reasons))
        self.assertTrue(any("class name" in reason for reason in reasons))
        # The report must never echo the sealed path back to the annotator-facing surface.
        self.assertNotIn("train_a.jpg", json.dumps(observation_issues(rows, alias_map, labels)))

    def test_missing_review_ledger_blocks_emission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_review_ledger(Path(directory) / "absent_review_ledger.csv")

    def test_review_gate_rejects_every_non_approved_status(self) -> None:
        collated = [{"sample_id": "train_a.jpg"}]
        for status in ("pending", "rejected", "disputed"):
            with self.assertRaisesRegex(ValueError, "not approved"):
                require_reviewed(collated, {"train_a.jpg": {"review_status": status}})
        with self.assertRaisesRegex(ValueError, "lack a review-ledger entry"):
            require_reviewed(collated, {})
        self.assertNotEqual(require_reviewed(collated, {"train_a.jpg": {"review_status": "approved"}})["approved"], 0)

    def test_loader_rejects_duplicate_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits = self._dataset(root)
            row = self._paired_row()
            path = self._write_paired_csv(root, [row, dict(row)])
            with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
                load_paired_text_csv(path, root, splits, labels)

    def test_loader_rejects_sample_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits = self._dataset(root)
            path = self._write_paired_csv(root, [self._paired_row(sample_id="train_b.jpg")])
            with self.assertRaisesRegex(ValueError, "sample_id must equal the normalized image_path"):
                load_paired_text_csv(path, root, splits, labels)

    def test_loader_rejects_label_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits = self._dataset(root)
            path = self._write_paired_csv(root, [self._paired_row(text="this is tomato early blight")])
            with self.assertRaisesRegex(ValueError, "class name"):
                load_paired_text_csv(path, root, splits, labels)

    def test_loader_rejects_missing_or_invalid_source_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits = self._dataset(root)
            empty_source = self._write_paired_csv(root, [self._paired_row(source_reference="")])
            with self.assertRaisesRegex(ValueError, "Missing sample_id"):
                load_paired_text_csv(empty_source, root, splits, labels)
            placeholder_source = self._write_paired_csv(root, [self._paired_row(source_reference="ACTUAL_SOURCE_REFERENCE")])
            with self.assertRaisesRegex(ValueError, "placeholder"):
                load_paired_text_csv(placeholder_source, root, splits, labels)

    def test_failed_emission_does_not_create_or_replace_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "paired_observations.csv"
            with self.assertRaises(ValueError):
                write_observations_csv([{"not_a_column": "x"}], target)
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob("paired_observations.csv.*.partial")), [])

            target.write_text("preserve-me\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_observations_csv([{"not_a_column": "x"}], target)
            self.assertEqual(target.read_text(encoding="utf-8"), "preserve-me\n")
            self.assertEqual(list(root.glob("paired_observations.csv.*.partial")), [])

    def test_successful_emission_only_when_all_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits = self._dataset(root)
            alias_map = build_alias_map(["train_a.jpg", "val_a.jpg", "test_a.jpg"], "salt")
            observations = {
                "train_a.jpg": "leaf edge necrosis with dark sunken lesions",
                "val_a.jpg": "powdery white coating on the upper surface",
                "test_a.jpg": "small circular spots with yellow halos",
            }
            rows = [{"alias": alias, "observation": observations[sample_id], "observer_id": "o1",
                     "observed_at": "2026-01-01"} for alias, sample_id in alias_map.items()]
            collated = collate_observations(rows, alias_map, labels, text_origin="user_observation",
                                            provenance=self.VALID_PROVENANCE, source_reference=self.VALID_SOURCE)
            target = root / "paired_observations.csv"
            with self.assertRaisesRegex(ValueError, "lack a review-ledger entry"):
                require_reviewed(collated, {})
            self.assertFalse(target.exists())

            ledger = {row["sample_id"]: {"review_status": "approved"} for row in collated}
            require_reviewed(collated, ledger)
            csv_path = write_observations_csv(collated, target)
            pairs = load_paired_text_csv(csv_path, root, splits, labels)
            self.assertEqual({pair.split for pair in pairs}, {"train", "validation", "test"})
            self.assertEqual({pair.text_origin for pair in pairs}, {"user_observation"})

    def test_documented_windows_commands_are_single_line_and_placeholder_free(self) -> None:
        for command in self.WINDOWS_CMD_COMMANDS:
            self.assertTrue(command.startswith("python "), command)
            self.assertNotIn("\n", command)
            self.assertNotIn("`", command)  # no PowerShell backtick continuations
            self.assertNotIn("^", command)  # no CMD caret continuations
            self.assertNotIn("<", command)  # no angle-bracket placeholders
            self.assertNotIn(">", command)
            self.assertNotIn("ACTUAL_", command.upper())
            self.assertNotIn("YOUR_ACTUAL", command.upper())


if __name__ == "__main__":
    unittest.main()
