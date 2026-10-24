"""Regression tests for the automated image-description generation pipeline.

These cover the behaviours the pipeline promises: label-free prompt handling,
provenance that can never be mistaken for a human observation, resumable and
atomic storage, duplicate prevention, torn-line tolerance, manifest-driven
eligibility (so no ineligible or label-carrying sample can enter generation),
and the guard rails that stop paid/remote backends from running unconfigured.
"""
from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ml.annotation.generation import (
    FAILED,
    FLAGGED,
    GENERATED,
    RECORD_COLUMNS,
    GenerationError,
    GenerationRecord,
    GenerationStore,
    PROMPTS,
    assess,
    build_backend,
    iter_eligible_targets,
    preflight,
    provenance_for,
    sanitize,
    sha256_file,
    source_reference_for,
    strip_prompt_prefix,
)
from ml.fusion.data import HUMAN_TEXT_ORIGINS, MODEL_TEXT_ORIGIN, validate_text_not_label_derived
from ml.fusion.eligibility import build_eligibility_manifest, write_eligibility_manifest
from ml.models.class_labels import load_class_labels
from ml.annotation.quality import audit_generation
from scripts.generate_image_annotations import _estimate_remote_cost, _parse_args

LOCAL_MODEL = "Salesforce/blip-image-captioning-base"


class _StubBackend:
    """Minimal stand-in exposing only the attributes ``provenance_for`` reads."""

    def __init__(self, name: str) -> None:
        self.name = name


def _make_dataset(root: Path) -> tuple[dict[int, str], dict[str, Path], Path, dict]:
    """Build a tiny but real dataset (unique image bytes) plus a valid manifest."""
    images = root / "images"
    images.mkdir(parents=True)
    payloads = {
        "train_a.jpg": b"IMG-A", "train_b.jpg": b"IMG-B",
        "val_a.jpg": b"IMG-C", "test_a.jpg": b"IMG-D",
    }
    for name, payload in payloads.items():
        (images / name).write_bytes(payload)
    (root / "classes.txt").write_text("0 tomato early blight\n1 apple scab\n", encoding="utf-8")
    (root / "train_split.txt").write_text("train_a.jpg=0=1\ntrain_b.jpg=1=1\n", encoding="utf-8")
    (root / "val_split.txt").write_text("val_a.jpg=1=1\n", encoding="utf-8")
    (root / "test_split.txt").write_text("test_a.jpg=0=1\n", encoding="utf-8")
    labels = {0: "tomato early blight", 1: "apple scab"}
    splits = {
        "train": Path("train_split.txt"),
        "validation": Path("val_split.txt"),
        "test": Path("test_split.txt"),
    }
    manifest = build_eligibility_manifest(root, splits, labels)
    manifest_path = root / "manifest.json"
    write_eligibility_manifest(manifest, manifest_path)
    return labels, splits, manifest_path, manifest


def _write_store(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(RECORD_COLUMNS))
        writer.writerows(rows)


def _row(sample_id: str, status: str = GENERATED, description: str = "leaf shows dark spots") -> list[str]:
    record = GenerationRecord(sample_id=sample_id, status=status, description=description)
    fields = record.to_row()
    return [fields[column] for column in RECORD_COLUMNS]


def _valid_row(sample_id: str, image_path: Path, description: str, **overrides: str) -> list[str]:
    """A record that passes every audit check unless an override breaks it on purpose."""
    stub = _StubBackend("local")
    prompt = PROMPTS["v1"]
    record = GenerationRecord(
        sample_id=sample_id,
        status=GENERATED,
        description=description,
        model=LOCAL_MODEL,
        model_version="base",
        prompt_version=prompt.version,
        prompt_sha256=prompt.sha256(),
        generated_at="2026-10-17T00:00:00+00:00",
        image_sha256=sha256_file(image_path),
        backend="local",
        provenance=provenance_for(stub, prompt, LOCAL_MODEL),
        source_reference=source_reference_for(stub, LOCAL_MODEL),
    )
    fields = record.to_row()
    fields.update(overrides)
    return [fields[column] for column in RECORD_COLUMNS]


def _audit(root: Path, labels, splits, manifest_path: Path, store_path: Path):
    return audit_generation(
        store_path, dataset_root=root, manifest_path=manifest_path,
        split_files=splits, labels=labels,
    )


class TextHygieneTests(unittest.TestCase):
    def test_sanitize_collapses_all_whitespace(self) -> None:
        self.assertEqual(sanitize("  a close-up\n view\t of  the leaf "), "a close-up view of the leaf")
        self.assertEqual(sanitize(""), "")

    def test_assess_empty_description_is_failed(self) -> None:
        status, notes = assess("", None)
        self.assertEqual(status, FAILED)
        self.assertTrue(notes)

    def test_assess_short_and_low_confidence_are_flagged(self) -> None:
        self.assertEqual(assess("two words", None)[0], FLAGGED)
        self.assertEqual(assess("a clear leaf with dark spots", -3.4)[0], FLAGGED)

    def test_assess_repetitive_wording_is_flagged(self) -> None:
        status, notes = assess("leaf leaf leaf leaf leaf leaf leaf leaf", None)
        self.assertEqual(status, FLAGGED)
        self.assertTrue(any("repetitive" in note for note in notes))

    def test_assess_normal_description_passes(self) -> None:
        status, notes = assess("a leaf with several dark brown spots on its surface", -0.6)
        self.assertEqual(status, GENERATED)
        self.assertEqual(notes, [])


class RecordContractTests(unittest.TestCase):
    def test_row_matches_declared_column_order(self) -> None:
        fields = GenerationRecord(sample_id="x.jpg", status=GENERATED).to_row()
        self.assertEqual(list(fields), list(RECORD_COLUMNS))

    def test_generated_text_is_never_a_human_origin(self) -> None:
        self.assertEqual(GenerationRecord(sample_id="x.jpg", status=GENERATED).text_origin, MODEL_TEXT_ORIGIN)
        self.assertNotIn(MODEL_TEXT_ORIGIN, HUMAN_TEXT_ORIGINS)

    def test_provenance_names_model_prompt_and_declares_non_human(self) -> None:
        prompt = PROMPTS["v1"]
        provenance = provenance_for(_StubBackend("local"), prompt, LOCAL_MODEL)
        self.assertIn(LOCAL_MODEL, provenance)
        self.assertIn(prompt.version, provenance)
        self.assertIn("not a human observation", provenance)

    def test_source_reference_is_a_real_uri(self) -> None:
        reference = source_reference_for(_StubBackend("local"), LOCAL_MODEL)
        self.assertTrue(reference.startswith("https://huggingface.co/"))


class GenerationStoreTests(unittest.TestCase):
    def test_append_then_reload_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.csv"
            store = GenerationStore(path)
            store.append(GenerationRecord(sample_id="a.jpg", status=GENERATED, description="dark spots"))
            store.append(GenerationRecord(sample_id="b.jpg", status=FAILED))
            store.close()
            reloaded = GenerationStore(path)
            reloaded.load()
            self.assertEqual(set(reloaded.records), {"a.jpg", "b.jpg"})
            self.assertEqual(reloaded.completed(), {"a.jpg"})
            self.assertEqual(reloaded.discarded_lines, 0)

    def test_load_drops_a_torn_trailing_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.csv"
            _write_store(path, [_row("a.jpg")])
            with path.open("a", newline="", encoding="utf-8") as handle:
                handle.write("truncated,only,three\n")
            store = GenerationStore(path)
            store.load()
            self.assertEqual(set(store.records), {"a.jpg"})
            self.assertEqual(store.discarded_lines, 1)

    def test_load_discards_a_duplicate_that_is_not_the_last_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.csv"
            _write_store(path, [_row("a.jpg"), _row("a.jpg", status=FAILED), _row("b.jpg")])
            store = GenerationStore(path)
            store.load()
            self.assertEqual(set(store.records), {"a.jpg", "b.jpg"})
            self.assertEqual(store.records["a.jpg"]["status"], GENERATED)
            self.assertEqual(store.discarded_lines, 1)

    def test_load_rejects_an_unexpected_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.csv"
            path.write_text("wrong,header\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                GenerationStore(path).load()

    def test_checkpoint_compacts_atomically_and_leaves_no_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.csv"
            store = GenerationStore(path)
            store.append(GenerationRecord(sample_id="a.jpg", status=GENERATED))
            store.append(GenerationRecord(sample_id="b.jpg", status=GENERATED))
            with path.open("a", newline="", encoding="utf-8") as handle:
                handle.write("torn,line\n")
            store.checkpoint()
            store.close()
            with path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], list(RECORD_COLUMNS))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(len(fields) == len(RECORD_COLUMNS) for fields in rows))
            self.assertEqual([p.name for p in Path(directory).iterdir() if p.suffix == ".partial"], [])

    def test_checkpoint_then_append_resumes_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.csv"
            store = GenerationStore(path)
            store.append(GenerationRecord(sample_id="a.jpg", status=GENERATED))
            store.maybe_checkpoint(1)
            store.append(GenerationRecord(sample_id="b.jpg", status=GENERATED))
            store.close()
            reloaded = GenerationStore(path)
            reloaded.load()
            self.assertEqual(reloaded.completed(), {"a.jpg", "b.jpg"})


class BackendGuardTests(unittest.TestCase):
    def test_hf_inference_backend_requires_a_token(self) -> None:
        with mock.patch.dict(os.environ, {"HF_TOKEN": "", "HUGGINGFACE_HUB_TOKEN": ""}):
            with self.assertRaises(GenerationError):
                build_backend("hf-inference", "any/model", prompt=PROMPTS["v1"])

    def test_openai_compatible_backend_requires_configuration(self) -> None:
        env = {"VLM_BASE_URL": "", "VLM_API_KEY": "", "VLM_MODEL": ""}
        with mock.patch.dict(os.environ, env):
            with self.assertRaises(GenerationError):
                build_backend("openai-compatible", "any/model", prompt=PROMPTS["v1"])

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaises(GenerationError):
            build_backend("does-not-exist", "any/model", prompt=PROMPTS["v1"])

    def test_remote_backend_refuses_to_run_without_an_explicit_price(self) -> None:
        args = _parse_args(["--backend", "hf-inference", "--all"])
        with self.assertRaises(SystemExit) as caught:
            _estimate_remote_cost(args, 17870)
        self.assertIn("cost estimate", str(caught.exception))

    def test_remote_backend_refuses_without_paid_usage_confirmation(self) -> None:
        args = _parse_args(["--backend", "hf-inference", "--all", "--price-per-1k-images", "0.5"])
        with self.assertRaises(SystemExit) as caught:
            _estimate_remote_cost(args, 10)
        self.assertIn("confirm-paid-usage", str(caught.exception))


class EligibleTargetTests(unittest.TestCase):
    def test_targets_are_manifest_driven_and_carry_no_class_information(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits, manifest_path, manifest = _make_dataset(root)
            targets = list(iter_eligible_targets(root, manifest_path, splits, labels))
            self.assertEqual({sample for sample, *_ in targets}, {item["sample_id"] for item in manifest["eligible"]})
            self.assertEqual(len(targets), manifest["eligible_rows"])
            by_id = {item["sample_id"]: item for item in manifest["eligible"]}
            for sample_id, image_path, digest, split in targets:
                self.assertTrue(image_path.is_relative_to((root / "images").resolve()))
                self.assertEqual(digest, by_id[sample_id]["sha256"])
                self.assertEqual(digest, sha256_file(image_path))
                self.assertEqual(split, by_id[sample_id]["split"])

    def test_a_tampered_manifest_is_rejected_before_any_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits, manifest_path, manifest = _make_dataset(root)
            manifest["eligible"][0]["class_id"] = 999
            write_eligibility_manifest(manifest, manifest_path)
            with self.assertRaises(ValueError):
                list(iter_eligible_targets(root, manifest_path, splits, labels))


class PreflightTests(unittest.TestCase):
    def test_output_cannot_be_the_human_workbook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits, manifest_path, _ = _make_dataset(root)
            workbook = root / "annotator_workbook.csv"
            with self.assertRaises(ValueError):
                preflight(
                    dataset_root=root, manifest_path=manifest_path, split_files=splits,
                    output_path=workbook, workbook_path=workbook, model_id=LOCAL_MODEL,
                )

    def test_preflight_reports_ready_and_counts_without_generating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits, manifest_path, manifest = _make_dataset(root)
            report = preflight(
                dataset_root=root, manifest_path=manifest_path, split_files=splits,
                output_path=root / "store.csv", workbook_path=root / "workbook.csv",
                model_id=LOCAL_MODEL,
            )
            self.assertTrue(report["ready"])
            self.assertEqual(report["missing_dependencies"], [])
            self.assertEqual(report["eligible_samples"], manifest["eligible_rows"])
            self.assertEqual(report["remaining"], manifest["eligible_rows"])
            self.assertEqual(report["records_on_disk"], 0)


class QualityPolicyTests(unittest.TestCase):
    """Per-record content flags exclude only that record; store-integrity flags halt everything."""

    def test_label_leak_excludes_only_the_offending_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits, manifest_path, _ = _make_dataset(root)
            images = root / "images"
            store_path = root / "store.csv"
            _write_store(store_path, [
                _valid_row("train_a.jpg", images / "train_a.jpg", "leaf with several dark brown spots"),
                _valid_row("train_b.jpg", images / "train_b.jpg", "the leaf shows apple scab lesions"),
            ])
            report, ok_rows, verdicts = _audit(root, labels, splits, manifest_path, store_path)
            self.assertFalse(report["integrity_blocked"])
            self.assertTrue(report["emittable"])
            self.assertEqual(report["checks"]["class_label_leakage"]["count"], 1)
            self.assertEqual([row["sample_id"] for row in ok_rows], ["train_a.jpg"])
            self.assertEqual(report["records_requiring_human_verification"], 1)
            leaking = next(v for v in verdicts if v.sample_id == "train_b.jpg")
            self.assertFalse(leaking.ok)
            self.assertIn("class-label leakage", leaking.reasons)

    def test_unsupported_disease_claim_excludes_only_the_offending_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits, manifest_path, _ = _make_dataset(root)
            images = root / "images"
            store_path = root / "store.csv"
            _write_store(store_path, [
                _valid_row("train_a.jpg", images / "train_a.jpg", "leaves show pale mottled patches"),
                _valid_row("train_b.jpg", images / "train_b.jpg", "the plant is infected by a fungus"),
            ])
            report, ok_rows, _ = _audit(root, labels, splits, manifest_path, store_path)
            self.assertFalse(report["integrity_blocked"])
            self.assertEqual(report["checks"]["unsupported_disease_claim"]["count"], 1)
            self.assertEqual([row["sample_id"] for row in ok_rows], ["train_a.jpg"])

    def test_excluded_records_are_never_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits, manifest_path, _ = _make_dataset(root)
            images = root / "images"
            store_path = root / "store.csv"
            _write_store(store_path, [
                _valid_row("train_b.jpg", images / "train_b.jpg", "the leaf shows apple scab lesions"),
            ])
            _audit(root, labels, splits, manifest_path, store_path)
            with store_path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["description"], "the leaf shows apple scab lesions")

    def test_duplicate_records_halt_emission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits, manifest_path, _ = _make_dataset(root)
            images = root / "images"
            store_path = root / "store.csv"
            row = _valid_row("train_a.jpg", images / "train_a.jpg", "leaf with dark spots")
            _write_store(store_path, [row, row])
            report, _ok, _ = _audit(root, labels, splits, manifest_path, store_path)
            self.assertTrue(report["integrity_blocked"])
            self.assertFalse(report["emittable"])
            self.assertEqual(report["decision"], "BLOCKED")
            self.assertGreaterEqual(report["checks"]["duplicate_records"]["count"], 1)

    def test_stale_image_hash_halts_emission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits, manifest_path, _ = _make_dataset(root)
            images = root / "images"
            store_path = root / "store.csv"
            _write_store(store_path, [
                _valid_row(
                    "train_a.jpg", images / "train_a.jpg", "leaf with dark spots",
                    image_sha256="0" * 64,
                ),
            ])
            report, ok_rows, _ = _audit(root, labels, splits, manifest_path, store_path)
            self.assertTrue(report["integrity_blocked"])
            self.assertEqual(ok_rows, [])
            self.assertGreaterEqual(report["checks"]["image_identity_or_corruption"]["count"], 1)

    def test_missing_provenance_halts_emission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits, manifest_path, _ = _make_dataset(root)
            images = root / "images"
            store_path = root / "store.csv"
            _write_store(store_path, [
                _valid_row("train_a.jpg", images / "train_a.jpg", "leaf with dark spots", provenance=""),
            ])
            report, ok_rows, _ = _audit(root, labels, splits, manifest_path, store_path)
            self.assertTrue(report["integrity_blocked"])
            self.assertEqual(ok_rows, [])
            self.assertGreaterEqual(report["checks"]["missing_or_invalid_provenance"]["count"], 1)

    def test_invalid_model_metadata_halts_emission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits, manifest_path, _ = _make_dataset(root)
            images = root / "images"
            store_path = root / "store.csv"
            _write_store(store_path, [
                _valid_row("train_a.jpg", images / "train_a.jpg", "leaf with dark spots", model=""),
            ])
            report, ok_rows, _ = _audit(root, labels, splits, manifest_path, store_path)
            self.assertTrue(report["integrity_blocked"])
            self.assertEqual(ok_rows, [])
            self.assertGreaterEqual(report["checks"]["invalid_model_metadata"]["count"], 1)

    def test_missing_records_do_not_produce_phantom_emission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits, manifest_path, manifest = _make_dataset(root)
            images = root / "images"
            store_path = root / "store.csv"
            _write_store(store_path, [
                _valid_row("train_a.jpg", images / "train_a.jpg", "leaf with dark spots"),
            ])
            report, ok_rows, _ = _audit(root, labels, splits, manifest_path, store_path)
            self.assertEqual(report["missing_generations"], manifest["eligible_rows"] - 1)
            self.assertEqual(len(ok_rows), 1)


class PrefixStrippingTests(unittest.TestCase):
    """Verify that prompt-prefix stripping handles tokenizer variations, hyphens, and delimiters."""

    def test_exact_prefix_is_stripped(self) -> None:
        prefix = "a close-up view of the plant surface showing"
        raw = "a close-up view of the plant surface showing several small dark brown spots"
        self.assertEqual(strip_prompt_prefix(raw, prefix), "several small dark brown spots")

    def test_spaced_hyphen_variation_is_stripped(self) -> None:
        """BLIP decodes 'close-up' with surrounding spaces ('close - up')."""
        prefix = "a close-up view of the plant surface showing"
        raw = "a close - up view of the plant surface showing the green leaves"
        self.assertEqual(strip_prompt_prefix(raw, prefix), "the green leaves")

    def test_irregular_hyphen_and_whitespace_variations(self) -> None:
        prefix = "a close-up view of the plant surface showing"
        variations = [
            "a  close-up   view of the plant surface showing yellowing",
            "a close -up view of the plant surface showing yellowing",
            "a close- up view of the plant surface showing yellowing",
            "a  close  -  up  view of the plant surface showing yellowing",
        ]
        for var in variations:
            self.assertEqual(strip_prompt_prefix(var, prefix), "yellowing")

    def test_punctuation_delimiters_are_stripped(self) -> None:
        prefix = "symptoms visible on this leaf include"
        self.assertEqual(
            strip_prompt_prefix("symptoms visible on this leaf include: necrotic margins", prefix),
            "necrotic margins",
        )
        self.assertEqual(
            strip_prompt_prefix("symptoms visible on this leaf include - necrotic margins", prefix),
            "necrotic margins",
        )
        self.assertEqual(
            strip_prompt_prefix("symptoms visible on this leaf include, necrotic margins", prefix),
            "necrotic margins",
        )

    def test_prompt_v3_symptom_ask_is_stripped(self) -> None:
        prefix = PROMPTS["v3"].text
        raw = "symptoms visible on this leaf include dark spots and yellowing around the edges"
        self.assertEqual(strip_prompt_prefix(raw, prefix), "dark spots and yellowing around the edges")

    def test_empty_string_returned_when_text_is_only_prefix(self) -> None:
        prefix = "symptoms visible on this leaf include"
        self.assertEqual(strip_prompt_prefix("symptoms visible on this leaf include", prefix), "")
        self.assertEqual(strip_prompt_prefix("symptoms visible on this leaf include:", prefix), "")
        self.assertEqual(strip_prompt_prefix("  symptoms visible on this leaf include  ", prefix), "")

    def test_unmatched_prefix_leaves_text_intact(self) -> None:
        prefix = "symptoms visible on this leaf include"
        text = "a healthy green leaf with no visible lesions"
        self.assertEqual(strip_prompt_prefix(text, prefix), text)

    def test_empty_or_none_inputs_handled_safely(self) -> None:
        self.assertEqual(strip_prompt_prefix("", "some prefix"), "")
        self.assertEqual(strip_prompt_prefix("some text", ""), "some text")


class ProvenancePreservationTests(unittest.TestCase):
    """Verify that candidate model generations preserve non-human provenance across versions."""

    def test_prompt_v3_is_registered_and_label_free(self) -> None:
        self.assertIn("v3", PROMPTS)
        prompt = PROMPTS["v3"]
        self.assertEqual(prompt.version, "v3")
        self.assertEqual(prompt.text, "symptoms visible on this leaf include")
        self.assertEqual(prompt.style, "caption_prefix")
        self.assertTrue(prompt.sha256())

    def test_provenance_string_identifies_prompt_and_model(self) -> None:
        stub = _StubBackend("local")
        for p_ver in ("v1", "v3"):
            prompt = PROMPTS[p_ver]
            prov = provenance_for(stub, prompt, LOCAL_MODEL)
            self.assertIn(f"prompt {p_ver}", prov)
            self.assertIn(LOCAL_MODEL, prov)
            self.assertIn(f"prompt sha256 {prompt.sha256()}", prov)
            self.assertIn("not a human observation and not independently reviewed", prov)

    def test_text_origin_is_always_model_generated(self) -> None:
        record = GenerationRecord(sample_id="test.jpg", status=GENERATED, prompt_version="v3")
        self.assertEqual(record.text_origin, MODEL_TEXT_ORIGIN)
        self.assertNotIn(record.text_origin, HUMAN_TEXT_ORIGINS)


class LabelLeakageChecksTests(unittest.TestCase):
    """Verify that class label leakage is detected and rejected without altering dataset records."""

    def test_exact_class_name_leakage_is_detected(self) -> None:
        labels = {0: "apple scab", 1: "coffee leaf rust"}
        with self.assertRaises(ValueError) as ctx:
            validate_text_not_label_derived("the leaf shows apple scab on the surface", labels)
        self.assertIn("PlantWild class name", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_text_not_label_derived("signs of coffee leaf rust visible", labels)
        self.assertIn("PlantWild class name", str(ctx.exception))

    def test_descriptive_symptoms_without_class_name_pass(self) -> None:
        labels = {0: "apple scab", 1: "coffee leaf rust"}
        # "rust" alone or "scab" alone or "lesions" does not match exact class phrase
        validate_text_not_label_derived("the leaf surface shows dark brown circular spots", labels)
        validate_text_not_label_derived("yellowing edges and necrosis along the veins", labels)

    def test_label_leakage_in_audit_excludes_record_without_rewriting_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, splits, manifest_path, _ = _make_dataset(root)
            images = root / "images"
            store_path = root / "store.csv"
            _write_store(store_path, [
                _valid_row("train_a.jpg", images / "train_a.jpg", "leaf with dark spots"),
                _valid_row("train_b.jpg", images / "train_b.jpg", "lesions caused by apple scab"),
            ])
            report, ok_rows, verdicts = _audit(root, labels, splits, manifest_path, store_path)
            self.assertEqual(report["checks"]["class_label_leakage"]["count"], 1)
            self.assertEqual([r["sample_id"] for r in ok_rows], ["train_a.jpg"])
            # Verify the original store was untouched and not rewritten
            with store_path.open(newline="", encoding="utf-8-sig") as h:
                raw_rows = list(csv.DictReader(h))
            self.assertEqual(len(raw_rows), 2)
            self.assertEqual(raw_rows[1]["description"], "lesions caused by apple scab")


if __name__ == "__main__":
    unittest.main()
