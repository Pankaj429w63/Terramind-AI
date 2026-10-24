from __future__ import annotations

import contextlib
import datetime
import io
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient
from PIL import Image

from ml.annotation import review_tool
from ml.annotation.review_tool import check_context, create_app, load_review_context, main
from ml.fusion.annotation import (
    WORKBOOK_COLUMNS,
    alias_for,
    build_alias_map,
    load_annotator_workbook,
    require_independent_review,
    write_workbook,
)

OBSERVATION = "young leaf with irregular dark brown spots and a thin pale margin"


class BlindedReviewToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._extra: list[tempfile.TemporaryDirectory] = []
        self.root = Path(self._tmp.name)
        self.images = self.root / "images"
        class_dir = self.images / "apple scab"
        class_dir.mkdir(parents=True)
        self.sample_id = "apple scab/1.jpg"
        Image.new("RGB", (8, 8), (20, 120, 40)).save(class_dir / "1.jpg")

        self.salt = "review-tool-salt"
        self.alias = alias_for(self.sample_id, self.salt)
        self.alias_map_path = self.root / "alias_map.json"
        self.alias_map_path.write_text(
            json.dumps({"salt": self.salt, "aliases": {self.alias: self.sample_id}}), encoding="utf-8"
        )
        self.workbook_path = self.root / "workbook.csv"
        write_workbook([{"alias": self.alias, "observation": "", "observer_id": "", "observed_at": ""}], self.workbook_path)

        self.labels = {0: "apple scab"}
        self.context = load_review_context(
            self.alias_map_path, self.workbook_path, self.images, labels=self.labels
        )

    def tearDown(self) -> None:
        for extra in self._extra:
            extra.cleanup()
        self._tmp.cleanup()

    def _make_repo(self, sample_ids: list[str]) -> dict:
        """Build an isolated images/alias-map/workbook triple with several aliases."""
        tmp = tempfile.TemporaryDirectory()
        self._extra.append(tmp)
        root = Path(tmp.name)
        images = root / "images"
        aliases: dict[str, str] = {}
        for index, sample_id in enumerate(sample_ids):
            path = images / sample_id
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (10 + index, 120, 40)).save(path)
            aliases[alias_for(sample_id, self.salt)] = sample_id
        alias_map_path = root / "alias_map.json"
        alias_map_path.write_text(
            json.dumps({"salt": self.salt, "aliases": aliases}), encoding="utf-8"
        )
        workbook_path = root / "workbook.csv"
        write_workbook(
            [{"alias": alias, "observation": "", "observer_id": "", "observed_at": ""} for alias in aliases],
            workbook_path,
        )
        context = load_review_context(alias_map_path, workbook_path, images, labels=self.labels)
        return {
            "context": context,
            "workbook": workbook_path,
            "images": images,
            "alias_map": alias_map_path,
            "aliases": aliases,
            "by_sample": {sample_id: alias for alias, sample_id in aliases.items()},
        }

    # ---- core context ----

    def test_context_resolves_every_alias(self) -> None:
        report = check_context(self.context)
        self.assertEqual(report["decision"], "READY")
        self.assertEqual(report["resolvable_images"], 1)
        self.assertEqual(report["unresolved_aliases"], 0)
        # The report must never leak a path or a sample id.
        self.assertNotIn(self.sample_id, json.dumps(report))

    def test_next_alias_advances_and_skip_is_respected(self) -> None:
        self.assertEqual(self.context.next_alias(), self.alias)
        self.context.skip(self.alias)
        self.assertIsNone(self.context.next_alias())

    def test_record_saves_genuine_metadata_without_fabrication(self) -> None:
        self.context.set_observer("reviewer-7")
        self.context.record(self.alias, OBSERVATION)

        row = load_annotator_workbook(self.workbook_path)[0]
        self.assertEqual(row["observation"], OBSERVATION)
        self.assertEqual(row["observer_id"], "reviewer-7")
        self.assertEqual(row["observed_at"], datetime.date.today().isoformat())
        # the timestamp is a real ISO date, not an invented one
        datetime.date.fromisoformat(row["observed_at"])

    def test_record_refuses_to_overwrite_existing_observation(self) -> None:
        self.context.set_observer("reviewer-7")
        self.context.record(self.alias, OBSERVATION)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.context.record(self.alias, "a completely different later description")
        self.assertEqual(load_annotator_workbook(self.workbook_path)[0]["observation"], OBSERVATION)

    def test_observer_is_required_and_placeholders_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "observer id"):
            self.context.set_observer("unknown")
        with self.assertRaisesRegex(ValueError, "observer id"):
            self.context.set_observer("   ")
        with self.assertRaisesRegex(ValueError, "observer id"):
            self.context.record(self.alias, OBSERVATION)

    def test_label_derived_observation_is_rejected(self) -> None:
        self.context.set_observer("reviewer-7")
        with self.assertRaisesRegex(ValueError, "class name"):
            self.context.record(self.alias, "apple scab")
        # nothing was written for a rejected observation
        self.assertEqual(load_annotator_workbook(self.workbook_path)[0]["observation"], "")

    def test_unresolvable_alias_blocks_the_tool(self) -> None:
        traversal = self.root / "secret.txt"
        traversal.write_text("do not serve me", encoding="utf-8")
        bad_alias = alias_for("../secret.txt", self.salt)
        bad_map = self.root / "bad_map.json"
        bad_map.write_text(
            json.dumps({"salt": self.salt, "aliases": {bad_alias: "../secret.txt"}}), encoding="utf-8"
        )
        bad_workbook = self.root / "bad_workbook.csv"
        write_workbook(
            [{"alias": bad_alias, "observation": "", "observer_id": "", "observed_at": ""}], bad_workbook
        )
        context = load_review_context(bad_map, bad_workbook, self.images, labels=self.labels)
        with self.assertRaises(ValueError):
            context.resolve_image(bad_alias)
        report = check_context(context)
        self.assertEqual(report["decision"], "BLOCKED")
        with self.assertRaises(RuntimeError):
            create_app(context)

    # ---- local HTTP interface ----

    def _client(self) -> TestClient:
        return TestClient(create_app(self.context))

    def test_page_exposes_no_paths_labels_or_splits(self) -> None:
        body = self._client().get("/").text
        for leaked in (self.sample_id, "images/", "apple", ".jpg", "split", "train"):
            self.assertNotIn(leaked, body)

    def test_http_flow_saves_observation_and_hides_the_mapping(self) -> None:
        client = self._client()
        self.assertEqual(client.get("/api/next").json()["alias"], self.alias)

        image_response = client.get(f"/image/{self.alias}")
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(image_response.headers["content-type"], "image/jpeg")
        self.assertTrue(image_response.content.startswith(b"\xff\xd8"))
        joined_headers = json.dumps(dict(image_response.headers)).casefold()
        self.assertNotIn("apple", joined_headers)
        self.assertNotIn(self.sample_id.casefold(), joined_headers)

        self.assertEqual(client.get("/image/obs-000000000000").status_code, 404)

        # saving without an observer id is refused, and leaks nothing
        refused = client.post("/api/observation", json={"alias": self.alias, "observation": OBSERVATION})
        self.assertEqual(refused.status_code, 400)
        self.assertNotIn(self.sample_id, refused.text)

        self.assertTrue(client.post("/api/observer", json={"observer_id": "reviewer-7"}).json()["ok"])
        saved = client.post("/api/observation", json={"alias": self.alias, "observation": OBSERVATION})
        self.assertEqual(saved.status_code, 200)

        row = load_annotator_workbook(self.workbook_path)[0]
        self.assertEqual(row["observation"], OBSERVATION)
        self.assertEqual(row["observer_id"], "reviewer-7")

        done = client.get("/api/next").json()
        self.assertTrue(done["done"])
        self.assertIsNone(done["alias"])

    def test_skip_endpoint_advances_after_http_flow(self) -> None:
        client = self._client()
        client.post("/api/observer", json={"observer_id": "reviewer-7"})
        self.assertTrue(client.post("/api/skip", json={"alias": self.alias}).json()["ok"])
        self.assertTrue(client.get("/api/next").json()["done"])
        self.assertEqual(client.post("/api/skip", json={"alias": "obs-unknown"}).status_code, 400)

    # ---- save isolation, progress, restart, unsaved text ----

    def test_save_updates_only_the_matching_alias_and_preserves_the_rest(self) -> None:
        repo = self._make_repo(["apple scab/1.jpg", "apple scab/2.jpg", "apple scab/3.jpg"])
        context, workbook = repo["context"], repo["workbook"]
        target = repo["by_sample"]["apple scab/2.jpg"]

        context.set_observer("reviewer-7")
        context.record(target, OBSERVATION)

        rows = load_annotator_workbook(workbook)
        filled = [row for row in rows if row["observation"]]
        self.assertEqual(len(filled), 1)
        self.assertEqual(filled[0]["alias"], target)
        for row in rows:
            if row["alias"] != target:
                self.assertEqual((row["observation"], row["observer_id"], row["observed_at"]), ("", "", ""))
        # workbook order is preserved (no rows reordered or dropped)
        self.assertEqual([row["alias"] for row in rows], list(repo["aliases"]))

    def test_duplicate_save_is_refused_and_original_text_kept(self) -> None:
        repo = self._make_repo(["apple scab/1.jpg", "apple scab/2.jpg"])
        context, workbook = repo["context"], repo["workbook"]
        first = repo["by_sample"]["apple scab/1.jpg"]
        context.set_observer("reviewer-7")
        context.record(first, OBSERVATION)

        with self.assertRaisesRegex(ValueError, "overwrite"):
            context.record(first, "different later text")
        rows = load_annotator_workbook(workbook)
        self.assertEqual([row["observation"] for row in rows if row["alias"] == first], [OBSERVATION])

    def test_successful_save_persists_the_alias_row_and_agrees_with_progress(self) -> None:
        repo = self._make_repo(["apple scab/1.jpg", "apple scab/2.jpg"])
        context, workbook = repo["context"], repo["workbook"]
        target = repo["by_sample"]["apple scab/2.jpg"]
        context.set_observer("reviewer-7")

        client = TestClient(create_app(context))
        saved = client.post("/api/observation", json={"alias": target, "observation": OBSERVATION})
        self.assertEqual(saved.status_code, 200)

        rows = load_annotator_workbook(workbook)
        filled = [row for row in rows if row["observation"]]
        self.assertEqual([row["alias"] for row in filled], [target])       # correct anonymous row
        self.assertEqual(filled[0]["observer_id"], "reviewer-7")
        self.assertEqual(filled[0]["observed_at"], datetime.date.today().isoformat())
        # on-disk truth and the in-session counters agree after a successful save
        self.assertEqual(context.progress(), {"completed": 1, "remaining": 1, "total": 2})
        self.assertEqual(client.get("/api/progress").json(), context.progress())

    def test_locked_workbook_write_failure_keeps_disk_and_memory_in_step(self) -> None:
        """A locked workbook must never silently mark unwritten text as done.

        ``os.replace`` raises ``PermissionError`` (WinError 32) when another program,
        e.g. Excel, holds the workbook open without delete sharing.
        """
        repo = self._make_repo(["apple scab/1.jpg", "apple scab/2.jpg"])
        context, workbook = repo["context"], repo["workbook"]
        before = workbook.read_bytes()
        target = next(iter(repo["aliases"]))
        context.set_observer("reviewer-7")

        locked = PermissionError(32, "The process cannot access the file because it is being used by another process")
        with mock.patch.object(review_tool, "_atomic_write_workbook", side_effect=locked):
            with self.assertRaises(PermissionError):
                context.record(target, OBSERVATION)

        self.assertEqual(workbook.read_bytes(), before)              # nothing was persisted
        self.assertEqual(context.progress()["completed"], 0)         # memory did not advance either
        self.assertEqual(context.next_alias(), target)               # the image is still waiting
        self.assertTrue(all(row["observation"] == "" for row in load_annotator_workbook(workbook)))

    def test_locked_workbook_reports_a_safe_json_error_over_http(self) -> None:
        repo = self._make_repo(["apple scab/1.jpg"])
        context, workbook = repo["context"], repo["workbook"]
        client = TestClient(create_app(context))
        client.post("/api/observer", json={"observer_id": "reviewer-7"})
        alias = next(iter(repo["aliases"]))

        locked = PermissionError(32, "sharing violation")
        with mock.patch.object(review_tool, "_atomic_write_workbook", side_effect=locked):
            response = client.post("/api/observation", json={"alias": alias, "observation": OBSERVATION})

        # a JSON body (not a plain-text 500), so the UI shows the real reason, not "Could not save..."
        self.assertEqual(response.status_code, 500)
        self.assertIn("application/json", response.headers["content-type"])
        message = response.json()["error"]
        self.assertIn("could not be written", message)
        self.assertIn("Excel", message)
        for leaked in (str(workbook), str(repo["images"]), "apple", ".jpg", "split"):
            self.assertNotIn(leaked, response.text)
        self.assertEqual(client.get("/api/progress").json()["completed"], 0)

    def test_progress_counts_reflect_completed_and_remaining(self) -> None:
        repo = self._make_repo(["apple scab/1.jpg", "apple scab/2.jpg", "apple scab/3.jpg"])
        context = repo["context"]
        self.assertEqual(context.progress(), {"completed": 0, "remaining": 3, "total": 3})
        context.set_observer("reviewer-7")
        context.record(next(iter(repo["aliases"])), OBSERVATION)
        self.assertEqual(context.progress(), {"completed": 1, "remaining": 2, "total": 3})

    def test_progress_endpoint_leaks_nothing_and_matches_context(self) -> None:
        repo = self._make_repo(["apple scab/1.jpg", "apple scab/2.jpg"])
        client = TestClient(create_app(repo["context"]))
        response = client.get("/api/progress")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), repo["context"].progress())
        for leaked in (repo["by_sample"]["apple scab/1.jpg"], "images/", "apple", ".jpg", "split", "train"):
            self.assertNotIn(leaked, response.text)

    def test_restart_preserves_saved_observations_and_resets_skips(self) -> None:
        repo = self._make_repo(["apple scab/1.jpg", "apple scab/2.jpg", "apple scab/3.jpg"])
        context, workbook = repo["context"], repo["workbook"]
        saved = repo["by_sample"]["apple scab/1.jpg"]
        skipped = repo["by_sample"]["apple scab/2.jpg"]

        context.set_observer("reviewer-7")
        context.record(saved, OBSERVATION)
        context.skip(skipped)
        self.assertNotIn(skipped, context.pending())
        self.assertNotIn(saved, context.pending())

        restarted = load_review_context(repo["alias_map"], workbook, repo["images"], labels=self.labels)
        self.assertEqual(restarted.progress()["completed"], 1)
        self.assertNotIn(saved, restarted.pending())          # saved work is never re-presented
        self.assertEqual(restarted.progress()["remaining"], 2)
        self.assertIn(skipped, restarted.pending())           # skips are session-only, not persisted
        row = [r for r in load_annotator_workbook(workbook) if r["alias"] == saved][0]
        self.assertEqual(row["observation"], OBSERVATION)
        self.assertEqual(row["observer_id"], "reviewer-7")

    def test_skip_never_writes_or_alters_the_workbook(self) -> None:
        repo = self._make_repo(["apple scab/1.jpg", "apple scab/2.jpg"])
        context, workbook = repo["context"], repo["workbook"]
        before = workbook.read_bytes()
        context.skip(next(iter(repo["aliases"])))
        self.assertEqual(workbook.read_bytes(), before)       # unsaved text is not silently stored
        self.assertTrue(all(row["observation"] == "" for row in load_annotator_workbook(workbook)))

    def test_page_shows_progress_and_guards_unsaved_text(self) -> None:
        body = self._client().get("/").text
        self.assertIn('id="progress"', body)
        self.assertIn("unsaved text", body)                   # Next warns before discarding typed text
        self.assertIn("/api/progress", body)

    def test_collection_tool_never_writes_review_fields_or_ledger(self) -> None:
        repo = self._make_repo(["apple scab/1.jpg", "apple scab/2.jpg"])
        context, workbook = repo["context"], repo["workbook"]
        context.set_observer("reviewer-7")
        context.record(next(iter(repo["aliases"])), OBSERVATION)
        row = load_annotator_workbook(workbook)[0]
        self.assertEqual(set(row.keys()), set(WORKBOOK_COLUMNS))   # no review_status / reviewer columns
        # the collection tool must not create or touch a review ledger or a paired CSV
        self.assertFalse((workbook.parent / "review_ledger.csv").exists())
        self.assertFalse((workbook.parent / "paired_observations.csv").exists())


class IndependentReviewGateTests(unittest.TestCase):
    """The annotator can never be treated as their own independent reviewer."""

    def _case(self, observer: str, reviewer: str, status: str = "approved") -> tuple:
        alias_map = build_alias_map(["apple scab/1.jpg"], "gate-salt")
        alias = next(iter(alias_map))
        sample_id = alias_map[alias]
        rows = [{"alias": alias, "observation": "brown spots", "observer_id": observer, "observed_at": "2026-01-01"}]
        ledger = {
            sample_id: {
                "sample_id": sample_id, "review_status": status,
                "reviewer": reviewer, "reviewed_at": "2026-01-02", "notes": "",
            }
        }
        return rows, alias_map, ledger

    def test_self_approval_is_rejected(self) -> None:
        rows, alias_map, ledger = self._case(observer="annotator-1", reviewer="annotator-1")
        with self.assertRaisesRegex(ValueError, "independent"):
            require_independent_review(rows, alias_map, ledger)

    def test_reviewer_matching_is_case_insensitive(self) -> None:
        rows, alias_map, ledger = self._case(observer="Annotator-1", reviewer="annotator-1")
        with self.assertRaisesRegex(ValueError, "independent"):
            require_independent_review(rows, alias_map, ledger)

    def test_distinct_reviewer_is_accepted(self) -> None:
        rows, alias_map, ledger = self._case(observer="annotator-1", reviewer="reviewer-2")
        self.assertEqual(require_independent_review(rows, alias_map, ledger), {"independent_reviews": 1})

    def test_approved_row_without_a_named_reviewer_is_rejected(self) -> None:
        rows, alias_map, ledger = self._case(observer="annotator-1", reviewer="")
        with self.assertRaisesRegex(ValueError, "named reviewer"):
            require_independent_review(rows, alias_map, ledger)

    def test_unapproved_rows_are_left_to_the_other_gates(self) -> None:
        rows, alias_map, ledger = self._case(observer="annotator-1", reviewer="annotator-1", status="pending")
        self.assertEqual(require_independent_review(rows, alias_map, ledger), {"independent_reviews": 0})


class CliBehaviorTests(unittest.TestCase):
    """`python -m scripts.review_annotations ...` start-up behaviour.

    The CLI is exercised against a tiny temporary repo so these tests never touch the
    real sealed map or the 17,870-row workbook. ``uvicorn.run`` is patched so no real
    socket is served; the blocking behaviour in production comes from uvicorn itself.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.images = root / "images"
        class_dir = self.images / "apple scab"
        class_dir.mkdir(parents=True)
        sample_id = "apple scab/1.jpg"
        Image.new("RGB", (8, 8), (20, 120, 40)).save(class_dir / "1.jpg")
        salt = "cli-salt"
        alias = alias_for(sample_id, salt)
        self.alias_map_path = root / "alias_map.json"
        self.alias_map_path.write_text(
            json.dumps({"salt": salt, "aliases": {alias: sample_id}}), encoding="utf-8"
        )
        self.workbook_path = root / "workbook.csv"
        write_workbook(
            [{"alias": alias, "observation": "", "observer_id": "", "observed_at": ""}],
            self.workbook_path,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _argv(self, *extra: str) -> list[str]:
        return [
            "--alias-map", str(self.alias_map_path),
            "--workbook", str(self.workbook_path),
            "--images-dir", str(self.images),
            *extra,
        ]

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    def test_check_flag_prints_report_and_exits_without_serving(self) -> None:
        import uvicorn

        served: list = []
        original = uvicorn.run
        uvicorn.run = lambda *args, **kwargs: served.append((args, kwargs))
        try:
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                with self.assertRaises(SystemExit) as ctx:
                    main(self._argv("--check"))
        finally:
            uvicorn.run = original

        self.assertEqual(ctx.exception.code, 0)
        self.assertEqual(served, [])                                  # --check never starts a server
        self.assertIn('"decision": "READY"', captured.getvalue())

    def test_server_is_started_on_the_requested_port(self) -> None:
        import uvicorn

        port = self._free_port()
        recorded: dict = {}
        original = uvicorn.run
        uvicorn.run = lambda app, **kwargs: recorded.update(kwargs)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                main(self._argv("--port", str(port)))
        finally:
            uvicorn.run = original

        # the app is handed to uvicorn, which blocks (keeps running) until Ctrl+C
        self.assertEqual(recorded.get("host"), "127.0.0.1")
        self.assertEqual(recorded.get("port"), port)

    def test_busy_port_is_reported_clearly_and_no_server_is_started(self) -> None:
        import uvicorn

        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        busy_port = holder.getsockname()[1]
        served: list = []
        original = uvicorn.run
        uvicorn.run = lambda *args, **kwargs: served.append((args, kwargs))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    main(self._argv("--port", str(busy_port)))
        finally:
            uvicorn.run = original
            holder.close()

        self.assertEqual(served, [])                                  # no misleading "ready" banner path
        self.assertIn(str(busy_port), str(ctx.exception.code))
        self.assertIn("already in use", str(ctx.exception.code))


if __name__ == "__main__":
    unittest.main()
