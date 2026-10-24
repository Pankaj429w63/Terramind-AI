"""Local, blinded plant-image review tool.

A human reviewer sees one plant image at a time behind an opaque alias and types a
genuine visual observation in plain language. The tool deliberately shows only the
alias, the image, the observation box, and the save/next controls: image paths, class
names, model predictions, and split membership are never sent to the browser.

The sealed alias -> sample_id map and the dataset root stay inside this local process.
Images are served by alias only, and every observation is validated with the same
helpers used by the rest of the annotation workflow before it is written into the
existing blinded workbook (``alias,observation,observer_id,observed_at``).

Nothing here is fabricated. The reviewer supplies their own observer id (entered once
and never invented); the recorded timestamp is the real date on which the observation
is saved; existing observations are never overwritten; and the official dataset splits
and the sealed alias map are opened read-only.

This tool only *collects* observations: it never generates text, never approves a row,
and never writes the review ledger or the paired training CSV. Independent review is a
separate step (see ``outputs/research/annotation_specification.md`` §8): a different
person adjudicates each observation into ``data/annotation/review_ledger_template.csv``,
and ``scripts/validate_annotations.py`` refuses to emit training data when an
observation is approved by its own annotator (``require_independent_review``).

Local usage (Windows CMD):
    python -m scripts.review_annotations --check      :: validate the mapping and exit
    python -m scripts.review_annotations              :: serve on http://127.0.0.1:8040
    python -m scripts.review_annotations --port 9000  :: serve on another local port
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import socket
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from ml.fusion.annotation import (
    load_alias_map,
    load_annotator_workbook,
    validate_observed_at,
    write_workbook,
)
from ml.fusion.data import PLACEHOLDER_PATTERN, PLACEHOLDER_VALUES, validate_text_not_label_derived
from ml.models.class_labels import load_class_labels

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "data" / "plantwild" / "plantwild"
DEFAULT_IMAGES_DIR = DEFAULT_DATASET_ROOT / "images"
DEFAULT_ALIAS_MAP = ROOT / "outputs" / "annotation" / "sealed" / "alias_map.json"
DEFAULT_WORKBOOK = ROOT / "outputs" / "annotation" / "annotator_workbook.csv"

_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

logger = logging.getLogger(__name__)


def _is_placeholder(value: str) -> bool:
    stripped = (value or "").strip()
    if not stripped:
        return True
    if stripped.casefold().strip(" .") in PLACEHOLDER_VALUES:
        return True
    return bool(PLACEHOLDER_PATTERN.search(stripped))


def _media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.casefold(), "application/octet-stream")


def _port_is_available(host: str, port: int) -> bool:
    """True when this process can bind the local port.

    A busy port must be detected before the tool claims to be "ready": otherwise
    ``uvicorn.run`` raises ``SystemExit(1)`` from inside its own runner *after* the
    startup banner has already been printed, which looks like the server exiting at
    startup. No ``SO_REUSEADDR`` is set so a genuinely busy port is reported.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _atomic_write_workbook(rows: Sequence[Mapping[str, str]], path: Path) -> Path:
    """Write the workbook so a failure can never truncate or corrupt existing annotations."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent,
        prefix=f"{path.name}.", suffix=".partial", delete=False,
    )
    handle.close()
    try:
        write_workbook(rows, Path(handle.name))
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return path


@dataclass
class ReviewContext:
    """Server-side state: the sealed mapping never leaves this object."""

    alias_map: Mapping[str, str]
    workbook_path: Path
    images_dir: Path
    rows: list[dict[str, str]]
    order: list[str]
    labels: Mapping[int, str] = field(default_factory=dict)
    observer_id: str = ""
    skipped: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def resolve_image(self, alias: str) -> Path:
        """Map an alias to its image path without ever revealing the path to a caller."""
        sample_id = self.alias_map.get(alias)
        if not sample_id:
            raise KeyError("unknown alias")
        root = self.images_dir.resolve()
        candidate = (root / sample_id).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("alias resolves outside the image directory")
        if not candidate.is_file():
            raise FileNotFoundError("image file is missing for alias")
        return candidate

    def _row(self, alias: str) -> dict[str, str]:
        for row in self.rows:
            if row.get("alias") == alias:
                return row
        raise KeyError("alias is not present in the workbook")

    def is_filled(self, alias: str) -> bool:
        return bool((self._row(alias).get("observation") or "").strip())

    def pending(self) -> list[str]:
        return [alias for alias in self.order if not self.is_filled(alias) and alias not in self.skipped]

    def progress(self) -> dict[str, int]:
        """Completed/remaining counts only: no paths, sample ids, labels, or splits."""
        completed = sum(1 for alias in self.order if self.is_filled(alias))
        return {"completed": completed, "remaining": len(self.order) - completed, "total": len(self.order)}

    def next_alias(self) -> str | None:
        pending = self.pending()
        return pending[0] if pending else None

    def set_observer(self, observer_id: str) -> str:
        """Record the reviewer's own identity; never invent one."""
        value = (observer_id or "").strip()
        if _is_placeholder(value):
            raise ValueError("enter your own observer id; placeholders such as 'unknown' are rejected")
        self.observer_id = value
        return value

    def skip(self, alias: str) -> None:
        self._row(alias)
        self.skipped.add(alias)

    def record(self, alias: str, observation: str) -> str:
        """Validate and save one genuine observation; existing observations are never replaced.

        The workbook is written *before* the in-memory row is updated. If the durable write
        fails (for example the workbook is locked by another program such as Excel) the
        session keeps the image pending instead of silently treating unwritten text as done.
        """
        text = (observation or "").strip()
        if not text:
            raise ValueError("observation is empty")
        if not self.observer_id:
            raise ValueError("no observer id set for this session")
        with self._lock:
            row = self._row(alias)
            if (row.get("observation") or "").strip():
                raise ValueError("this image already has an observation; refusing to overwrite it")
            validate_text_not_label_derived(text, self.labels)
            observed_at = datetime.date.today().isoformat()
            validate_observed_at(observed_at, alias)
            filled = {
                **row,
                "observation": text,
                "observer_id": self.observer_id,
                "observed_at": observed_at,
            }
            rows = [filled if candidate is row else candidate for candidate in self.rows]
            _atomic_write_workbook(rows, self.workbook_path)
            row.update(filled)
        return alias


def _default_labels(images_dir: Path) -> dict[int, str]:
    classes_path = Path(images_dir).resolve().parent / "classes.txt"
    return load_class_labels(classes_path) if classes_path.is_file() else {}


def load_review_context(
    alias_map_path: Path = DEFAULT_ALIAS_MAP,
    workbook_path: Path = DEFAULT_WORKBOOK,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    *,
    labels: Mapping[int, str] | None = None,
) -> ReviewContext:
    """Load the sealed map and workbook; the workbook is validated for blinding on load."""
    alias_map, _salt = load_alias_map(alias_map_path)
    rows = load_annotator_workbook(workbook_path)
    order = [row["alias"] for row in rows]
    unknown = [alias for alias in order if alias not in alias_map]
    if unknown:
        raise ValueError(f"{len(unknown)} workbook alias(es) are absent from the sealed mapping")
    resolved_labels = dict(labels) if labels is not None else _default_labels(images_dir)
    return ReviewContext(
        alias_map=alias_map,
        workbook_path=Path(workbook_path),
        images_dir=Path(images_dir),
        rows=rows,
        order=order,
        labels=resolved_labels,
    )


def check_context(context: ReviewContext) -> dict[str, object]:
    """Report readiness without ever repeating paths, sample ids, or class labels."""
    resolvable = 0
    unresolved = 0
    for alias in context.order:
        try:
            context.resolve_image(alias)
            resolvable += 1
        except (KeyError, ValueError, FileNotFoundError):
            unresolved += 1
    filled = sum(1 for alias in context.order if context.is_filled(alias))
    report: dict[str, object] = {
        "workbook_rows": len(context.order),
        "aliases_defined": len(context.alias_map),
        "resolvable_images": resolvable,
        "unresolved_aliases": unresolved,
        "filled_rows": filled,
        "missing_rows": len(context.order) - filled,
        "decision": "READY" if unresolved == 0 else "BLOCKED",
    }
    report["blocker"] = (
        f"{unresolved} alias(es) do not resolve to an image; fix the sealed map before reviewing"
        if unresolved
        else f"{len(context.order) - filled} of {len(context.order)} images still need an observation"
    )
    return report


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plant image review</title>
<style>
  :root { color-scheme: light; }
  body { margin: 0; font-family: system-ui, sans-serif; background: #f4f6f4; color: #1c2b22; }
  main { max-width: 720px; margin: 0 auto; padding: 1.5rem; }
  h1 { font-size: 1.1rem; font-weight: 600; }
  #gate, #review { background: #fff; border: 1px solid #d8e0d8; border-radius: 10px; padding: 1.25rem; }
  #alias { font-family: ui-monospace, monospace; font-size: 1.2rem; margin: 0 0 .75rem; }
  #progress { margin: 0 0 .75rem; font-size: .85rem; color: #40574a; }
  #photo { display: block; width: 100%; max-height: 60vh; object-fit: contain; background: #eef2ee; border-radius: 8px; }
  label { display: block; font-size: .85rem; margin: 1rem 0 .35rem; color: #40574a; }
  input, textarea { width: 100%; box-sizing: border-box; font: inherit; padding: .6rem; border: 1px solid #c3cfc3; border-radius: 8px; }
  textarea { min-height: 6rem; resize: vertical; }
  .row { display: flex; gap: .75rem; margin-top: 1rem; }
  button { font: inherit; padding: .6rem 1.1rem; border-radius: 8px; border: 1px solid #2f6b45; background: #2f6b45; color: #fff; cursor: pointer; }
  button.secondary { background: #fff; color: #2f6b45; }
  button:disabled { opacity: .5; cursor: default; }
  #status { margin-top: .9rem; font-size: .9rem; color: #40574a; min-height: 1.2rem; }
</style>
</head>
<body>
<main>
  <h1>Plant image review</h1>
  <section id="gate">
    <label for="observer">Your observer id</label>
    <input id="observer" autocomplete="off" placeholder="Enter the id you use for annotation">
    <div class="row"><button id="start">Start review</button></div>
    <p id="status"></p>
  </section>
  <section id="review" hidden>
    <p id="alias"></p>
    <p id="progress"></p>
    <img id="photo" alt="Plant image under review">
    <label for="obs">Your observation</label>
    <textarea id="obs" placeholder="Describe what you actually see in the plant image, in plain language."></textarea>
    <div class="row">
      <button id="save">Save</button>
      <button id="next" class="secondary">Next</button>
    </div>
    <p id="review-status"></p>
  </section>
</main>
<script>
const el = (id) => document.getElementById(id);
let current = null;
let observer = "";

async function api(path, body) {
  const res = await fetch(path, body ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  } : {});
  return res;
}

el("start").addEventListener("click", async () => {
  const value = el("observer").value.trim();
  if (!value) { el("status").textContent = "Enter your observer id to begin."; return; }
  const res = await api("/api/observer", { observer_id: value });
  if (!res.ok) { el("status").textContent = "Could not start: use a real observer id."; return; }
  observer = value;
  el("gate").hidden = true;
  el("review").hidden = false;
  await loadNext();
});

async function refreshProgress() {
  const res = await api("/api/progress");
  if (!res.ok) return;
  const data = await res.json();
  el("progress").textContent = "Completed: " + data.completed + " · Remaining: " + data.remaining;
}

async function loadNext() {
  el("review-status").textContent = "";
  await refreshProgress();
  const res = await api("/api/next");
  const data = await res.json();
  if (!data.alias) {
    current = null;
    el("alias").textContent = "";
    el("photo").removeAttribute("src");
    el("obs").value = "";
    el("save").disabled = true;
    el("next").disabled = true;
    el("review-status").textContent = "Review complete: no images are waiting for an observation.";
    return;
  }
  current = data.alias;
  el("alias").textContent = current;
  el("photo").src = "/image/" + encodeURIComponent(current);
  el("obs").value = "";
  el("save").disabled = false;
  el("next").disabled = false;
  el("obs").focus();
}

el("save").addEventListener("click", async () => {
  if (!current) return;
  const text = el("obs").value.trim();
  if (!text) { el("review-status").textContent = "Write an observation before saving."; return; }
  const res = await api("/api/observation", { alias: current, observation: text });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    el("review-status").textContent = data.error || "Could not save this observation.";
    return;
  }
  await loadNext();
});

el("next").addEventListener("click", async () => {
  if (!current) return;
  if (el("obs").value.trim() && !window.confirm("This image has unsaved text. Skip it and discard that text?")) {
    return;
  }
  await api("/api/skip", { alias: current });
  await loadNext();
});
</script>
</body>
</html>
"""


class _ObserverBody(BaseModel):
    observer_id: str


class _ObservationBody(BaseModel):
    alias: str
    observation: str


class _AliasBody(BaseModel):
    alias: str


def create_app(context: ReviewContext) -> FastAPI:
    """Build the local FastAPI app. Images are served by alias only; paths never leave."""
    report = check_context(context)
    if report["decision"] != "READY":
        raise RuntimeError(f"review tool cannot start: {report['blocker']}")

    app = FastAPI(title="TerraMind blinded review", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    def page() -> HTMLResponse:
        return HTMLResponse(_PAGE)

    @app.post("/api/observer")
    def set_observer(body: _ObserverBody) -> JSONResponse:
        try:
            context.set_observer(body.observer_id)
        except ValueError as error:
            return JSONResponse({"error": str(error)}, status_code=400)
        return JSONResponse({"ok": True})

    @app.get("/api/next")
    def next_image() -> JSONResponse:
        alias = context.next_alias()
        return JSONResponse({"alias": alias, "done": alias is None})

    @app.get("/api/progress")
    def progress() -> JSONResponse:
        return JSONResponse(context.progress())

    @app.post("/api/skip")
    def skip(body: _AliasBody) -> JSONResponse:
        try:
            context.skip(body.alias)
        except KeyError:
            return JSONResponse({"error": "unknown alias"}, status_code=400)
        return JSONResponse({"ok": True})

    @app.post("/api/observation")
    def save_observation(body: _ObservationBody) -> JSONResponse:
        try:
            context.record(body.alias, body.observation)
        except KeyError:
            return JSONResponse({"error": "unknown alias"}, status_code=400)
        except ValueError as error:
            return JSONResponse({"error": str(error)}, status_code=400)
        except OSError:
            logger.exception("could not write the annotation workbook for one observation")
            return JSONResponse(
                {
                    "error": "The annotation workbook could not be written. Close it in any other "
                             "program (for example Excel) and save again."
                },
                status_code=500,
            )
        return JSONResponse({"ok": True})

    @app.get("/image/{alias}")
    def image(alias: str) -> Response:
        try:
            path = context.resolve_image(alias)
        except (KeyError, ValueError, FileNotFoundError):
            raise HTTPException(status_code=404, detail="unknown image")
        data = path.read_bytes()
        return Response(
            content=data,
            media_type=_media_type(path),
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.middleware("http")
    async def no_store(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    return app


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Local, blinded plant-image review tool. Saves observations into the existing "
                    "annotator workbook; never reveals image paths, class labels, or splits.",
    )
    parser.add_argument("--alias-map", type=Path, default=DEFAULT_ALIAS_MAP)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--classes", type=Path, default=None, help="classes.txt used for label-leak checks")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (keep it local)")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--check", action="store_true", help="validate the mapping/workbook and exit without serving")
    args = parser.parse_args(argv)

    labels = load_class_labels(args.classes) if args.classes else None
    context = load_review_context(args.alias_map, args.workbook, args.images_dir, labels=labels)
    report = check_context(context)
    print(json.dumps(report, indent=2), flush=True)
    if args.check:
        raise SystemExit(0 if report["decision"] == "READY" else 1)
    if report["decision"] != "READY":
        raise SystemExit(f"review tool cannot start: {report['blocker']}")

    import uvicorn

    app = create_app(context)
    if not _port_is_available(args.host, args.port):
        raise SystemExit(
            f"cannot start the review server: port {args.port} on {args.host} is already in use "
            "(a previous review server may still be running). Stop that process, or start this "
            f"tool on another port, e.g. --port {args.port + 1}."
        )
    print(
        f"\nBlinded review tool ready — open http://{args.host}:{args.port} in a browser.\n"
        "Only the alias, the image, your observation box, and save/next are shown.\n"
        "Press Ctrl+C to stop.\n",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
