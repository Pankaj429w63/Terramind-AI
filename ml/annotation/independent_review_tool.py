"""Dedicated independent plant-observation review server and UI.

Presents plant images alongside annotator observations and metadata for an
independent reviewer to adjudicate (approve, reject, dispute) into review_ledger.csv.

Enforces strict research-integrity gates:
1. Displays actual plant image, observation text, sample ID, and observer ID.
2. Strictly prohibits self-review (reviewer cannot equal observer_id).
3. Requires non-empty reviewer identity and informative notes for rejections/disputes.
4. Records real calendar date (datetime.date.today().isoformat()); never fabricates dates.
5. Preserves pending status until explicitly adjudicated by a real reviewer.
6. Writes atomically to review_ledger.csv conforming to REVIEW_COLUMNS schema.
"""
from __future__ import annotations

import argparse
import csv
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
    REVIEW_COLUMNS,
    REVIEW_STATUSES,
    load_alias_map,
    load_annotator_workbook,
    load_review_ledger,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "data" / "plantwild" / "plantwild"
DEFAULT_IMAGES_DIR = DEFAULT_DATASET_ROOT / "images"
DEFAULT_ALIAS_MAP = ROOT / "outputs" / "annotation" / "pilot" / "sealed" / "alias_map.json"
DEFAULT_WORKBOOK = ROOT / "outputs" / "annotation" / "pilot" / "annotator_workbook_reannotation.csv"
DEFAULT_LEDGER = ROOT / "outputs" / "annotation" / "pilot" / "review_ledger.csv"

_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

logger = logging.getLogger(__name__)


def _port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
            return True
        except OSError:
            return False


@dataclass
class ReviewItem:
    index: int
    sample_id: str
    alias: str
    observation: str
    observer_id: str
    observed_at: str
    review_status: str
    reviewer: str
    reviewed_at: str
    notes: str


@dataclass
class IndependentReviewContext:
    alias_map_path: Path
    workbook_path: Path
    ledger_path: Path
    images_dir: Path
    alias_map: dict[str, str] = field(default_factory=dict)
    items: list[ReviewItem] = field(default_factory=list)
    items_by_sample_id: dict[str, ReviewItem] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reload(self) -> None:
        with self.lock:
            alias_map, _ = load_alias_map(self.alias_map_path)
            workbook_rows = load_annotator_workbook(self.workbook_path)
            ledger = load_review_ledger(self.ledger_path)

            wb_by_alias = {row["alias"]: row for row in workbook_rows}

            # Build items ordered by sample_id for deterministic review order
            items: list[ReviewItem] = []
            items_by_id: dict[str, ReviewItem] = {}

            # Map sample_id back to alias
            sample_to_alias = {sample_id: alias for alias, sample_id in alias_map.items()}

            for idx, sample_id in enumerate(sorted(sample_to_alias.keys()), 1):
                alias = sample_to_alias[sample_id]
                wb_row = wb_by_alias.get(alias, {})
                ledger_row = ledger.get(sample_id, {})

                item = ReviewItem(
                    index=idx,
                    sample_id=sample_id,
                    alias=alias,
                    observation=wb_row.get("observation", ""),
                    observer_id=wb_row.get("observer_id", ""),
                    observed_at=wb_row.get("observed_at", ""),
                    review_status=ledger_row.get("review_status", "pending"),
                    reviewer=ledger_row.get("reviewer", ""),
                    reviewed_at=ledger_row.get("reviewed_at", ""),
                    notes=ledger_row.get("notes", ""),
                )
                items.append(item)
                items_by_id[sample_id] = item

            self.alias_map = alias_map
            self.items = items
            self.items_by_sample_id = items_by_id

    def progress(self) -> dict[str, int]:
        with self.lock:
            counts = {"pending": 0, "approved": 0, "rejected": 0, "disputed": 0}
            for it in self.items:
                st = it.review_status.casefold()
                counts[st] = counts.get(st, 0) + 1
            total = len(self.items)
            reviewed = total - counts.get("pending", 0)
            return {
                "total": total,
                "reviewed": reviewed,
                "pending": counts.get("pending", 0),
                "approved": counts.get("approved", 0),
                "rejected": counts.get("rejected", 0),
                "disputed": counts.get("disputed", 0),
            }

    def record_decision(
        self,
        sample_id: str,
        review_status: str,
        reviewer: str,
        notes: str = "",
    ) -> ReviewItem:
        status_clean = (review_status or "").strip().casefold()
        if status_clean not in REVIEW_STATUSES or status_clean == "pending":
            raise ValueError(
                f"Invalid review decision {review_status!r}; must be one of: 'approved', 'rejected', 'disputed'"
            )

        reviewer_clean = (reviewer or "").strip()
        if not reviewer_clean:
            raise ValueError("Reviewer identity cannot be empty. Please provide your reviewer ID.")

        notes_clean = (notes or "").strip()
        if status_clean in ("rejected", "disputed") and not notes_clean:
            raise ValueError(f"Review decision '{status_clean}' requires explanatory notes.")

        with self.lock:
            item = self.items_by_sample_id.get(sample_id)
            if item is None:
                raise KeyError(f"Sample ID {sample_id!r} not found in review queue")

            # Self-review prevention gate
            observer = (item.observer_id or "").strip()
            if observer and reviewer_clean.casefold() == observer.casefold():
                raise ValueError(
                    f"Self-review prohibited: Reviewer '{reviewer_clean}' cannot review observation "
                    f"authored by observer '{observer}'. Independent review requires a distinct person."
                )

            today_iso = datetime.date.today().isoformat()

            # Update in-memory state
            item.review_status = status_clean
            item.reviewer = reviewer_clean
            item.reviewed_at = today_iso
            item.notes = notes_clean

            # Persist atomically to review_ledger.csv
            self._save_ledger()
            return item

    def _save_ledger(self) -> None:
        """Atomic write to review_ledger.csv preserving schema and row order."""
        rows = []
        for it in self.items:
            rows.append({
                "sample_id": it.sample_id,
                "review_status": it.review_status,
                "reviewer": it.reviewer,
                "reviewed_at": it.reviewed_at,
                "notes": it.notes,
            })

        parent = self.ledger_path.parent
        parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile("w", dir=parent, delete=False, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            tmp_name = handle.name

        os.replace(tmp_name, self.ledger_path)

    def resolve_image(self, sample_id: str) -> Path:
        rel = Path(sample_id)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe image path: {sample_id}")
        resolved = (self.images_dir / rel).resolve()
        if not resolved.is_relative_to(self.images_dir.resolve()):
            raise ValueError(f"Image path escapes images directory: {sample_id}")
        if not resolved.is_file():
            raise FileNotFoundError(f"Image file not found: {sample_id}")
        return resolved


def load_independent_review_context(
    alias_map_path: Path,
    workbook_path: Path,
    ledger_path: Path,
    images_dir: Path,
) -> IndependentReviewContext:
    ctx = IndependentReviewContext(
        alias_map_path=Path(alias_map_path).resolve(),
        workbook_path=Path(workbook_path).resolve(),
        ledger_path=Path(ledger_path).resolve(),
        images_dir=Path(images_dir).resolve(),
    )
    ctx.reload()
    return ctx


def check_review_context(context: IndependentReviewContext) -> dict[str, object]:
    progress = context.progress()
    unresolved_images = []
    missing_observations = []

    for it in context.items:
        try:
            context.resolve_image(it.sample_id)
        except Exception:
            unresolved_images.append(it.sample_id)
        if not it.observation.strip():
            missing_observations.append(it.sample_id)

    ready = (len(unresolved_images) == 0 and len(missing_observations) == 0)
    return {
        "total_items": progress["total"],
        "reviewed_items": progress["reviewed"],
        "pending_items": progress["pending"],
        "approved_items": progress["approved"],
        "rejected_items": progress["rejected"],
        "disputed_items": progress["disputed"],
        "unresolved_images": len(unresolved_images),
        "missing_observations": len(missing_observations),
        "decision": "READY" if ready else "BLOCKED",
        "blocker": (
            None if ready else
            f"{len(unresolved_images)} missing image(s) or {len(missing_observations)} missing observation(s)"
        ),
    }


class _DecisionBody(BaseModel):
    sample_id: str
    review_status: str
    reviewer: str
    notes: str = ""


_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TerraMind AI — Independent Annotation Review Tool</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --success: #10b981;
      --danger: #ef4444;
      --warning: #f59e0b;
      --font: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: #020617;
      border-bottom: 1px solid var(--card-border);
      padding: 1rem 1.5rem;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
    }
    .header-title h1 { font-size: 1.25rem; font-weight: 700; color: #38bdf8; }
    .header-title p { font-size: 0.85rem; color: var(--text-muted); }
    .progress-bar-container {
      display: flex;
      align-items: center;
      gap: 1rem;
      flex-grow: 1;
      max-width: 480px;
    }
    .progress-track {
      flex-grow: 1;
      height: 10px;
      background: #334155;
      border-radius: 5px;
      overflow: hidden;
      display: flex;
    }
    .prog-seg { height: 100%; transition: width 0.3s; }
    .prog-approved { background: var(--success); }
    .prog-rejected { background: var(--danger); }
    .prog-disputed { background: var(--warning); }
    .reviewer-session {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      background: var(--card-bg);
      padding: 0.5rem 0.75rem;
      border-radius: 6px;
      border: 1px solid var(--card-border);
    }
    .reviewer-session label { font-size: 0.8rem; color: var(--text-muted); }
    .reviewer-session input {
      background: #0f172a;
      border: 1px solid #475569;
      color: var(--text);
      padding: 0.3rem 0.5rem;
      border-radius: 4px;
      font-size: 0.85rem;
      width: 170px;
    }
    main {
      flex-grow: 1;
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 1.5rem;
      padding: 1.5rem;
      max-width: 1600px;
      margin: 0 auto;
      width: 100%;
    }
    @media (max-width: 960px) {
      main { grid-template-columns: 1fr; }
    }
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .image-viewer {
      flex-grow: 1;
      min-height: 400px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #020617;
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid #1e293b;
    }
    .image-viewer img {
      max-width: 100%;
      max-height: 600px;
      object-fit: contain;
    }
    .meta-box {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
      background: #0f172a;
      padding: 0.75rem;
      border-radius: 6px;
      font-size: 0.85rem;
    }
    .meta-label { color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .meta-val { font-weight: 600; word-break: break-all; }
    .observer-badge {
      grid-column: 1 / -1;
      padding: 0.5rem 0.75rem;
      background: #1e3a8a;
      border: 1px solid #3b82f6;
      border-radius: 4px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .self-review-warn {
      background: #7f1d1d !important;
      border-color: #ef4444 !important;
      color: #fca5a5 !important;
      font-weight: 700;
    }
    .obs-box {
      background: #020617;
      border-left: 4px solid var(--primary);
      padding: 1rem;
      border-radius: 0 6px 6px 0;
      font-size: 1.05rem;
      line-height: 1.6;
    }
    .guideline-box {
      background: #0f172a;
      border: 1px solid #334155;
      padding: 0.75rem 1rem;
      border-radius: 6px;
      font-size: 0.8rem;
      color: var(--text-muted);
    }
    .guideline-box ul { margin-left: 1.25rem; margin-top: 0.25rem; }
    .notes-input {
      width: 100%;
      background: #0f172a;
      border: 1px solid #475569;
      color: var(--text);
      padding: 0.75rem;
      border-radius: 6px;
      font-family: inherit;
      font-size: 0.9rem;
      resize: vertical;
      min-height: 70px;
    }
    .action-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.75rem;
    }
    button {
      cursor: pointer;
      font-family: inherit;
      font-weight: 600;
      border-radius: 6px;
      padding: 0.75rem 1rem;
      border: none;
      transition: opacity 0.2s, transform 0.1s;
    }
    button:active { transform: scale(0.98); }
    button:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-approve { background: var(--success); color: #022c22; }
    .btn-reject { background: var(--danger); color: #450a0a; }
    .btn-dispute { background: var(--warning); color: #451a03; }
    .nav-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.5rem;
      padding-top: 0.5rem;
      border-top: 1px solid var(--card-border);
    }
    .btn-nav { background: #334155; color: var(--text); padding: 0.5rem 1rem; font-size: 0.85rem; }
    .status-pill {
      display: inline-block;
      padding: 0.2rem 0.6rem;
      border-radius: 12px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
    }
    .pill-pending { background: #475569; color: #f1f5f9; }
    .pill-approved { background: var(--success); color: #022c22; }
    .pill-rejected { background: var(--danger); color: #450a0a; }
    .pill-disputed { background: var(--warning); color: #451a03; }
    #toast {
      position: fixed;
      bottom: 1.5rem;
      right: 1.5rem;
      padding: 0.75rem 1.25rem;
      border-radius: 6px;
      font-size: 0.9rem;
      font-weight: 600;
      display: none;
      z-index: 100;
    }
    .toast-success { background: #065f46; color: #a7f3d0; border: 1px solid #10b981; }
    .toast-error { background: #991b1b; color: #fecaca; border: 1px solid #ef4444; }
  </style>
</head>
<body>
  <header>
    <div class="header-title">
      <h1>TerraMind AI — Independent Review</h1>
      <p>Blinded Plant Observation Adjudication Gateway</p>
    </div>
    <div class="progress-bar-container">
      <div style="font-size: 0.85rem; font-weight: 600;" id="prog-text">0 / 178</div>
      <div class="progress-track">
        <div class="prog-seg prog-approved" id="seg-approved" style="width: 0%"></div>
        <div class="prog-seg prog-rejected" id="seg-rejected" style="width: 0%"></div>
        <div class="prog-seg prog-disputed" id="seg-disputed" style="width: 0%"></div>
      </div>
      <div style="font-size: 0.75rem; color: var(--text-muted);" id="prog-details">A: 0 | R: 0 | D: 0</div>
    </div>
    <div class="reviewer-session">
      <label for="reviewer-id-input">Reviewer ID:</label>
      <input type="text" id="reviewer-id-input" placeholder="e.g. reviewer_auditor_01" autocomplete="off">
    </div>
  </header>

  <main>
    <div class="panel">
      <div class="image-viewer">
        <img id="leaf-image" src="" alt="Plant leaf image">
      </div>
      <div class="meta-box">
        <div>
          <div class="meta-label">Sample ID</div>
          <div class="meta-val" id="meta-sample-id">—</div>
        </div>
        <div>
          <div class="meta-label">Blinded Alias</div>
          <div class="meta-val" id="meta-alias">—</div>
        </div>
        <div>
          <div class="meta-label">Observation Date</div>
          <div class="meta-val" id="meta-observed-at">—</div>
        </div>
        <div>
          <div class="meta-label">Current Ledger Status</div>
          <div class="meta-val" id="meta-status"><span class="status-pill pill-pending">pending</span></div>
        </div>
        <div class="observer-badge" id="observer-banner">
          <div>
            <span style="font-size: 0.75rem; text-transform: uppercase; opacity: 0.8;">Recorded Observer:</span>
            <strong id="meta-observer-id" style="font-size: 0.95rem; margin-left: 0.5rem;">—</strong>
          </div>
          <div id="self-review-indicator" style="font-size: 0.75rem;">Independent Reviewer Required</div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div>
        <div class="meta-label" style="margin-bottom: 0.5rem;">Human Visual Observation</div>
        <div class="obs-box" id="observation-text">Loading observation...</div>
      </div>

      <div class="guideline-box">
        <strong>Review Adjudication Checklist:</strong>
        <ul>
          <li><strong>Factual accuracy:</strong> Description matches visible foliage in the photo.</li>
          <li><strong>Diagnostic neutrality:</strong> No disease names, pathogen claims, or clinical diagnosis.</li>
          <li><strong>Objective vocabulary:</strong> Avoid subjective terms ("healthy", "sick", "severity").</li>
          <li><strong>Independence:</strong> You must not be the annotator who wrote this observation.</li>
        </ul>
      </div>

      <div>
        <label for="review-notes" class="meta-label" style="display: block; margin-bottom: 0.35rem;">Review Notes / Audit Rationale</label>
        <textarea id="review-notes" class="notes-input" placeholder="Enter reason for approval or rejection/dispute rationale..."></textarea>
      </div>

      <div class="action-row">
        <button type="button" class="btn-approve" id="btn-approve" onclick="submitDecision('approved')">Approve (A)</button>
        <button type="button" class="btn-reject" id="btn-reject" onclick="submitDecision('rejected')">Reject (R)</button>
        <button type="button" class="btn-dispute" id="btn-dispute" onclick="submitDecision('disputed')">Dispute (D)</button>
      </div>

      <div class="nav-bar">
        <button type="button" class="btn-nav" id="btn-prev" onclick="nav(-1)">&larr; Previous</button>
        <div style="font-size: 0.85rem; color: var(--text-muted);">
          Item <span id="cur-idx">1</span> of <span id="tot-idx">178</span>
        </div>
        <button type="button" class="btn-nav" id="btn-next" onclick="nav(1)">Next &rarr;</button>
        <button type="button" class="btn-nav" style="background: #1e3a8a;" onclick="jumpNextPending()">Next Pending &raquo;</button>
      </div>
    </div>
  </main>

  <div id="toast"></div>

  <script>
    let currentIndex = 1;
    let totalItems = 178;
    let currentItem = null;

    const reviewerInput = document.getElementById("reviewer-id-input");
    reviewerInput.value = localStorage.getItem("terramind_reviewer_id") || "";
    reviewerInput.addEventListener("input", (e) => {
      localStorage.setItem("terramind_reviewer_id", e.target.value.trim());
      checkSelfReview();
    });

    function showToast(msg, isError = false) {
      const t = document.getElementById("toast");
      t.className = isError ? "toast-error" : "toast-success";
      t.textContent = msg;
      t.style.display = "block";
      setTimeout(() => { t.style.display = "none"; }, 3500);
    }

    async function loadItem(index) {
      try {
        const res = await fetch(`/api/item/${index}`);
        if (!res.ok) throw new Error("Failed to load item");
        currentItem = await res.json();
        currentIndex = currentItem.index;
        totalItems = currentItem.total;

        document.getElementById("cur-idx").textContent = currentIndex;
        document.getElementById("tot-idx").textContent = totalItems;
        document.getElementById("meta-sample-id").textContent = currentItem.sample_id;
        document.getElementById("meta-alias").textContent = currentItem.alias;
        document.getElementById("meta-observed-at").textContent = currentItem.observed_at || "—";
        document.getElementById("meta-observer-id").textContent = currentItem.observer_id || "—";
        document.getElementById("observation-text").textContent = currentItem.observation || "(No text)";
        document.getElementById("review-notes").value = currentItem.notes || "";

        const statusEl = document.getElementById("meta-status");
        statusEl.innerHTML = `<span class="status-pill pill-${currentItem.review_status}">${currentItem.review_status}</span>`;

        document.getElementById("leaf-image").src = `/image/${encodeURIComponent(currentItem.sample_id)}`;

        checkSelfReview();
        updateProgress();
      } catch (e) {
        showToast(e.message, true);
      }
    }

    function checkSelfReview() {
      if (!currentItem) return;
      const rev = reviewerInput.value.trim().toLowerCase();
      const obs = (currentItem.observer_id || "").trim().toLowerCase();
      const banner = document.getElementById("observer-banner");
      const ind = document.getElementById("self-review-indicator");
      const approveBtn = document.getElementById("btn-approve");

      if (rev && obs && rev === obs) {
        banner.className = "observer-badge self-review-warn";
        ind.textContent = "SELF-REVIEW DETECTED — Approval Blocked";
        approveBtn.disabled = true;
      } else {
        banner.className = "observer-badge";
        ind.textContent = rev ? `Reviewer: ${rev}` : "Enter Reviewer ID above";
        approveBtn.disabled = false;
      }
    }

    async function updateProgress() {
      try {
        const res = await fetch("/api/progress");
        const p = await res.json();
        document.getElementById("prog-text").textContent = `${p.reviewed} / ${p.total} Reviewed`;
        document.getElementById("prog-details").textContent = `A: ${p.approved} | R: ${p.rejected} | D: ${p.disputed} | P: ${p.pending}`;

        const total = p.total || 178;
        document.getElementById("seg-approved").style.width = `${(p.approved / total) * 100}%`;
        document.getElementById("seg-rejected").style.width = `${(p.rejected / total) * 100}%`;
        document.getElementById("seg-disputed").style.width = `${(p.disputed / total) * 100}%`;
      } catch (_) {}
    }

    async function submitDecision(status) {
      const reviewer = reviewerInput.value.trim();
      if (!reviewer) {
        showToast("Please enter your Reviewer ID before adjudicating.", true);
        reviewerInput.focus();
        return;
      }
      const notes = document.getElementById("review-notes").value.trim();
      if ((status === "rejected" || status === "disputed") && !notes) {
        showToast(`Notes are required when marking an observation ${status}.`, true);
        document.getElementById("review-notes").focus();
        return;
      }

      try {
        const res = await fetch("/api/decision", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sample_id: currentItem.sample_id,
            review_status: status,
            reviewer: reviewer,
            notes: notes
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Submission failed");

        showToast(`Saved: ${currentItem.sample_id} -> ${status.toUpperCase()}`);
        if (currentIndex < totalItems) {
          loadItem(currentIndex + 1);
        } else {
          loadItem(currentIndex);
        }
      } catch (e) {
        showToast(e.message, true);
      }
    }

    function nav(delta) {
      const target = currentIndex + delta;
      if (target >= 1 && target <= totalItems) {
        loadItem(target);
      }
    }

    async function jumpNextPending() {
      try {
        const res = await fetch("/api/next_pending");
        const data = await res.json();
        if (data.index) {
          loadItem(data.index);
        } else {
          showToast("All 178 items have been reviewed!");
        }
      } catch (e) {
        showToast(e.message, true);
      }
    }

    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      if (e.key === "a" || e.key === "A") submitDecision("approved");
      if (e.key === "r" || e.key === "R") submitDecision("rejected");
      if (e.key === "d" || e.key === "D") submitDecision("disputed");
      if (e.key === "ArrowLeft") nav(-1);
      if (e.key === "ArrowRight") nav(1);
    });

    window.addEventListener("DOMContentLoaded", () => {
      loadItem(1);
    });
  </script>
</body>
</html>
"""


def create_independent_review_app(context: IndependentReviewContext) -> FastAPI:
    app = FastAPI(title="TerraMind AI Independent Review Tool", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index_page() -> HTMLResponse:
        return HTMLResponse(_PAGE_HTML)

    @app.get("/api/progress")
    def get_progress() -> JSONResponse:
        return JSONResponse(context.progress())

    @app.get("/api/item/{index}")
    def get_item(index: int) -> JSONResponse:
        with context.lock:
            if index < 1 or index > len(context.items):
                raise HTTPException(status_code=404, detail=f"Index {index} out of range (1..{len(context.items)})")
            it = context.items[index - 1]
            return JSONResponse({
                "index": it.index,
                "total": len(context.items),
                "sample_id": it.sample_id,
                "alias": it.alias,
                "observation": it.observation,
                "observer_id": it.observer_id,
                "observed_at": it.observed_at,
                "review_status": it.review_status,
                "reviewer": it.reviewer,
                "reviewed_at": it.reviewed_at,
                "notes": it.notes,
            })

    @app.get("/api/next_pending")
    def get_next_pending() -> JSONResponse:
        with context.lock:
            for it in context.items:
                if it.review_status.casefold() == "pending":
                    return JSONResponse({"index": it.index, "sample_id": it.sample_id})
            return JSONResponse({"index": None, "sample_id": None})

    @app.post("/api/decision")
    def post_decision(body: _DecisionBody) -> JSONResponse:
        try:
            item = context.record_decision(
                sample_id=body.sample_id,
                review_status=body.review_status,
                reviewer=body.reviewer,
                notes=body.notes,
            )
            return JSONResponse({
                "ok": True,
                "sample_id": item.sample_id,
                "review_status": item.review_status,
                "reviewer": item.reviewer,
                "reviewed_at": item.reviewed_at,
            })
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=400)
        except KeyError as err:
            return JSONResponse({"error": str(err)}, status_code=404)

    @app.get("/image/{sample_id:path}")
    def get_image(sample_id: str) -> Response:
        try:
            path = context.resolve_image(sample_id)
        except (KeyError, ValueError, FileNotFoundError):
            raise HTTPException(status_code=404, detail="Image not found")
        data = path.read_bytes()
        media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
        return Response(content=data, media_type=media_type)

    return app


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Dedicated independent plant observation review server and UI.",
    )
    parser.add_argument("--alias-map", type=Path, default=DEFAULT_ALIAS_MAP)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--review-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (localhost only)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--check", action="store_true", help="Validate context and exit without starting web server")
    args = parser.parse_args(argv)

    context = load_independent_review_context(
        alias_map_path=args.alias_map,
        workbook_path=args.workbook,
        ledger_path=args.review_ledger,
        images_dir=args.images_dir,
    )
    report = check_review_context(context)
    print(json.dumps(report, indent=2), flush=True)

    if args.check:
        raise SystemExit(0 if report["decision"] == "READY" else 1)
    if report["decision"] != "READY":
        raise SystemExit(f"Independent review server cannot start: {report['blocker']}")

    import uvicorn

    app = create_independent_review_app(context)
    if not _port_is_available(args.host, args.port):
        raise SystemExit(
            f"Cannot start independent review server: port {args.port} on {args.host} is already in use."
        )

    print(
        f"\nIndependent review tool ready — open http://{args.host}:{args.port} in a browser.\n"
        f"Target Review Ledger: {context.ledger_path}\n"
        f"Progress: {report['reviewed_items']}/{report['total_items']} reviewed "
        f"({report['pending_items']} pending)\n"
        "Press Ctrl+C to stop.\n",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
