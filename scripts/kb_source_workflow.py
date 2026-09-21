#!/usr/bin/env python3
"""
TerraMind AI — Knowledge Base Source-Collection Workflow CLI.

Usage:
  python scripts/kb_source_workflow.py validate [--json] [--csv ERRORS_CSV]
  python scripts/kb_source_workflow.py report [--json] [--summary-only]
  python scripts/kb_source_workflow.py export-ready-manifest
  python scripts/kb_source_workflow.py add-review --entry PATH_TO_REVIEW_JSON
  python scripts/kb_source_workflow.py audit-summary

Never modifies ML / training files or the existing RAG ingestion pipeline.
All output files land under data/knowledge_base/_workflow/ only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from kb_validation.config import load_trusted_config
from kb_validation.workflow import SourceCollectionWorkflow


def _workflow() -> SourceCollectionWorkflow:
    kb_root = PROJECT_ROOT / "data" / "knowledge_base"
    return SourceCollectionWorkflow(kb_root=kb_root)


def cmd_validate(args: argparse.Namespace) -> int:
    wf = _workflow()
    report = wf.validate_all()
    if args.csv:
        p = wf.export_error_csv(report, Path(args.csv))
        print(f"[ok] wrote error CSV -> {p}")
    if args.json:
        p = wf.write_report(report)
        print(f"[ok] wrote validation report JSON -> {p}")
        print()
    print(wf.summarize(report))
    return 0 if report.passed == report.scanned_files else 1


def cmd_report(args: argparse.Namespace) -> int:
    wf = _workflow()
    report = wf.validate_all()
    if args.json:
        p = wf.write_report(report)
        print(f"[ok] wrote validation report JSON -> {p}")
    if not args.summary_only or args.json is False:
        print(wf.summarize(report))
    if not args.summary_only:
        print()
        print("=== Per-document failures (first 25) ===")
        shown = 0
        for r in report.results:
            if r.passed:
                continue
            errs = "; ".join(f"{e.code}:{e.field or '?'}" for e in r.errors())
            print(f"  [FAIL] {r.document_id or '?':<38s} {r.category or '?':<12s} {errs}")
            shown += 1
            if shown >= 25:
                break
        if report.failed > 25:
            print(f"  ... and {report.failed - 25} more — see JSON report for full details.")
    return 0 if report.passed == report.scanned_files else 1


def cmd_export_ready(args: argparse.Namespace) -> int:
    wf = _workflow()
    report = wf.validate_all()
    p = wf.write_ready_manifest(report)
    print(f"[ok] ready-for-ingest manifest -> {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    print(
        f"[ok] {data['validated_documents_total']}/{data['total_documents_scanned']} "
        f"documents passed source validation "
        f"({data.get('fraction_validated', 0):.1%})."
    )
    if data["validated_documents_total"] == 0:
        print("[note] No documents are ready yet. Replace placeholder content/sources first.")
    return 0


def cmd_add_review(args: argparse.Namespace) -> int:
    entry_path = Path(args.entry).resolve()
    if not entry_path.is_file():
        print(f"[err] review JSON file not found: {entry_path}", file=sys.stderr)
        return 2
    payload = json.loads(entry_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "entry" in payload and isinstance(payload["entry"], dict):
        entry = payload["entry"]
    elif isinstance(payload, dict) and "document_id" in payload:
        entry = payload
    else:
        print(
            "[err] review JSON must be either {entry: {...}} (per template) or a flat entry "
            "with document_id, reviewed_by, reviewed_date, sourced.",
            file=sys.stderr,
        )
        return 2
    wf = _workflow()
    wf.add_review_entry(entry)
    p = wf.ensure_audit_log()
    print(f"[ok] appended 1 review entry -> {p}")
    return 0


def cmd_audit_summary(args: argparse.Namespace) -> int:
    wf = _workflow()
    log_path = wf.ensure_audit_log()
    log = json.loads(log_path.read_text(encoding="utf-8"))
    entries = list(log.get("entries") or [])
    total = len(entries)
    sourced = sum(1 for e in entries if e.get("sourced") is True)
    by_reviewer: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    for e in entries:
        rev = e.get("reviewed_by") or "<unknown>"
        by_reviewer[rev] = by_reviewer.get(rev, 0) + 1
        cat = e.get("category") or "?"
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print(f"Audit log       : {log_path}")
    print(f"Total entries   : {total}")
    print(f"Sourced         : {sourced}")
    if total:
        print(f"Sourced %       : {100 * sourced / total:.1f}%")
    print(f"By reviewer     : {by_reviewer}")
    print(f"By category     : {by_cat}")

    report = wf.validate_all()
    print()
    print("Current KB source-validation status:")
    print(wf.summarize(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kb_source_workflow",
        description="TerraMind AI KB source-collection / verification workflow.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="Validate all KB documents.")
    p.add_argument("--json", action="store_true", help="Write JSON report to _workflow/validation_report.json")
    p.add_argument("--csv", default=None, help="Write error breakdown CSV to this path")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("report", help="Validate + print per-document failure summary.")
    p.add_argument("--json", action="store_true", help="Also write full JSON report.")
    p.add_argument("--summary-only", action="store_true", help="Only show summary, no per-doc details.")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("export-ready-manifest", help="Generate ready-for-ingest manifest (only validated docs).")
    p.set_defaults(func=cmd_export_ready)

    p = sub.add_parser("add-review", help="Append a source-review entry to the audit log.")
    p.add_argument("--entry", required=True, help="Path to a JSON file matching source_review_entry.template.json")
    p.set_defaults(func=cmd_add_review)

    p = sub.add_parser("audit-summary", help="Show audit log counts + current validation status.")
    p.set_defaults(func=cmd_audit_summary)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
