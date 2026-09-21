from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kb_validation.config import TrustedSourceConfig, load_trusted_config
from kb_validation.validator import ValidationResult, validate_file, VALID_CATEGORIES


RAG_CATEGORY_ALIAS = {
    "diseases": "diseases",
    "treatment": "treatment",
    "fertilizer": "fertilizer",
    "plant_care": "plant care",
    "agriculture": "agriculture",
}

REVIEW_LOG_TEMPLATE: dict[str, Any] = {
    "schema_version": "1.0.0",
    "description": "Human-source-review tracking log. One entry per document per review cycle.",
    "entries": []
}

REVIEW_LOG_ENTRY_SCHEMA: dict[str, Any] = {
    "document_id": "",
    "document_path": "",
    "class_label": "",
    "crop": "",
    "disease": None,
    "category": "",
    "reviewed_by": "",
    "reviewed_date": "",
    "source_url": "",
    "source_org": "",
    "sourced": False,
    "reliability_set": "",
    "conflicts": [],
    "notes": "",
}


@dataclass
class WorkflowReport:
    scanned_files: int = 0
    passed: int = 0
    failed: int = 0
    results: list[ValidationResult] = field(default_factory=list)
    error_histogram: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, dict[str, int]] = field(default_factory=dict)
    by_tier: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned_files": self.scanned_files,
            "passed": self.passed,
            "failed": self.failed,
            "error_histogram": self.error_histogram,
            "by_category": self.by_category,
            "by_tier": self.by_tier,
            "results": [r.to_dict() for r in self.results],
        }


class SourceCollectionWorkflow:
    def __init__(self, kb_root: Path | str,
                 config_path: Path | str | None = None) -> None:
        self.kb_root = Path(kb_root).resolve()
        if not self.kb_root.is_dir():
            raise FileNotFoundError(f"KB root not a directory: {self.kb_root}")
        if config_path is None:
            config_path = self.kb_root / "_schemas" / "trusted_sources.config.json"
        self.config_path = Path(config_path).resolve()
        self.config: TrustedSourceConfig = load_trusted_config(self.config_path)
        self._audit_log_path = self.kb_root / "_workflow" / "source_review_log.json"
        self._report_path = self.kb_root / "_workflow" / "validation_report.json"
        self._ready_manifest_path = self.kb_root / "_workflow" / "rag_ingestion.ready_for_ingest.json"

    # ------------------------------------------------------------------ discovery
    def discover_documents(self) -> list[Path]:
        files: list[Path] = []
        for folder_key, rag_cat in RAG_CATEGORY_ALIAS.items():
            folder = self.kb_root / folder_key
            if not folder.is_dir():
                continue
            for suffix in {".json"}:
                for fp in folder.rglob(f"*{suffix}"):
                    if fp.name.startswith("_"):
                        continue
                    if fp.suffix.lower() != ".json":
                        continue
                    files.append(fp)
        return sorted(set(files))

    # ---------------------------------------------------------------- validation
    def validate_all(self, *, stop_on_error: bool = False) -> WorkflowReport:
        report = WorkflowReport()
        for fp in self.discover_documents():
            try:
                result = validate_file(fp, self.config)
            except Exception as exc:
                from kb_validation.validator import ValidationIssue, ValidationResult
                result = ValidationResult(
                    document_path=str(fp.resolve()),
                    document_id=None,
                    title=None,
                    category=None,
                    passed=False,
                    issues=[ValidationIssue(
                        code="UNREADABLE_DOCUMENT",
                        message=f"Could not parse document file: {exc}",
                        severity="error",
                        field=None,
                        suggestion="Fix JSON syntax.",
                    )],
                )
            report.scanned_files += 1
            report.results.append(result)
            if result.passed:
                report.passed += 1
            else:
                report.failed += 1
            if stop_on_error and report.failed:
                break
            self._tally_errors(result, report)
            self._tally_buckets(result, report)
        return report

    def _tally_errors(self, result: ValidationResult, report: WorkflowReport) -> None:
        for issue in result.errors():
            report.error_histogram[issue.code] = report.error_histogram.get(issue.code, 0) + 1

    def _tally_buckets(self, result: ValidationResult, report: WorkflowReport) -> None:
        cat = result.category or "unknown"
        bucket = report.by_category.setdefault(cat, {"passed": 0, "failed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed" if result.passed else "failed"] += 1
        tier = result.minimum_required_tier or "unknown"
        tb = report.by_tier.setdefault(tier, {"passed": 0, "failed": 0, "total": 0})
        tb["total"] += 1
        tb["passed" if result.passed else "failed"] += 1

    # -------------------------------------------------------- audit log / review
    def ensure_audit_log(self) -> Path:
        path = self._audit_log_path
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            data = dict(REVIEW_LOG_TEMPLATE)
            data["entries"] = []
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def add_review_entry(self, entry: dict[str, Any]) -> None:
        path = self.ensure_audit_log()
        log = json.loads(path.read_text(encoding="utf-8"))
        required_fields = [k for k, _v in REVIEW_LOG_ENTRY_SCHEMA.items() if k in {"document_id", "reviewed_by", "reviewed_date", "sourced"}]
        for k in required_fields:
            if k not in entry or (isinstance(entry[k], str) and not entry[k]) or entry[k] is None:
                raise ValueError(f"review entry missing required field: {k}")
        log["entries"].append(entry)
        path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    # --------------------------------------------------------------- reporting
    def write_report(self, report: WorkflowReport) -> Path:
        out = self._report_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    def export_error_csv(self, report: WorkflowReport, out_path: Path | str) -> Path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "document_path", "document_id", "category", "title", "error_code",
                "severity", "field", "message", "suggestion",
            ])
            for r in report.results:
                if r.passed:
                    continue
                for issue in r.errors():
                    writer.writerow([
                        r.document_path or "",
                        r.document_id or "",
                        r.category or "",
                        r.title or "",
                        issue.code,
                        issue.severity,
                        issue.field or "",
                        issue.message,
                        issue.suggestion or "",
                    ])
        return p

    # ---------------------------------------------------------- ready manifest
    def build_ready_manifest(self, report: WorkflowReport) -> dict[str, Any]:
        passed_results = [r for r in report.results if r.passed]
        sources: list[dict[str, Any]] = []
        for folder_key, rag_cat in RAG_CATEGORY_ALIAS.items():
            folder = self.kb_root / folder_key
            passed_in_folder = [r for r in passed_results if r.category == rag_cat]
            if not passed_in_folder:
                continue
            sources.append({
                "source": str(folder.resolve()),
                "category": rag_cat,
                "format": "folder_of_json_validated",
                "document_count_validated": len(passed_in_folder),
                "document_ids": sorted({r.document_id for r in passed_in_folder if r.document_id}),
                "loader_note": (
                    "All listed document_ids passed source-verification validation. "
                    "Load only documents whose metadata.status == 'sourced' and "
                    "document_id is included in document_ids list."
                ),
            })
        return {
            "version": "1.0.0",
            "description": (
                "Ready-for-ingestion manifest generated by SourceCollectionWorkflow. "
                "Includes ONLY documents that passed the trusted-source validator."
            ),
            "generated_by": "kb_validation.workflow.SourceCollectionWorkflow",
            "trusted_sources_config": str(self.config_path),
            "validated_documents_total": len(passed_results),
            "total_documents_scanned": report.scanned_files,
            "fraction_validated": (
                round(len(passed_results) / report.scanned_files, 4) if report.scanned_files else 0.0
            ),
            "sources": sources,
            "failed_document_count": report.failed,
            "top_failure_codes": sorted(
                report.error_histogram.items(), key=lambda kv: kv[1], reverse=True
            )[:15],
        }

    def write_ready_manifest(self, report: WorkflowReport) -> Path:
        manifest = self.build_ready_manifest(report)
        out = self._ready_manifest_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    # ---------------------------------------------------------- summary helpers
    def summarize(self, report: WorkflowReport) -> str:
        lines = []
        lines.append(f"Scanned : {report.scanned_files}")
        lines.append(f"Passed  : {report.passed}")
        lines.append(f"Failed  : {report.failed}")
        if report.scanned_files:
            pct = round(100.0 * report.passed / report.scanned_files, 2)
            lines.append(f"Pass-rate: {pct}%")
        lines.append("By category:")
        for cat, b in report.by_category.items():
            lines.append(f"  - {cat:<15s} total={b['total']:<4d} passed={b['passed']:<4d} failed={b['failed']:<4d}")
        if report.error_histogram:
            lines.append("Top error codes:")
            for code, n in sorted(report.error_histogram.items(), key=lambda kv: kv[1], reverse=True)[:10]:
                lines.append(f"  - {code:<30s} {n}")
        return "\n".join(lines)
