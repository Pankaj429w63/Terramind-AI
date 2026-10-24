"""Aggregate real fusion test evaluations into a comparison report.

Reads ``outputs/fusion/<kind>/evaluation/test_metrics.json`` (written by
``ml/fusion/evaluate.py``) for early/intermediate/late fusion, plus the
image-only and text-only baselines recorded inside those files. Only metrics
that were actually measured are reported: a missing evaluation is reported as
"not evaluated", never as a fabricated score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUSION_ROOT = ROOT / "outputs" / "fusion"
DEFAULT_OUTPUT = ROOT / "outputs" / "research" / "three_fusion_models_comparison.md"
KINDS = ("early", "intermediate", "late")
METRIC_KEYS = (
    "test_examples", "test_accuracy", "macro_precision", "macro_recall", "macro_f1",
    "weighted_f1", "top3_accuracy", "parameter_count", "model_size_bytes",
    "inference_latency_ms_per_image",
)


def _row_from(block: dict, source: str) -> dict:
    row = {"model": block.get("model"), "source_file": source}
    for key in METRIC_KEYS:
        row[key] = block.get(key)
    return row


def build(fusion_root: Path = DEFAULT_FUSION_ROOT) -> dict:
    rows: list[dict] = []
    per_kind: dict[str, dict] = {}
    baseline_rows: list[dict] = []
    text_rows: list[dict] = []
    manifest_hashes: set[str] = set()
    test_sizes: set[int] = set()
    absent: list[str] = []

    for kind in KINDS:
        metrics_path = Path(fusion_root) / kind / "evaluation" / "test_metrics.json"
        if not metrics_path.is_file():
            absent.append(kind)
            continue
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            absent.append(kind)
            continue
        per_kind[kind] = payload
        rows.append(_row_from(payload["fusion"], str(metrics_path)))
        baseline_rows.append(_row_from(payload["image_only_baseline"], str(metrics_path)))
        if "text_only_baseline" in payload:
            text_rows.append(_row_from(payload["text_only_baseline"], str(metrics_path)))
        if payload.get("eligibility_manifest_sha256"):
            manifest_hashes.add(payload["eligibility_manifest_sha256"])
        if payload.get("test_examples") is not None:
            test_sizes.add(payload["test_examples"])

    baseline = baseline_rows[0] if baseline_rows else None
    text_only = text_rows[0] if text_rows else None
    if baseline is not None:
        rows.append(baseline)
    if text_only is not None:
        rows.append(text_only)

    consistent = len(manifest_hashes) <= 1 and len(test_sizes) <= 1
    report = {
        "report": "three_fusion_models_comparison",
        "fusion_root": str(Path(fusion_root).resolve()),
        "status": "EVALUATED" if per_kind else "NOT_EVALUATED",
        "evaluated_kinds": sorted(per_kind),
        "not_evaluated_kinds": absent,
        "same_eligible_samples": consistent if per_kind else None,
        "test_examples": next(iter(test_sizes)) if len(test_sizes) == 1 else None,
        "eligibility_manifest_sha256": next(iter(manifest_hashes)) if len(manifest_hashes) == 1 else None,
        "rows": rows,
        "notes": [
            "Only metrics present in real test_metrics.json files are shown.",
            "The image-only and text-only rows are the baselines recorded by the fusion evaluation.",
            "Fusion must not be claimed to improve on the baselines unless these rows show it.",
        ],
    }
    return report


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def to_markdown(report: dict) -> str:
    lines = [
        "# TerraMind AI — Three-Model Fusion Comparison",
        "",
        f"- Status: **{report['status']}**",
        f"- Evaluated kinds: {', '.join(report['evaluated_kinds']) or 'none'}",
        f"- Not evaluated: {', '.join(report['not_evaluated_kinds']) or 'none'}",
        f"- Same eligible samples across evaluated runs: {report['same_eligible_samples']}",
        f"- Fixed test example count: {report['test_examples']}",
        "",
    ]
    if not report["rows"]:
        lines.append("No fusion test evaluation is recorded yet; there is nothing to compare.")
        lines.append("")
        lines.extend(f"- {note}" for note in report["notes"])
        return "\n".join(lines) + "\n"
    header = ["model", *METRIC_KEYS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in report["rows"]:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key in header) + " |")
    lines.append("")
    lines.extend(f"- {note}" for note in report["notes"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the three-fusion-model comparison report from measured evaluations.")
    parser.add_argument("--fusion-root", type=Path, default=DEFAULT_FUSION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.fusion_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(to_markdown(report), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "evaluated_kinds": report["evaluated_kinds"],
        "not_evaluated_kinds": report["not_evaluated_kinds"], "output": str(args.output),
    }, indent=2))
    print(f"Wrote {args.output} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
