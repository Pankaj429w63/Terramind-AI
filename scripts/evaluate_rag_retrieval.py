"""Evaluate local RAG retrieval on a small, manually labeled query set.

Relevance labels are source-document basenames selected by a human before
retrieval. Metrics are hit-rate, reciprocal rank, and recall at five.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rag.pipelines.knowledge import pipeline

OUT = ROOT / "outputs" / "research"
K = 5

# Manually inspected query-to-source relevance judgments; one relevant source
# per query. These judgments are a smoke-sized evaluation set, not a benchmark.
CASES = [
    {"id": "disease_apple_scab", "category": "diseases", "query": "apple scab leaf lesions disease", "relevant": ["apple_scab.diseases.json"]},
    {"id": "disease_tomato_late_blight", "category": "diseases", "query": "tomato late blight symptoms lesions", "relevant": ["tomato_late_blight.diseases.json"]},
    {"id": "treatment_tomato_late_blight", "category": "treatment", "query": "tomato late blight treatment fungicide", "relevant": ["tomato_late_blight.treatment.json"]},
    {"id": "treatment_apple_scab", "category": "treatment", "query": "apple scab management control fungicide", "relevant": ["apple_scab.treatment.json"]},
    {"id": "fertilizer_tomato", "category": "fertilizer", "query": "tomato fertilizer nitrogen guidance", "relevant": ["tomato.fertilizer.json"]},
    {"id": "fertilizer_corn", "category": "fertilizer", "query": "corn nutrient fertilizer nitrogen recommendations", "relevant": ["corn.fertilizer.json"]},
    {"id": "care_cucumber", "category": "plant care", "query": "cucumber plant care watering", "relevant": ["cucumber.plant_care.json"]},
    {"id": "care_tomato", "category": "plant care", "query": "tomato care pruning watering", "relevant": ["tomato.plant_care.json"]},
    {"id": "agriculture_corn", "category": "agriculture", "query": "corn cultivation agriculture", "relevant": ["corn.agriculture.json"]},
    {"id": "agriculture_apple", "category": "agriculture", "query": "apple orchard growing production guide", "relevant": ["apple.agriculture.json"]},
]


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = pipeline.stats()
    rows = []
    case_details = []
    for case in CASES:
        result = pipeline.retrieve(case["query"], limit=K, category=case["category"])
        relevant = set(case["relevant"])
        ranked = result["sources"]
        names = [Path(str(item.get("source", ""))).name for item in ranked]
        relevant_ranks = [i + 1 for i, name in enumerate(names) if name in relevant]
        hits = len(set(names) & relevant)
        reciprocal_rank = 1.0 / min(relevant_ranks) if relevant_ranks else 0.0
        row = {
            "id": case["id"], "category": case["category"], "query": case["query"],
            "k": K, "relevant_sources": sorted(relevant), "retrieved_sources": names,
            "hit_at_5": int(hits > 0), "precision_at_5": hits / K,
            "recall_at_5": hits / len(relevant), "reciprocal_rank": reciprocal_rank,
            "citation_present": int(bool(ranked) and all(item.get("source") and item.get("title") for item in ranked)),
            "error": result.get("error"),
        }
        rows.append(row)
        case_details.append({**row, "ranked": result["sources"]})
    summary = {
        "evaluation": "manually labeled retrieval smoke set",
        "labeling": "one manually selected relevant KB source basename per query; source was checked against the query and KB title/content before running retrieval",
        "k": K,
        "query_count": len(rows),
        "metrics": {
            "hit_rate_at_5": sum(r["hit_at_5"] for r in rows) / len(rows),
            "precision_at_5": sum(r["precision_at_5"] for r in rows) / len(rows),
            "recall_at_5": sum(r["recall_at_5"] for r in rows) / len(rows),
            "mean_reciprocal_rank": sum(r["reciprocal_rank"] for r in rows) / len(rows),
            "citation_presence_rate": sum(r["citation_present"] for r in rows) / len(rows),
        },
        "index": stats,
        "cases": case_details,
    }
    (OUT / "rag_retrieval_evaluation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT / "rag_retrieval_evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "category", "query", "k", "relevant_sources", "retrieved_sources", "hit_at_5", "precision_at_5", "recall_at_5", "reciprocal_rank", "citation_present", "error"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "relevant_sources": ";".join(row["relevant_sources"]), "retrieved_sources": ";".join(row["retrieved_sources"])})
    print(json.dumps({"index": stats, "metrics": summary["metrics"], "json": str(OUT / "rag_retrieval_evaluation.json")}, indent=2))
    return summary


if __name__ == "__main__":
    main()
