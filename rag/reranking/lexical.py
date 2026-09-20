from __future__ import annotations

import re


def rerank(query: str, results: list[dict], limit: int) -> list[dict]:
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    ranked = []
    for result in results:
        text_terms = set(re.findall(r"[a-z0-9]+", result.get("text", "").lower()))
        overlap = len(terms & text_terms) / max(1, len(terms))
        ranked.append({**result, "retrieval_score": round(0.7 * float(result.get("score", 0)) + 0.3 * overlap, 6)})
    return sorted(ranked, key=lambda item: item["retrieval_score"], reverse=True)[:limit]
