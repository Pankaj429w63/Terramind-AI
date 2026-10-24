"""Optional OpenAI-compatible answer generation with a source-only fallback."""
from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.request import Request, urlopen

log = logging.getLogger("terramind.llm")


class AnswerGenerator:
    def __init__(self, timeout_seconds: float = 20.0) -> None:
        self.timeout_seconds = timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(os.getenv("TERRAMIND_LLM_BASE_URL") and os.getenv("TERRAMIND_LLM_MODEL"))

    def generate(self, query: str, evidence: list[dict[str, Any]]) -> dict[str, str]:
        if not evidence:
            return {"text": "I could not find supporting information in the indexed sources. Please try a more specific question or consult a local agricultural specialist.", "mode": "no_evidence", "status": "no_retrieval_results"}
        if self.configured:
            try:
                answer = self._remote(query, evidence)
                return {"text": answer, "mode": "llm", "status": "completed"}
            except Exception as error:
                # Expose the failure state, but keep the evidence-backed fallback useful.
                log.warning("LLM provider failed; using evidence template (%s): %s", type(error).__name__, error)
                return {"text": self._template(evidence), "mode": "evidence_template", "status": f"provider_error: {type(error).__name__}"}
        return {"text": self._template(evidence), "mode": "evidence_template", "status": "provider_not_configured"}

    def _remote(self, query: str, evidence: list[dict[str, Any]]) -> str:
        base = os.environ["TERRAMIND_LLM_BASE_URL"].rstrip("/")
        model = os.environ["TERRAMIND_LLM_MODEL"]
        items = [{"id": f"S{i + 1}", "title": x.get("title"), "text": x.get("text", "")[:1800]} for i, x in enumerate(evidence)]
        body = {"model": model, "temperature": 0, "messages": [
            {"role": "system", "content": "Answer only from supplied evidence. If evidence is insufficient, say so. Cite claims using [S1] style IDs. Never diagnose from this evidence."},
            {"role": "user", "content": json.dumps({"question": query, "evidence": items}, ensure_ascii=False)},
        ]}
        headers = {"Content-Type": "application/json"}
        key = os.getenv("TERRAMIND_LLM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = Request(f"{base}/chat/completions", data=json.dumps(body).encode(), headers=headers, method="POST")
        with urlopen(request, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        answer = data["choices"][0]["message"]["content"].strip()
        if not answer:
            raise ValueError("Provider returned an empty answer")
        # Keep provider-generated citation tokens tied to evidence actually supplied.
        allowed = {f"[S{i + 1}]" for i in range(len(evidence))}
        import re
        citations = re.findall(r"\[S\d+\]", answer)
        if not citations or any(token not in allowed for token in citations):
            raise ValueError("Provider cited an unknown evidence ID")
        return answer

    @staticmethod
    def _template(evidence: list[dict[str, Any]]) -> str:
        lines = ["Evidence from TerraMind's indexed sources (these sources are guidance, not a diagnosis):"]
        for i, item in enumerate(evidence, 1):
            excerpt = str(item.get("excerpt") or item.get("text") or "").strip()
            if len(excerpt) > 700:
                excerpt = excerpt[:697].rstrip() + "..."
            lines.append(f"\n[S{i}] {item.get('title') or 'Untitled source'} — {excerpt}")
        return "\n".join(lines)
