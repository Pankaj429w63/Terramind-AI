from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


@dataclass
class TierInfo:
    name: str
    rank: int
    label: str
    description: str
    domains: list[str] = field(default_factory=list)
    domain_suffixes: list[str] = field(default_factory=list)
    url_patterns: list[str] = field(default_factory=list)
    minimum_allowlist_policy: str = "allowlist_only"

    def domain_set(self) -> set[str]:
        return set(self.domains)


@dataclass
class PlaceholderRules:
    url_patterns: list[str]
    source_patterns: list[str]
    title_patterns: list[str]
    content_patterns: list[str]
    author_patterns: list[str]
    date_patterns: list[str]
    blocked_domains: list[str]

    def _match_any(self, value: str, patterns: list[str]) -> bool:
        if not isinstance(value, str):
            return False
        for pat in patterns:
            try:
                if re.search(pat, value, flags=re.IGNORECASE):
                    return True
            except re.error:
                if pat.lower() in value.lower():
                    return True
        return False

    def is_placeholder_url(self, url: str) -> bool:
        if not isinstance(url, str) or not url:
            return True
        if self._match_any(url, self.url_patterns):
            return True
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            return any(host == d or host.endswith("." + d) for d in self.blocked_domains)
        except Exception:
            return True

    def is_placeholder_source(self, source: str) -> bool:
        return self._match_any(source, self.source_patterns)

    def is_placeholder_title(self, title: str) -> bool:
        return self._match_any(title, self.title_patterns)

    def is_placeholder_content(self, content: str) -> bool:
        return self._match_any(content, self.content_patterns)

    def is_placeholder_author(self, author: str) -> bool:
        return self._match_any(author, self.author_patterns)

    def is_placeholder_date(self, date: str) -> bool:
        return self._match_any(date, self.date_patterns)


@dataclass
class TrustedSourceConfig:
    schema_version: str
    priority_order: list[str]
    tiers: dict[str, TierInfo]
    placeholders: PlaceholderRules
    reliability_upgrade_mode: str
    minimum_required_rules: list[dict[str, Any]]
    raw: dict[str, Any]

    def tier_for_reliability(self, reliability: str) -> TierInfo | None:
        return self.tiers.get(reliability)

    def rank_of(self, reliability: str) -> int:
        tier = self.tiers.get(reliability)
        return tier.rank if tier else 999

    def is_higher_or_equal(self, reliability: str, minimum: str) -> bool:
        return self.rank_of(reliability) <= self.rank_of(minimum)

    def infer_tier_from_url(self, url: str) -> tuple[str | None, str]:
        """
        Returns (tier_name, reason). tier_name is None if no trusted tier matches.
        """
        if not isinstance(url, str):
            return None, "missing url"
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
        except Exception:
            return None, "unparseable url"
        if not host:
            return None, "empty hostname"
        for tier_name in self.priority_order:
            tier = self.tiers.get(tier_name)
            if tier is None or tier_name == "unverified":
                continue
            if host in tier.domain_set():
                return tier_name, f"host {host} in {tier_name} allowlist"
            for suf in tier.domain_suffixes:
                if host.endswith(suf):
                    return tier_name, f"host {host} matches suffix {suf} for {tier_name}"
            for pat in tier.url_patterns:
                try:
                    if re.search(pat, url, flags=re.IGNORECASE):
                        return tier_name, f"url matches pattern {pat} for {tier_name}"
                except re.error:
                    continue
        return None, "no trusted-tier domain/pattern match"

    def minimum_reliability_for_category(self, category: str) -> str:
        for rule in self.minimum_required_rules:
            if category in rule.get("categories", []):
                return rule.get("minimum_reliability", "unverified")
        return "unverified"


def load_trusted_config(path: Path | str) -> TrustedSourceConfig:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Trusted sources config not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))

    priority = list(raw.get("priority_order") or [])
    tiers_raw = raw.get("reliability_tiers") or {}
    tiers: dict[str, TierInfo] = {}
    for name, info in tiers_raw.items():
        tiers[name] = TierInfo(
            name=name,
            rank=int(info.get("rank", 99)),
            label=str(info.get("label", name)),
            description=str(info.get("description", "")),
            domains=list(info.get("domains") or []),
            domain_suffixes=list(info.get("domain_suffixes") or []),
            url_patterns=list(info.get("url_patterns") or []),
            minimum_allowlist_policy=str(info.get("minimum_allowlist_policy", "allowlist_only")),
        )

    ph = raw.get("placeholder_detection") or {}
    placeholders = PlaceholderRules(
        url_patterns=list(ph.get("url_patterns") or []),
        source_patterns=list(ph.get("source_patterns") or []),
        title_patterns=list(ph.get("title_patterns") or []),
        content_patterns=list(ph.get("content_patterns") or []),
        author_patterns=list(ph.get("author_patterns") or []),
        date_patterns=list(ph.get("date_patterns") or []),
        blocked_domains=list(ph.get("blocked_domains") or []),
    )

    upgrade_mode = str((raw.get("reliability_upgrade_rules") or {}).get("mode", "warn_if_conflict"))
    min_rules = list(raw.get("minimum_required_rules") or [])

    return TrustedSourceConfig(
        schema_version=str(raw.get("schema_version", "1.0.0")),
        priority_order=priority,
        tiers=tiers,
        placeholders=placeholders,
        reliability_upgrade_mode=upgrade_mode,
        minimum_required_rules=min_rules,
        raw=raw,
    )
