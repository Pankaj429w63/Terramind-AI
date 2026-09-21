from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kb_validation.config import PlaceholderRules, TrustedSourceConfig


REQUIRED_FIELDS = [
    "schema_version",
    "document_id",
    "category",
    "source",
    "title",
    "crop",
    "disease",
    "content",
    "url",
    "author",
    "date",
    "region",
    "reliability",
    "metadata",
]

VALID_CATEGORIES = {"diseases", "treatment", "fertilizer", "plant care", "agriculture"}
VALID_REGIONS = {"global", "north_america", "europe", "asia", "africa", "south_america", "oceania"}


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: str
    field: str | None = None
    suggestion: str | None = None

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    @property
    def is_warning(self) -> bool:
        return self.severity == "warning"


@dataclass
class ValidationResult:
    document_path: str | None
    document_id: str | None
    title: str | None
    category: str | None
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    inferred_tier: str | None = None
    inferred_tier_reason: str | None = None
    minimum_required_tier: str | None = None

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.is_error]

    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.is_warning]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_path": self.document_path,
            "document_id": self.document_id,
            "title": self.title,
            "category": self.category,
            "passed": self.passed,
            "errors": [i.__dict__ for i in self.errors()],
            "warnings": [i.__dict__ for i in self.warnings()],
            "inferred_tier": self.inferred_tier,
            "inferred_tier_reason": self.inferred_tier_reason,
            "minimum_required_tier": self.minimum_required_tier,
        }


class DocumentValidator:
    def __init__(self, config: TrustedSourceConfig) -> None:
        self.config = config

    def validate(self, doc: dict[str, Any], path: str | None = None) -> ValidationResult:
        issues: list[ValidationIssue] = []
        doc_id = doc.get("document_id")
        title = doc.get("title")
        category = doc.get("category")

        issues.extend(self._check_required_fields(doc))
        issues.extend(self._check_type_and_enums(doc))
        issues.extend(self._check_placeholders(doc))
        issues.extend(self._check_url_domain_reliability(doc))
        issues.extend(self._check_minimum_tier(doc))
        issues.extend(self._check_metadata_status(doc))
        issues.extend(self._check_date_format(doc))

        passed = not any(i.is_error for i in issues)

        inferred, reason = None, None
        min_tier = None
        if category in VALID_CATEGORIES:
            inferred, reason = self.config.infer_tier_from_url(doc.get("url", ""))
            min_tier = self.config.minimum_reliability_for_category(category)

        return ValidationResult(
            document_path=path,
            document_id=doc_id,
            title=title,
            category=category,
            passed=passed,
            issues=issues,
            inferred_tier=inferred,
            inferred_tier_reason=reason,
            minimum_required_tier=min_tier,
        )

    def _check_required_fields(self, doc: dict[str, Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for field in REQUIRED_FIELDS:
            if field not in doc:
                issues.append(ValidationIssue(
                    code="MISSING_FIELD",
                    message=f"Required field '{field}' is missing from the document.",
                    severity="error",
                    field=field,
                    suggestion="Add the missing field as defined in knowledge-document-v1.schema.json.",
                ))
        return issues

    def _check_type_and_enums(self, doc: dict[str, Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        category = doc.get("category")
        if category is not None:
            if not isinstance(category, str) or category not in VALID_CATEGORIES:
                issues.append(ValidationIssue(
                    code="INVALID_CATEGORY",
                    message=f"category '{category}' is not one of the RAG-allowed: {sorted(VALID_CATEGORIES)}.",
                    severity="error",
                    field="category",
                    suggestion="Use one of: diseases, treatment, fertilizer, plant care, agriculture.",
                ))
        region = doc.get("region")
        if region is not None:
            if not isinstance(region, str) or region not in VALID_REGIONS:
                issues.append(ValidationIssue(
                    code="INVALID_REGION",
                    message=f"region '{region}' is not one of the allowed values.",
                    severity="error",
                    field="region",
                    suggestion=f"Use one of: {sorted(VALID_REGIONS)}.",
                ))
        reliability = doc.get("reliability")
        if reliability is not None and reliability not in self.config.tiers:
            issues.append(ValidationIssue(
                code="INVALID_RELIABILITY",
                message=f"reliability '{reliability}' is not a known tier in trusted_sources.config.json.",
                severity="error",
                field="reliability",
                suggestion=f"Use one of: {sorted(self.config.tiers.keys())}.",
            ))
        metadata = doc.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            issues.append(ValidationIssue(
                code="INVALID_METADATA",
                message="metadata must be a JSON object (dict).",
                severity="error",
                field="metadata",
            ))
        return issues

    def _check_placeholders(self, doc: dict[str, Any]) -> list[ValidationIssue]:
        ph: PlaceholderRules = self.config.placeholders
        issues: list[ValidationIssue] = []

        if ph.is_placeholder_url(doc.get("url", "")):
            issues.append(ValidationIssue(
                code="PLACEHOLDER_URL",
                message="URL is placeholder, example.com, REPLACE-WITH, empty, or matches a blocked domain.",
                severity="error",
                field="url",
                suggestion="Replace with a real, citable, authoritative source URL.",
            ))
        if ph.is_placeholder_source(doc.get("source", "")):
            issues.append(ValidationIssue(
                code="PLACEHOLDER_SOURCE",
                message="source field contains '[SOURCE REQUIRED]', 'REPLACE-WITH', or other placeholder marker.",
                severity="error",
                field="source",
                suggestion="Name the real source (journal, agency, extension bulletin).",
            ))
        if ph.is_placeholder_title(doc.get("title", "")):
            issues.append(ValidationIssue(
                code="PLACEHOLDER_TITLE",
                message="title begins with '[PLACEHOLDER]' or a TODO marker.",
                severity="error",
                field="title",
                suggestion="Write a real descriptive title matching the document content.",
            ))
        if ph.is_placeholder_content(doc.get("content", "")):
            issues.append(ValidationIssue(
                code="PLACEHOLDER_CONTENT",
                message="content contains '[REQUIRED:...]', placeholder markers, or instructions to 'not fabricate'.",
                severity="error",
                field="content",
                suggestion="Replace placeholder with factual, properly cited agricultural content.",
            ))
        if ph.is_placeholder_author(doc.get("author", "")):
            issues.append(ValidationIssue(
                code="PLACEHOLDER_AUTHOR",
                message="author is still the placeholder string.",
                severity="error",
                field="author",
                suggestion="Provide real author names, institution, or issuing body.",
            ))
        if ph.is_placeholder_date(doc.get("date", "")):
            issues.append(ValidationIssue(
                code="PLACEHOLDER_DATE",
                message="date is '0000-00-00' or another placeholder.",
                severity="error",
                field="date",
                suggestion="Set publication date in ISO 8601 YYYY-MM-DD.",
            ))
        return issues

    def _check_url_domain_reliability(self, doc: dict[str, Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        url = doc.get("url", "")
        reliability = doc.get("reliability")
        if not isinstance(url, str) or not url:
            return issues
        try:
            parsed = urlparse(url)
            scheme = parsed.scheme
            if scheme not in {"http", "https"}:
                issues.append(ValidationIssue(
                    code="INVALID_URL_SCHEME",
                    message="URL scheme must be http or https.",
                    severity="error",
                    field="url",
                ))
        except Exception:
            issues.append(ValidationIssue(
                code="UNPARSEABLE_URL",
                message="URL could not be parsed.",
                severity="error",
                field="url",
            ))
            return issues

        inferred, reason = self.config.infer_tier_from_url(url)
        if inferred is None and doc.get("url"):
            ph = self.config.placeholders
            if not ph.is_placeholder_url(doc.get("url", "")):
                issues.append(ValidationIssue(
                    code="UNTRUSTED_DOMAIN",
                    message=(
                        f"URL domain is not present in any trusted tier allowlist "
                        f"(peer_reviewed/government/extension/established_org/industry/community)."
                    ),
                    severity="error",
                    field="url",
                    suggestion=(
                        "Switch source URL to a trusted domain in the configured tier allowlists, "
                        "or extend trusted_sources.config.json with this domain in the appropriate tier "
                        "after vetting."
                    ),
                ))

        if inferred is not None and reliability is not None and reliability != "unverified":
            inferred_rank = self.config.rank_of(inferred)
            current_rank = self.config.rank_of(str(reliability))
            if current_rank < inferred_rank:
                mode = self.config.reliability_upgrade_mode
                if mode == "warn_if_conflict":
                    issues.append(ValidationIssue(
                        code="RELIABILITY_UNDERSTATED",
                        message=(
                            f"URL domain resolves to tier '{inferred}' (rank {inferred_rank}) but the "
                            f"document claims lower tier '{reliability}' (rank {current_rank})."
                        ),
                        severity="warning",
                        field="reliability",
                        suggestion=f"Consider upgrading reliability to '{inferred}'.",
                    ))
            elif current_rank > inferred_rank:
                issues.append(ValidationIssue(
                    code="RELIABILITY_OVERSTATED",
                    message=(
                        f"URL domain only matches tier '{inferred}' (rank {inferred_rank}) but the "
                        f"document claims higher tier '{reliability}' (rank {current_rank})."
                    ),
                    severity="error",
                    field="reliability",
                    suggestion=(
                        f"Either reduce reliability to '{inferred}' or add a corroborating source "
                        f"from a '{reliability}' tier domain."
                    ),
                ))
        return issues

    def _check_minimum_tier(self, doc: dict[str, Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        category = doc.get("category")
        reliability = doc.get("reliability")
        if category not in VALID_CATEGORIES:
            return issues
        minimum = self.config.minimum_reliability_for_category(str(category))
        if reliability == "unverified":
            issues.append(ValidationIssue(
                code="UNVERIFIED_RELIABILITY",
                message=(
                    f"reliability is set to 'unverified'; this tier is BLOCKED from ingestion. "
                    f"Category '{category}' requires at minimum '{minimum}'."
                ),
                severity="error",
                field="reliability",
                suggestion=(
                    f"Source this document from a domain in the '{minimum}' tier or higher "
                    f"(peer_reviewed > government > extension > established_org)."
                ),
            ))
        elif not self.config.is_higher_or_equal(str(reliability), minimum):
            issues.append(ValidationIssue(
                code="BELOW_MINIMUM_RELIABILITY",
                message=(
                    f"Category '{category}' requires minimum reliability '{minimum}' "
                    f"(rank {self.config.rank_of(minimum)}) but document uses '{reliability}' "
                    f"(rank {self.config.rank_of(str(reliability))})."
                ),
                severity="error",
                field="reliability",
                suggestion=f"Upgrade the source to at least the '{minimum}' tier.",
            ))
        return issues

    def _check_metadata_status(self, doc: dict[str, Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        meta = doc.get("metadata")
        if not isinstance(meta, dict):
            return issues
        status = meta.get("status")
        replaced = meta.get("placeholder_replaced")
        if isinstance(status, str) and status.lower() == "placeholder":
            issues.append(ValidationIssue(
                code="METADATA_PLACEHOLDER_STATUS",
                message="metadata.status == 'placeholder' — document has not been flagged as sourced yet.",
                severity="error",
                field="metadata.status",
                suggestion="After replacing placeholder content & source, set metadata.status = 'sourced'.",
            ))
        if isinstance(replaced, str) and replaced.lower() != "true":
            issues.append(ValidationIssue(
                code="METADATA_PLACEHOLDER_NOT_REPLACED",
                message="metadata.placeholder_replaced is not 'true' — markers indicate content was not filled.",
                severity="error",
                field="metadata.placeholder_replaced",
                suggestion="Set metadata.placeholder_replaced = 'true' after content/source review.",
            ))
        last_reviewed = meta.get("last_reviewed")
        if isinstance(last_reviewed, str) and not last_reviewed:
            issues.append(ValidationIssue(
                code="METADATA_LAST_REVIEWED_EMPTY",
                message="metadata.last_reviewed is empty.",
                severity="warning",
                field="metadata.last_reviewed",
                suggestion="Set ISO 8601 date YYYY-MM-DD of when the source was last validated.",
            ))
        return issues

    def _check_date_format(self, doc: dict[str, Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        value = doc.get("date")
        if not isinstance(value, str):
            return issues
        if self.config.placeholders.is_placeholder_date(value):
            return issues
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            issues.append(ValidationIssue(
                code="INVALID_DATE_FORMAT",
                message="date must use ISO 8601 YYYY-MM-DD.",
                severity="error",
                field="date",
            ))
            return issues
        try:
            y, m, d = (int(p) for p in value.split("-"))
            date(y, m, d)
        except ValueError:
            issues.append(ValidationIssue(
                code="INVALID_DATE",
                message=f"date value {value} is not a valid calendar date.",
                severity="error",
                field="date",
            ))
        return issues


def validate_document(doc: dict[str, Any], config: TrustedSourceConfig,
                      path: str | None = None) -> ValidationResult:
    return DocumentValidator(config).validate(doc, path=path)


def validate_file(path: Path | str, config: TrustedSourceConfig) -> ValidationResult:
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    return validate_document(doc, config, path=str(p.resolve()))
