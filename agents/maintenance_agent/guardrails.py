"""
MaintenAI— Maintenance Agent Safety Guardrails

Enforces strict boundaries:
  1. Grounded numbers: RUL and anomaly metrics must strictly originate from verified tool output.
  2. Grounded procedures: SOP repair recommendations must include manual citations.
  3. Action boundary: No physical work order execution without explicit human authorization.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_grounding(
    response_text: str,
    ml_health: dict[str, Any] | None,
    retrieved_docs: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """
    Validate that response conforms to grounding standards.
    """
    violations: list[str] = []

    # 1. Check citation presence if recommendations are made
    if any(
        keyword in response_text.lower()
        for keyword in ["recommend", "sop", "procedure", "inspection", "replace", "torque"]
    ):
        # Check if citations or document titles exist in text or retrieved docs
        doc_names = [doc.get("source_file", "").lower() for doc in retrieved_docs]
        has_citation = (
            any(doc_name in response_text.lower() for doc_name in doc_names if doc_name)
            or "[" in response_text
        )

        if not has_citation and retrieved_docs:
            violations.append(
                "Response provides repair procedures without citing retrieved maintenance manuals."
            )

    # 2. Check for human-in-the-loop reminder if work order mentioned
    if (
        "work order" in response_text.lower()
        and "draft" not in response_text.lower()
        and "approval" not in response_text.lower()
    ):
        violations.append(
            "Work orders must be explicitly designated as DRAFTS requiring HUMAN APPROVAL."
        )

    # 3. Check for low-relevance caution when evidence is weak (< 0.35)
    scored_docs = [d for d in retrieved_docs if "relevance_score" in d]
    if scored_docs:
        max_score = max((float(d["relevance_score"]) for d in scored_docs), default=0.0)
        if max_score < 0.35:
            has_caution = any(
                term in response_text.lower()
                for term in ["low", "insufficient", "weak", "physical validation", "caution"]
            )
            if not has_caution:
                violations.append(
                    f"Retrieved document evidence is weak (max score: {max_score:.2f} < 0.35); response must disclose low confidence."
                )

    # 4. Numeric grounding validation against verified tool output
    if ml_health:
        expected_rul = ml_health.get("predicted_rul_cycles")
        if expected_rul is not None:
            rul_str = f"{float(expected_rul):.1f}"
            # If RUL is mentioned in text, it must match tool output
            if (
                "rul" in response_text.lower()
                and "cycle" in response_text.lower()
                and rul_str not in response_text
                and f"{int(float(expected_rul))}" not in response_text
            ):
                violations.append(
                    f"Fabricated or mismatched RUL value in response. Expected ~{rul_str} cycles from tool output."
                )

    return len(violations) == 0, violations
