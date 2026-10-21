"""
MaintenAI— Maintenance Agent & Tools Unit Tests

Tests diagnostic state progression, tool execution, grounding guardrails,
and human-in-the-loop work order creation.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from agents.maintenance_agent.graph import agent_app
from agents.maintenance_agent.guardrails import validate_grounding
from agents.maintenance_agent.tools import (
    draft_work_order,
    get_machine_telemetry,
    get_ml_health_assessment,
    search_maintenance_manuals,
)


@pytest.mark.unit
def test_agent_tools():
    # 1. Telemetry tool
    tel = get_machine_telemetry("turbofan-001")
    assert tel["machine_id"] == "turbofan-001"
    assert "temperature" in tel
    assert "vibration" in tel

    # 2. ML health tool
    health = get_ml_health_assessment("turbofan-001")
    assert "anomaly_score" in health
    assert "predicted_rul_cycles" in health
    assert "probable_failure_modes" in health

    # 3. RAG manual search tool
    manuals = search_maintenance_manuals("bearing lubrication vibration", equipment_type="bearing")
    assert isinstance(manuals, list)

    # 4. Draft work order tool
    wo = draft_work_order(
        machine_id="turbofan-001",
        title="Inspect Turbofan Seals",
        category="OVERHAUL",
        priority="HIGH",
        procedures=["Isolate unit", "Check seals"],
        required_parts=["SEAL-01"],
        citations=["manual.md"],
    )
    assert wo["status"] == "PENDING_APPROVAL"
    assert wo["work_order_id"].startswith("WO-")


@pytest.mark.unit
def test_agent_guardrails():
    # Grounded response with citations
    valid, violations = validate_grounding(
        response_text="Based on [turbofan_cf6_manual.md], perform SOP inspection. Draft work order requires human approval.",
        ml_health={"anomaly_score": 0.85},
        retrieved_docs=[{"source_file": "turbofan_cf6_manual.md"}],
    )
    assert valid is True

    # Response making repair recommendations without citations
    invalid, violations = validate_grounding(
        response_text="I recommend replacing the rotor and increasing the torque immediately.",
        ml_health={"anomaly_score": 0.85},
        retrieved_docs=[{"source_file": "turbofan_cf6_manual.md"}],
    )
    assert invalid is False
    assert len(violations) > 0


@pytest.mark.unit
def test_langgraph_agent_invocation():
    state = {
        "messages": [HumanMessage(content="Diagnose machine turbofan-001 and draft a work order.")],
        "machine_id": "turbofan-001",
        "retrieved_docs": [],
    }

    result = agent_app.invoke(state)

    assert result["machine_id"] == "turbofan-001"
    assert "ml_health" in result
    assert result["requires_human_approval"] is True
    assert result["work_order"] is not None
    assert result["work_order"]["status"] == "PENDING_APPROVAL"
    assert len(result["messages"]) > 1
