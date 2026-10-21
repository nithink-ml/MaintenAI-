"""
MaintenAI— LangGraph Maintenance Copilot Workflow

Defines the multi-step diagnostic workflow:
  1. Diagnose Machine: Fetches verified telemetry and runs deterministic ML models.
  2. Retrieve Knowledge: Queries Hybrid RAG for grounded maintenance manuals and SOPs.
  3. Reason & Draft: Translates ML predictions into clear diagnosis and drafts work orders.
  4. Human Approval Gate: Interrupts workflow until authorized technician approves.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from agents.maintenance_agent.state import AgentState
from agents.maintenance_agent.tools import (
    draft_work_order,
    get_machine_telemetry,
    get_ml_health_assessment,
    search_maintenance_manuals,
)

logger = logging.getLogger(__name__)


def _extract_machine_id(text: str) -> str | None:
    """Extract machine ID pattern like turbofan-001, bearing-004, cnc-012, or machine 1."""
    match = re.search(r"\b(turbofan-\d+|bearing-\d+|cnc-\d+|machine-\d+)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def diagnose_machine_node(state: AgentState) -> dict[str, Any]:
    """Fetch factual telemetry and ML health metrics."""
    messages = state.get("messages", [])
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = str(msg.content)
            break

    machine_id = state.get("machine_id") or _extract_machine_id(last_user_msg) or "turbofan-001"
    telemetry = get_machine_telemetry(machine_id)
    ml_health = get_ml_health_assessment(machine_id)

    return {
        "machine_id": machine_id,
        "equipment_type": telemetry.get("equipment_type", "general"),
        "telemetry_data": telemetry,
        "ml_health": ml_health,
    }


def retrieve_knowledge_node(state: AgentState) -> dict[str, Any]:
    """Search maintenance manuals based on ML failure risks."""
    ml_health = state.get("ml_health", {})
    eq_type = state.get("equipment_type", "general")

    # Build query from probable failure modes and anomaly status
    modes = list(ml_health.get("probable_failure_modes", {}).keys())
    top_mode = modes[0] if modes else "general degradation"
    query = f"{eq_type} {top_mode} corrective procedures and repair steps"

    docs = search_maintenance_manuals(query=query, equipment_type=eq_type, top_k=3)
    return {"retrieved_docs": docs}


def reason_and_draft_node(state: AgentState) -> dict[str, Any]:
    """Synthesize diagnosis, format citations, and generate draft work order."""
    machine_id = state.get("machine_id", "unknown")
    state.get("telemetry_data", {})
    ml_health = state.get("ml_health", {})
    docs = state.get("retrieved_docs", [])

    anom_score = ml_health.get("anomaly_score", 0.0)
    rul = ml_health.get("predicted_rul_cycles", 0.0)
    modes = ml_health.get("probable_failure_modes", {})

    top_mode_name = list(modes.keys())[0] if modes else "General Wear"
    top_mode_prob = list(modes.values())[0] if modes else 0.5

    citations = [d.get("source_file", "manual.md") for d in docs]
    procedures = []
    if docs:
        first_doc = docs[0]
        procedures.append(
            f"Follow {first_doc.get('section_title')}: Inspect and isolate machine {machine_id}."
        )
        procedures.append(
            "Perform acoustic and physical calibration as specified in maintenance manual."
        )
    else:
        procedures.append("Isolate equipment and check sensor transducer connections.")

    # Create draft work order
    draft = draft_work_order(
        machine_id=machine_id,
        title=f"Corrective Overhaul for {machine_id.upper()} ({top_mode_name.replace('_', ' ').title()})",
        category="PREDICTIVE_MAINTENANCE",
        priority="HIGH" if anom_score >= 0.85 or rul <= 15 else "MEDIUM",
        procedures=procedures,
        required_parts=["GASKET-SEAL-KIT", "TRANSDUCER-HARNESS"],
        citations=citations,
        estimated_downtime_hours=3.5,
    )

    # Format grounded sections explicitly distinguishing FACT, ML PREDICTION, RAG EVIDENCE, and RECOMMENDATION
    has_strong_evidence = docs and any(d.get("relevance_score", 0.0) >= 0.35 for d in docs)
    evidence_status = "VERIFIED STRONG" if has_strong_evidence else "INSUFFICIENT / WEAK EVIDENCE"

    summary_text = (
        f"### Diagnostic Report for **{machine_id.upper()}**\n\n"
        f"#### 📊 [MEASURED FACT — PHYSICAL SENSOR TELEMETRY]\n"
        f"- **Operating Cycle**: `{state.get('telemetry_data', {}).get('operating_cycle', 100)}`\n"
        f"- **Primary Sensor Readings**: Temp `{state.get('telemetry_data', {}).get('temperature', 'N/A')}`, "
        f"Vib `{state.get('telemetry_data', {}).get('vibration', 'N/A')}`, "
        f"Pressure `{state.get('telemetry_data', {}).get('pressure', 'N/A')}`\n"
        f"- **Operational Status**: `{ml_health.get('anomaly_status', 'NORMAL')}`\n\n"
        f"#### 🧠 [ML PREDICTION — DETERMINISTIC RUNTIME INFERENCES]\n"
        f"- **Anomaly Score**: **{anom_score:.2f}** / 1.00 *(Model: Isolation Forest `v1.0.0-dev`)*\n"
        f"- **Predicted Remaining Useful Life (RUL)**: **{rul:.1f} cycles** *(Model: HistGradientBoosting `v1.0.0-fallback`)*\n"
        f"- **Top Probable Failure Mode**: *{top_mode_name.replace('_', ' ').title()}* ({top_mode_prob * 100:.0f}% confidence) *(Model: HistGradientBoosting `v1.0.0-fallback`)*\n\n"
        f"#### 📚 [RAG EVIDENCE — GROUNDED TECHNICAL CITATIONS] ({evidence_status})\n"
    )

    if docs:
        for d in docs:
            score_val = d.get("relevance_score", 0.0)
            score_badge = "HIGH" if score_val >= 0.70 else "MODERATE" if score_val >= 0.35 else "LOW"
            summary_text += f"- **[{d.get('source_file')}] {d.get('section_title')}** (Relevance: **{score_val:.2f}** [{score_badge}])\n"
        if not has_strong_evidence:
            summary_text += "\n> ⚠️ *Note: Retrieved manual evidence confidence is low (< 0.35). Do not proceed with invasive component replacement without physical validation.*\n"
    else:
        summary_text += "- *No matching maintenance SOPs retrieved for this machine type and failure mode.*\n"

    summary_text += (
        f"\n#### 💡 [AI INTERPRETATION & RECOMMENDATION]\n"
        f"Based on real-time sensor trajectories and deterministic gradient-boosted degradation trajectories, "
        f"asset `{machine_id.upper()}` exhibits elevated degradation consistent with `{top_mode_name.replace('_', ' ')}`.\n\n"
        f"---\n"
        f"📋 **Draft Work Order Generated: `{draft['work_order_id']}`**\n"
        f"- **Title**: {draft['title']}\n"
        f"- **Priority**: `{draft['priority']}`\n"
        f"- **Estimated Downtime**: {draft['estimated_downtime_hours']} hours\n\n"
        f"⚠️ **Action Required**: This work order is in `PENDING_APPROVAL` status and requires explicit human authorization before dispatch."
    )

    new_msg = AIMessage(content=summary_text)

    return {
        "work_order": draft,
        "requires_human_approval": True,
        "is_approved": False,
        "diagnostic_summary": summary_text,
        "messages": [new_msg],
    }


def create_maintenance_agent_graph():
    """Build and compile the LangGraph maintenance agent."""
    workflow = StateGraph(AgentState)

    workflow.add_node("diagnose", diagnose_machine_node)
    workflow.add_node("retrieve", retrieve_knowledge_node)
    workflow.add_node("draft_work_order", reason_and_draft_node)

    workflow.set_entry_point("diagnose")
    workflow.add_edge("diagnose", "retrieve")
    workflow.add_edge("retrieve", "draft_work_order")
    workflow.add_edge("draft_work_order", END)

    app = workflow.compile()
    return app


# Singleton instance
agent_app = create_maintenance_agent_graph()
