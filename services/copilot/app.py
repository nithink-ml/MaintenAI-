"""
MaintenAI— Maintenance Copilot Service (FastAPI)

Exposes interactive diagnostic chat with the LangGraph Maintenance Agent,
RAG-grounded citations, and human-in-the-loop work order approvals.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agents.maintenance_agent.graph import agent_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MaintenAIMaintenance Copilot Service",
    version="0.1.0",
    description="Agentic AI copilot orchestrating physical machine diagnostics and COPIED to the API gateway work-order store",
)

# Persistent in-memory storage for agent-generated work orders.
# The API gateway reads this store directly (see services.api_gateway.app)
# so agent-created drafts are visible and approvable through the gateway.
_WORK_ORDERS: dict[str, dict[str, Any]] = {}


class ChatRequest(BaseModel):
    message: str
    machine_id: str | None = None
    session_id: str | None = "default-session"


class ChatResponse(BaseModel):
    response: str
    machine_id: str | None = None
    ml_health: dict[str, Any] | None = None
    retrieved_docs: list[dict[str, Any]] = Field(default_factory=list)
    work_order: dict[str, Any] | None = None
    requires_human_approval: bool = False


class WorkOrderApprovalRequest(BaseModel):
    work_order_id: str
    action: str = "APPROVE"  # APPROVE | REJECT
    notes: str | None = None


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "copilot"}


@app.post("/api/v1/copilot/chat", response_model=ChatResponse)
def chat_with_copilot(req: ChatRequest) -> ChatResponse:
    """Interact with the LangGraph Maintenance Copilot Agent."""
    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "machine_id": req.machine_id,
        "retrieved_docs": [],
    }

    try:
        final_state = agent_app.invoke(initial_state)

        # Record work order if generated
        wo = final_state.get("work_order")
        if wo and "work_order_id" in wo:
            _WORK_ORDERS[wo["work_order_id"]] = wo

        messages = final_state.get("messages", [])
        last_ai_msg = messages[-1].content if messages else "Analysis complete."

        return ChatResponse(
            response=str(last_ai_msg),
            machine_id=final_state.get("machine_id"),
            ml_health=final_state.get("ml_health"),
            retrieved_docs=final_state.get("retrieved_docs", []),
            work_order=wo,
            requires_human_approval=final_state.get("requires_human_approval", False),
        )
    except Exception as e:
        logger.error("Error executing agent: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/copilot/work_orders")
def list_work_orders() -> dict[str, Any]:
    """List all created maintenance work orders."""
    return {"work_orders": list(_WORK_ORDERS.values()), "total": len(_WORK_ORDERS)}


@app.post("/api/v1/copilot/work_orders/approve")
def approve_work_order(req: WorkOrderApprovalRequest) -> dict[str, Any]:
    """Request COPILOT work-order state change and mirror it into the gateway store.

    This endpoint does NOT authorize the action itself; the API gateway enforces
    JWT + RBAC on the original approval/rejection routes. This service performs
    idempotency checks (already approved/rejected) and mirrors status changes.
    """
    if req.work_order_id not in _WORK_ORDERS:
        raise HTTPException(status_code=404, detail=f"Work Order {req.work_order_id} not found")

    wo = _WORK_ORDERS[req.work_order_id]
    new_status: str
    if req.action.upper() == "APPROVE":
        new_status = "APPROVED"
    elif req.action.upper() == "REJECT":
        new_status = "REJECTED"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported action '{req.action}'")

    if wo["status"] == new_status:
        raise HTTPException(
            status_code=409,
            detail=f"Work Order {req.work_order_id} is already {new_status}",
        )

    wo["status"] = new_status
    if new_status == "APPROVED":
        wo["approved_by"] = req.notes if req.notes else "autosynced by copilot service"
        wo["approval_notes"] = req.notes
    else:
        wo["rejected_by"] = req.notes if req.notes else "autosynced by copilot service"
        wo["rejection_notes"] = req.notes

    logger.info("Copilot Work Order %s -> %s", req.work_order_id, new_status)
    return {"status": "success", "work_order": wo}


# Exposed helpers used when the API gateway enforces RBAC through the gateway
# (the gateway is the authoritative approver for copilot work orders).

def approve_work_order_internal(
    work_order_id: str, *, approved_by: str | None = None, notes: str | None = None
) -> dict[str, Any] | None:
    if work_order_id not in _WORK_ORDERS:
        return None
    wo = _WORK_ORDERS[work_order_id]
    if wo["status"] == "APPROVED":
        return {"error": f"Work Order {work_order_id} is already APPROVED by '{wo.get('approved_by')}'"}
    if wo["status"] != "PENDING_APPROVAL":
        return {"error": f"Work Order {work_order_id} is in status '{wo['status']}'; only PENDING_APPROVAL can be approved"}
    wo["status"] = "APPROVED"
    wo["approved_by"] = approved_by or "unspecified"
    wo["approval_notes"] = notes
    return wo


def reject_work_order_internal(
    work_order_id: str, *, rejected_by: str | None = None, notes: str | None = None
) -> dict[str, Any] | None:
    if work_order_id not in _WORK_ORDERS:
        return None
    wo = _WORK_ORDERS[work_order_id]
    if wo["status"] == "REJECTED":
        return {"error": f"Work Order {work_order_id} is already REJECTED"}
    if wo["status"] != "PENDING_APPROVAL":
        return {"error": f"Work Order {work_order_id} is in status '{wo['status']}'; only PENDING_APPROVAL can be rejected"}
    wo["status"] = "REJECTED"
    wo["rejected_by"] = rejected_by or "unspecified"
    wo["rejection_notes"] = notes
    return wo

# Backward-compatible aliases kept for the API gateway's import below.
_approve_copilot_work_order_internal = approve_work_order_internal
_reject_copilot_work_order_internal = reject_work_order_internal
