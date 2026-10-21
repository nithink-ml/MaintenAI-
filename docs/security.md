# MaintenAI— Security Architecture, RBAC & AI Safety Audit

**Date**: September 23, 2026  
**Status**: Verified & Audited  
**Static Analysis Tool**: Bandit 1.9.4 (SAST Passed: 0 High, 0 Medium)  

---

## 1. Authentication & Role-Based Access Control (RBAC)

MaintenAIimplements cryptographically secure, stateless authentication using JSON Web Tokens (HMAC-SHA256).

### Role Hierarchy & Permissions Matrix

| Role | Telemetry & Health Inspection | Alert Acknowledgment | Copilot Diagnostics | Work Order Creation (Draft) | Work Order Authorization & Approval | Model Management & Config |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Operator** | Read-Only | Yes | No | No | No | No |
| **Technician** | Read-Only | Yes | Yes | Yes (Draft) | **DENIED** | No |
| **Engineer** | Read-Only | Yes | Yes | Yes (Draft) | **AUTHORIZED** | No |
| **Admin** | Read-Write | Yes | Yes | Yes (Draft) | **AUTHORIZED** | **AUTHORIZED** |

### Authorization Gates
- **Work Order Approval Gate**:
  - Endpoint: `POST /api/v1/work-orders/{id}/approve`
  - Protection: `require_role([UserRole.ENGINEER, UserRole.ADMIN])`
  - Audit Trail: Immutable records storing `approved_by`, `approved_at`, and `notes`. Work orders cannot jump directly to `APPROVED` upon creation by AI agents.

---

## 2. Infrastructure & Credential Security

1. **Database & Cache Isolation**:
   - TimescaleDB and Redis operate in dedicated Docker networks (`factorypulse-net`).
   - Redis requires password authentication (`--requirepass ${REDIS_PASSWORD}`).
   - Connection pools utilize strict timeouts (2.0s max) preventing thread exhaustion attacks.
2. **Environment Variable Hygiene**:
   - All credentials and secrets are injected via `.env`.
   - `.env.example` provides sanitized template values only.
3. **Static Analysis & Vulnerability Scanning**:
   - Bandit SAST scan executed across `factorypulse/`, `services/`, `ml/`, `rag/`, and `agents/`.
   - Result: **0 High-severity issues, 0 Medium-severity issues**.
   - Cryptographic hashing enforces `usedforsecurity=False` on non-cryptographic content hashes (chunk hashing, term hashing).

---

## 3. Agent Safety & Prompt Injection Mitigation

1. **Deterministic Separation of Concerns**:
   - Machine learning inference (Anomaly scores, Remaining Useful Life, Failure mode probabilities) is performed **strictly** by deterministic Scikit-Learn models.
   - The LLM copilot acts purely as an explainer and orchestration agent; it has **no mathematical authority** to invent or compute RUL or anomaly scores.
2. **Guardrail Numeric Grounding**:
   - Output text is validated by `validate_grounding` in `agents/maintenance_agent/guardrails.py`.
   - Any stated RUL or anomaly metric must match tool outputs within tolerance, preventing model hallucination.
3. **Low-Confidence Disclaimers**:
   - RAG retrieval scores are scaled to $[0.0, 1.0]$ via Reciprocal Rank Fusion.
   - If maximum retrieval score is $< 0.35$, the agent is required by guardrail to append an explicit caution notice stating that manual evidence is weak and physical validation is required.
4. **Tool Execution Boundaries**:
   - Tools are limited to read-only SCADA queries (`get_machine_telemetry`, `get_ml_health_assessment`), manual searches (`search_maintenance_manuals`), and drafting non-executable work order requests (`draft_work_order`).
   - No arbitrary shell commands, code execution, or unrestricted database queries are accessible to the agent.
