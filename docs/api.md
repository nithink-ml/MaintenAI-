# FactoryPulse: REST API Reference

The MaintenAIAPI Gateway aggregates microservices and exposes standardized REST endpoints:

## Ingestion Endpoints
- `POST /api/v1/telemetry`: Ingest single sensor reading (JSON).
- `POST /api/v1/telemetry/batch`: Ingest batch of sensor readings.
- `GET /api/v1/telemetry/latest/{machine_id}`: Retrieve latest verified state.

## Inference Endpoints
- `POST /api/v1/predict/anomaly`: Compute continuous anomaly score [0.0 - 1.0].
- `POST /api/v1/predict/rul`: Predict Remaining Useful Life cycles.
- `POST /api/v1/predict/failure`: Estimate failure probability and mode breakdown.
- `POST /api/v1/predict/health_assessment`: Consolidated multi-model report.

## Fleet & Alert Endpoints
- `GET /api/v1/fleet/status`: Consolidated fleet-wide status matrix.
- `GET /api/v1/alerts`: List active deterministic alerts.
- `POST /api/v1/alerts/{alert_id}/ack`: Acknowledge alert.

## Copilot & Work Order Endpoints
- `POST /api/v1/copilot/chat`: Conversational diagnostic assistant.
- `GET /api/v1/copilot/work_orders`: List all draft and approved work orders.
- `POST /api/v1/copilot/work_orders/approve`: Human authorization gate.
