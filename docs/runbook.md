# MaintenAI— Operations & Deployment Runbook

Comprehensive operational guide for deploying, managing, monitoring, and troubleshooting the MaintenAIpredictive maintenance platform.

---

## 1. Infrastructure Services (Docker Compose)

### Start Infrastructure Containers
```bash
# Ensure .env is present
cp -n .env.example .env

# Start core infrastructure (TimescaleDB, Redis, Qdrant, MLflow)
docker compose up -d timescaledb redis qdrant mlflow

# Inspect container health
docker compose ps
```

### Verified Port Mappings

| Service | Container Name | Host Port | Internal Port | Health Check |
|---|---|---|---|---|
| **TimescaleDB** | `fp-timescaledb` | `5432` | `5432` | `pg_isready -U factorypulse` |
| **Redis** | `fp-redis` | `6379` | `6379` | `redis-cli -a factorypulse_dev ping` |
| **Qdrant** | `fp-qdrant` | `6333` / `6334` | `6333` | `curl -f http://localhost:6333/healthz` |
| **MLflow UI** | `fp-mlflow` | `5001` | `5000` | `curl -f http://localhost:5001/health` |

*(Note: MLflow external port is remapped to `5001` to prevent host collision with macOS AirPlay Receiver on port 5000).*

### Database Hypertable Initialization
```bash
# Execute schema DDL, verify sensor_readings hypertable, and seed machine fleet
.venv/bin/python3 -m factorypulse.db.init_db
```

---

## 2. Local Application Development Stack

### Start Microservices
```bash
# Terminal 1: Sensor Replay Simulator (10 machines, 200ms rate)
.venv/bin/python3 -m services.simulator --machines 10 --rate 200

# Terminal 2: API Gateway & SCADA Routing (Port 8000)
.venv/bin/uvicorn services.api_gateway.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3: Operations Frontend Dashboard (Port 3000)
cd apps/frontend && npm run dev
```

### Verify Dynamic Service Status
```bash
curl -s http://localhost:8000/api/v1/system/status | jq .
curl -s http://localhost:8000/api/v1/models/metadata | jq .
```

---

## 3. Production Verification & Testing

### Complete Automated Verification
```bash
# 1. Dual-Validation Audit (Synthetic Pipeline + Real Datasets)
.venv/bin/python3 scripts/verify_e2e_audit.py

# 2. Performance & Latency Benchmark
.venv/bin/python3 benchmarks/run_benchmarks.py

# 3. Security Vulnerability Scan (SAST)
.venv/bin/bandit -r MaintenAIservices ml rag agents -ll

# 4. Code Quality & Formatting
.venv/bin/ruff check .

# 5. Full Pytest Suite (173 Tests)
.venv/bin/pytest tests/ -q

# 6. Frontend Production Build
cd apps/frontend && npm run lint && npm run build
```

---

## 4. Troubleshooting Guide

### Port 5432 / 6379 Already in Use
If local PostgreSQL or Redis is running on the host:
```bash
# Stop local services or specify alternate ports in .env:
POSTGRES_PORT=5433
REDIS_PORT=6380
```

### MLflow Port 5000 Collision on macOS
macOS Control Center uses port 5000 for AirPlay Receiver. MaintenAIbinds MLflow externally to port 5001:
```bash
# Access MLflow Web UI at:
open http://localhost:5001
```

### Resetting TimescaleDB & Hypertable
```bash
# Tear down database container and volume
docker compose down -v timescaledb
docker compose up -d timescaledb

# Re-run schema initialization
.venv/bin/python3 -m factorypulse.db.init_db
```

### Inspecting Redis Live State Cache
```bash
docker exec -it fp-redis redis-cli -a factorypulse_dev KEYS "machine:*"
docker exec -it fp-redis redis-cli -a factorypulse_dev HGETALL "machine:M001:state"
```
