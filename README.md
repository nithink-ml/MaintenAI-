# MaintenAI: Real-Time Predictive Maintenance & Equipment Intelligence Platform

[![CI](https://github.com/nithink-ml/MaintenAI-/actions/workflows/ci.yml/badge.svg)](https://github.com/nithink-ml/MaintenAI-/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 14](https://img.shields.io/badge/frontend-Next.js%2014-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MaintenAI** is an enterprise-grade industrial predictive maintenance and equipment intelligence platform. It bridges continuous physical sensor telemetry with deterministic machine learning, deep learning time-series models, vector-backed retrieval-augmented generation (RAG), and human-in-the-loop agentic AI workflows.

---

## 🏗️ Architecture Overview

```mermaid
graph TD
    subgraph "Edge / Simulation"
        SIM[Sensor Simulator Service]
    end

    subgraph "Ingestion & Storage"
        ING[Ingestion Service (FastAPI)]
        TSDB[(TimescaleDB Time-Series DB)]
        REDIS[(Redis Cache / Feature Store)]
    end

    subgraph "Feature & ML Pipelines"
        FP[Feature Pipeline Service]
        DVC[DVC Data Versioning]
        MLFLOW[MLflow Model Registry]
        ANOM[Anomaly Detection Model<br/>Isolation Forest / LSTM Autoencoder]
        RUL[RUL Prediction Model<br/>HistGradientBoosting / Transformer]
        FAIL[Failure Classifier<br/>HistGradientBoosting / LightGBM]
    end

    subgraph "Serving & Alerting"
        INF[Inference Service (FastAPI)]
        ALERT[Alert Engine Service]
    end

    subgraph "Knowledge & Agentic AI"
        QDRANT[(Qdrant Vector DB)]
        RAG[RAG Knowledge Pipeline]
        AGENT[LangGraph Maintenance Agent]
        LLM[LLM Engine]
    end

    subgraph "Control & Interface"
        GW[API Gateway / Auth Middleware]
        FE[Operations Dashboard (Next.js 14)]
        HUMAN[Human Reliability Engineer]
    end

    SIM -->|HTTP / MQTT Telemetry| ING
    ING -->|Validate & Write Telemetry| TSDB
    ING -->|Cache Machine State| REDIS
    TSDB -->|Raw Telemetry| FP
    FP -->|Calculated Features| REDIS
    FP -->|Training Datasets| DVC
    
    DVC --> ANOM & RUL & FAIL
    ANOM & RUL & FAIL -->|Artifacts & Metrics| MLFLOW
    MLFLOW -->|Load Weights| INF

    REDIS -->|Online Features| INF
    TSDB -->|Historical Windows| INF
    INF -->|Health Scores| ALERT
    INF -->|Predictions API| GW

    ALERT -->|Trigger Alerts| GW
    GW --> FE

    FE -->|Copilot Query / Work Order| AGENT
    AGENT -->|1. Machine Telemetry| INF
    AGENT -->|2. Search SOPs & Manuals| RAG
    RAG -->|Vector Search & Rerank| QDRANT
    AGENT -->|3. Grounded Diagnostics| LLM
    AGENT -->|4. Propose Work Order Draft| FE

    FE -->|5. Review & Authorize| HUMAN
    HUMAN -->|6. Execute Approved Work Order| TSDB
```

---

## ⚙️ Core Engineering Principles

### Unidirectional Deterministic-First AI Pipeline
MaintenAIfollows a strict unidirectional separation of duties:
```
DATA → FEATURES → ML MODELS → EVALUATION → MODEL REGISTRY → INFERENCE → REAL-TIME DECISION → RAG → AGENT → HUMAN APPROVAL → ACTION
```

* **Deterministic & ML Models** calculate factual machine telemetry: Remaining Useful Life (RUL), anomaly scores, vibration statistics, and specific failure mode probabilities.
* **LLMs and Agents** never invent telemetry, modify thresholds arbitrarily, or generate statistical boundaries.
* **LangGraph Agent & RAG** act as an explainability and operational orchestration layer: retrieving authoritative maintenance SOPs, explaining machine diagnoses in plain language, preparing draft work orders, and strictly requiring human authorization before dispatching work orders or executing actions.

---

## 🚀 Key Modules & Capabilities

| Module | Description | Tech Stack |
|---|---|---|
| **Real-Time Telemetry & Edge Simulator** | Multi-machine streaming simulation generating realistic physics-based vibration, temperatures, tool wear, and pressure. | Python, FastAPI, MQTT, Pydantic |
| **Storage & Feature Store** | High-throughput time-series storage and sub-millisecond online feature caching. | TimescaleDB, Redis, SQLAlchemy |
| **Predictive ML Pipelines** | Unsupervised anomaly detection, multi-class failure classification, and remaining useful life (RUL) estimation. | Scikit-learn, LightGBM, XGBoost, PyTorch, MLflow, DVC |
| **Industrial RAG Pipeline** | Dense vector search with Cross-Encoder reranking over equipment manuals and standard operating procedures. | Qdrant, SentenceTransformers, LangChain |
| **Autonomous Copilot** | Multi-step agent graph equipped with diagnostic tools, safety guardrails, and human-in-the-loop approval gates. | LangGraph, Pydantic, Python |
| **Operations Dashboard** | Real-time fleet health matrix, component degradation trends, live telemetry diagnostics, and interactive AI copilot. | Next.js 14, Tailwind CSS, TypeScript, Lucide |

---

## 📊 Datasets & Models

MaintenAI integrates three standardized industrial predictive maintenance benchmarks:
1. **NASA C-MAPSS**: Turbofan engine degradation simulation for Remaining Useful Life (RUL) regression.
2. **NASA IMS Bearing**: Run-to-failure vibration acceleration dataset for temporal/spectral anomaly detection.
3. **AI4I 2020 Predictive Maintenance**: Real-world milling machine dataset for multi-class failure mode classification (Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Random Failures).

---

## 📁 Repository Structure

```
.
├── agents/                     # LangGraph maintenance agent graph, tools, guardrails
│   └── maintenance_agent/      # State machine, human approval gates, diagnostic tools
├── apps/
│   ├── frontend/               # Next.js 14 operations dashboard & copilot UI
│   └── operations-console/     # Operations management console
├── benchmarks/                 # System latency, throughput & model benchmark scripts
├── data/
│   ├── knowledge/              # SOPs & troubleshooting manuals for RAG ingestion
│   ├── raw/                    # Raw industrial datasets (DVC-managed)
│   ├── interim/                # Intermediate parsed datasets
│   └── processed/              # Engineered feature matrices & Parquet stores
├── docs/                       # Architectural specifications, EDA reports, system audits
│   ├── architecture.md         # Comprehensive system architecture & data contracts
│   ├── final_system_audit.md   # System audit report & production validation
│   ├── eda/                    # Exploratory Data Analysis figures & reports
│   └── runbook.md              # Operational troubleshooting & production runbook
├── factorypulse/               # Shared core library (schemas, database, logging, config)
├── ml/
│   ├── anomaly/                # Isolation Forest & LSTM Autoencoder pipelines
│   ├── common/                 # Dataset loaders, EDA scripts, data preparation
│   ├── evaluation/             # Model evaluation suites, drift monitoring
│   ├── failure_classification/ # Multi-class failure prediction (LightGBM/HistGBM)
│   ├── features/               # Feature registry, spectral & temporal extraction
│   └── rul/                    # Remaining Useful Life regression (XGBoost/Transformer)
├── models/                     # Model artifact registry directories
├── notebooks/                  # Interactive EDA, modeling, and RAG exploration
├── rag/                        # Chunking, embeddings, vector indexing, hybrid retrieval
├── scripts/                    # End-to-end audit and deployment verification scripts
├── services/
│   ├── alert_service/          # Automated threshold & anomaly alert dispatch
│   ├── api_gateway/            # Unified API gateway, auth, and model registry
│   ├── copilot/                # AI copilot service exposing agent endpoints
│   ├── feature_service/        # Real-time feature extraction service
│   ├── inference/              # Low-latency model serving service
│   ├── ingestion/              # High-throughput telemetry ingestion service
│   ├── realtime/               # Real-time end-to-end processing pipeline
│   └── simulator/              # Multi-machine synthetic sensor telemetry simulator
├── tests/                      # Full test suite (unit, integration, ML, RAG, API, E2E)
├── docker-compose.yml          # Container orchestration (DB, Cache, Services, UI)
├── dvc.yaml                    # DVC pipeline stages for reproducible ML
├── Makefile                    # Standard developer automation commands
└── pyproject.toml              # Python project configuration and dependencies
```

---

## ⚡ Quickstart

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend)
- Docker & Docker Compose (for database and services)

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/nithink-ml/MaintenAI-.git
cd MaintenAI-

# Copy environment variables
cp .env.example .env

# Create and activate Python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install Python package in editable mode with development dependencies
pip install -e ".[dev,notebooks]"
```

### 2. Start Infrastructure Services

```bash
# Start TimescaleDB, Redis, Qdrant, and MLflow
docker compose up -d timescaledb redis qdrant mlflow
```

### 3. Run Data & Training Pipelines

```bash
# Prepare datasets and extract features
make data-prepare

# Train anomaly, RUL, and failure classification models
make train-all

# Ingest technical documentation into Qdrant vector database
python -m rag.ingestion.run
```

### 4. Start Development Services

```bash
# Start the telemetry simulator
make simulate

# In separate terminals, launch the services:
make dev-ingestion   # Port 8003
make dev-inference   # Port 8001
make dev-api         # Port 8000

# Start Next.js frontend
cd apps/frontend
npm install
npm run dev          # http://localhost:3000
```

---

## 🧪 Testing & Validation

```bash
# Run full Pytest suite
make test

# Run unit tests only
make test-unit

# Run end-to-end platform audit
python scripts/verify_e2e_audit.py

# Run latency & throughput benchmarks
python benchmarks/run_benchmarks.py
```

---

## 🛡️ Security & Auditing

- **Input Validation**: Strict Pydantic models for all incoming machine telemetry.
- **Role-Based Access Control**: JWT-based bearer authentication and scope verification in API Gateway.
- **Human-in-the-Loop**: Destructive maintenance actions and work order execution require verified engineer authorization.
- **Security Scanners**: Automated Bandit security scans and Ruff linting integrated into GitHub Actions CI.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
