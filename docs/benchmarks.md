# MaintenAI— System Performance & Latency Benchmark Report

**Date**: 2026-09-23T10:42:54.097846+00:00  
**Hardware Environment**: macOS-26.5.2-arm64-arm-64bit | arm  
**Python Runtime**: 3.11.15  

---

## 1. Development / Synthetic Stream Processing Benchmarks

Empirically measured using real-time HTTP callers against live microservices and in-memory stores. All measurements report actual p50, p95, and p99 percentiles in milliseconds:

| Pipeline Stage / Endpoint | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) |
|---|---|---|---|---|---|
| **Telemetry Ingestion API** (`POST /api/v1/telemetry`) | 0.548 | 0.532 | 0.662 | 0.75 | ~1879.7 req/s |
| **Feature Store (Window=10)** | 0.305 | 0.288 | 0.444 | 0.549 | — |
| **Feature Store (Window=30)** | 0.295 | 0.29 | 0.338 | 0.416 | — |
| **Feature Store (Window=50)** | 0.303 | 0.293 | 0.359 | 0.54 | — |
| **ML Multi-Model Inference** (`POST /api/v1/predict`) | 29.974 | 26.469 | 57.913 | 69.506 | ~37.8 req/s |
| **Deterministic Alert Engine** | 0.002 | 0.002 | 0.003 | 0.056 | > 10,000 req/s |
| **Hybrid RAG Retrieval** (`search_manuals`) | 0.034 | 0.033 | 0.049 | 0.049 | — |
| **Gateway Model Metadata** (`GET /models/metadata`) | 0.931 | 0.925 | 1.07 | 1.203 | — |
| **Gateway Fleet Status** (`GET /fleet/status`) | 6.669 | 7.382 | 8.85 | 9.373 | — |
| **Gateway Live System Status** (`GET /system/status`) | 15.955 | 15.027 | 28.676 | 28.676 | — |

---

## 2. Real / Public Dataset Model Performance Validation

Separate offline test-set evaluation results across the 3 authoritative public benchmark datasets. These numbers reflect model generalization on historical test data and are **NEVER** combined with real-time stream latency:

### A. NASA C-MAPSS (FD001 Turbofan Run-to-Failure)
- **Active Model**: `HistGradientBoostingRegressor` (133 features)
  - **RMSE**: **87.57 cycles**
  - **MAE**: **67.28 cycles**
  - **NASA Asymmetric Score**: **1.19e+11**
- **Baseline Model**: PyTorch Transformer RUL
  - **RMSE**: 137.10 cycles
  - **MAE**: 117.99 cycles
  - **NASA Asymmetric Score**: 8.71e+13
- **Outcome**: The tabular gradient-boosted trees achieved 36.1% lower RMSE than the Transformer baseline on tabular vibration/sensor features.

### B. NASA IMS Bearing (Run-to-Failure Vibration)
- **Active Model**: `IsolationForestAnomalyDetector` (53 features)
  - **Accuracy**: **82.50%**
  - **Precision**: **92.31%**
  - **Recall**: **66.67%**
  - **F1-Score**: **0.7742**
  - **ROC-AUC**: **0.8510**
- **Baseline Model**: PyTorch LSTM Autoencoder
  - **Accuracy**: 45.00%
  - **F1-Score**: 0.6207
  - **ROC-AUC**: 0.5000
- **Outcome**: Isolation Forest delivered a superior F1-score (0.774 vs 0.621) and ROC-AUC (0.851 vs 0.500).

### C. AI4I 2020 Predictive Maintenance (Multi-Target Failure Classification)
- **Active Model**: `HistGradientBoostingClassifier` (13 features)
  - **ROC-AUC Macro**: **0.9346**
  - **PR-AUC**: **0.5758**
  - **F1 Macro**: **0.7143**
  - **Individual Failure Modes**:
    - Heat Dissipation Failure (HDF): ROC-AUC = **0.996**, F1 = **0.667**
    - Tool Wear Failure (TWF): ROC-AUC = **0.955**, F1 = **0.634**
    - Overstrain Failure (OSF): ROC-AUC = **0.962**, F1 = **0.500**
    - Power Failure (PWF): ROC-AUC = **0.985**, F1 = **0.333**
