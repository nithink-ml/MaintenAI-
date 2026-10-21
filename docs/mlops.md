# FactoryPulse: MLOps, Versioning & Drift Monitoring

## 1. Data Version Control (DVC)
DVC tracks raw datasets, processed feature tables, and pipeline stages defined in `dvc.yaml`.

```bash
# Run full DVC pipeline
dvc repro

# Check pipeline status
dvc status
```

## 2. Model Registry & Training Artifacts
Model checkpoints are serialized with metadata, normalization parameters, and feature schemas:
- `models/anomaly/isolation_forest.joblib`
- `models/anomaly/lstm_autoencoder.joblib`
- `models/rul/xgboost.joblib`
- `models/rul/transformer.joblib`
- `models/failure/lightgbm.joblib`

## 3. Drift Monitoring
Monitors Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests:
- $\text{PSI} < 0.10$: Stable baseline
- $0.10 \le \text{PSI} < 0.25$: Moderate drift (Warning alert)
- $\text{PSI} \ge 0.25$: Severe drift (Triggers automated retraining pipeline)
