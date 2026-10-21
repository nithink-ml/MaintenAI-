# FactoryPulse: AI & Machine Learning Specification

## 1. Machine Learning Model Catalog

MaintenAIdeploys three model families strictly separating deterministic mathematical/statistical predictions from natural language orchestration.

| Task | Baseline Model | Production Model | Target Dataset | Primary Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Anomaly Detection** | Scikit-Learn Isolation Forest | PyTorch LSTM Autoencoder | NASA IMS Bearing | F1-Score &gt;= 0.75, FPR &lt;= 0.10 |
| **RUL Prediction** | XGBoost Regressor | PyTorch Transformer Encoder | NASA C-MAPSS (FD001) | RMSE &lt;= 20.0, NASA Score |
| **Failure Classification** | Scikit-Learn Random Forest | LightGBM Multi-target | AI4I 2020 Predictive Maint. | Macro-F1 &gt;= 0.80, Precision &gt;= 0.75 |

---

## 2. Model Architectures & Training Protocols

### 2.1 Anomaly Detection: Isolation Forest & LSTM Autoencoder
- **Isolation Forest**: Fits random partitioning trees over temporal/spectral feature spaces. Anomaly score inverted and mapped to $[0.0, 1.0]$.
- **LSTM Autoencoder**: Encodes sliding multivariate sequences $(B, L=50, D)$ to latent space $(B, H=64)$ and reconstructs input. Anomaly score calculated from Mean Squared Error (MSE) reconstruction loss calibrated against the 95th percentile training baseline.

### 2.2 RUL Prediction: XGBoost & Transformer
- **XGBoost**: Monotonic tree splits on rolling aggregations and degradation indices. Early stopping on validation RMSE.
- **Transformer Encoder**: 3-layer Transformer with multi-head self-attention ($d_{model}=64, \text{heads}=4$), sinusoidal positional encodings, and GELU activation predicting remaining operational cycles.

### 2.3 Failure Classification: LightGBM
- Predicts binary failure flag and specific physical failure modes:
  - **Tool Wear Failure (TWF)**
  - **Heat Dissipation Failure (HDF)**
  - **Power Failure (PWF)**
  - **Overstrain Failure (OSF)**
  - **Random Failure (RNF)**
