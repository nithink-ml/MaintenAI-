# MaintenAI— Feature Engineering Specification & Catalog

**Feature Version**: `v1`  
**Pipeline Entrypoint**: [`ml.features.build_features`](file:///Users/nithin/Documents/projects/factorypulse/ml/features/build_features.py)  
**Configuration**: [`params.yaml`](file:///Users/nithin/Documents/projects/factorypulse/params.yaml)  

---

## 1. Executive Summary & Design Principles

The MaintenAIFeature Engineering pipeline produces domain-informed, leakage-free feature sets for the three core predictive maintenance workflows:
1. **C-MAPSS (FD001)** → Turbofan Remaining Useful Life (RUL) regression (~151 columns).
2. **IMS Bearing** → High-frequency vibration anomaly detection (~61 columns).
3. **AI4I 2020** → Multi-mode tool & equipment failure classification (~23 columns).

### Core Architectural Principles

1. **Strict Fit-Then-Transform (Zero Train-Serving Skew)**:
   - Preprocessing transformers (e.g. `StandardScaler`) are fit **strictly on training data**.
   - The fitted transformer is saved as `scaler.joblib` and applied identically to validation, test, and live streaming inference data.
   - Targets, machine IDs, and one-hot flags are excluded from scaling.

2. **Entity & Boundary Isolation (Zero Cross-Unit Contamination)**:
   - All temporal, rolling, lag, EMA, and differencing operations are strictly grouped by `machine_id` (engine unit or bearing ID).
   - Rolling windows and lags **never** cross boundaries between distinct assets.

3. **Causal Trailing Windows Only (Zero Temporal / Future Leakage)**:
   - All rolling and smoothing operations use trailing (right-aligned) windows:
     $$\bar{x}_t = \frac{1}{W} \sum_{i=0}^{W-1} x_{t-i}$$
   - No centered, backward-looking, or future-leaking aggregations are used.
   - Dynamic lifecycle normalizations use causal cumulative maxima: $\max_{k \le t} (\text{cycle}_k)$.

4. **Configurable & Auditable Registry**:
   - Feature parameters (window sizes, EMA spans, lag steps, scaler choices) are centralized in `params.yaml`.
   - Every feature has an explicit entry in `ml.features.registry` with documented physical rationale, source columns, and leakage protection notes.
   - Output parquet files are accompanied by a machine-readable `feature_manifest.json`.

5. **Extensible Scaler Architecture**:
   - The preprocessing module uses a factory pattern supporting `standard` (default v1 `StandardScaler`), `robust` (`RobustScaler`), and `minmax` (`MinMaxScaler`) via parameter configuration without code modification.

---

## 2. C-MAPSS (FD001) Feature Catalog (Turbofan RUL)

### 2.1 Sensor Selection & Priority Grouping
Based on Exploratory Data Analysis (EDA) correlation with Remaining Useful Life ($|\rho| > 0.5$):

| Sensor Column | Description | RUL Correlation ($\rho$) | Feature Treatment |
|---|---|---|---|
| `raw_sensor_15` | Bypass Ratio (BPR) | -0.651 | Rolling (mean, std, min, max), EMA, Lag, Delta |
| `raw_sensor_11` | Physical Fan Speed ($N_f$) | +0.651 | Rolling (mean, std, min, max), EMA, Lag, Delta |
| `raw_sensor_20` | Bleed Enthalpy ($H_{bleed}$) | +0.598 | Rolling (mean, std, min, max), EMA, Lag, Delta |
| `raw_sensor_12` | HPC Outlet Pressure ($P_{30}$) | -0.532 | Rolling (mean, std, min, max), EMA, Lag, Delta |
| `raw_sensor_9` | Physical Core Speed ($N_c$) | +0.506 | Rolling (mean, std, min, max), EMA, Lag, Delta |
| `raw_sensor_17` | Fuel Flow Ratio ($W_f$) | -0.500 | Rolling (mean, std, min, max), EMA, Lag, Delta |

**Secondary Informative Sensors** ($0.2 < |\rho| < 0.5$):
- Sensors: `raw_sensor_4` ($T_{50}$), `raw_sensor_14` ($N_{f,corr}$), `raw_sensor_3` ($T_{30}$), `raw_sensor_2` ($T_{24}$), `raw_sensor_7` ($T_{48}$), `raw_sensor_8` ($W_{cool}$), `raw_sensor_13` ($N_{c,corr}$), `raw_sensor_21` ($BPR_{bleed}$).
- Treatment: Passthrough + 1st order differencing (`delta1` and `delta1_pct`).

### 2.2 Transformations & Parameter Grid
- **Rolling Windows**: $W \in [5, 10, 20]$ cycles. Statistics: `mean`, `std`, `min`, `max`.
- **EMA Spans**: $\alpha \in [10, 30]$ cycles.
- **Lag Steps**: $L \in [1, 5]$ cycles.
- **Delta Periods**: 1 cycle (first difference + percentage change).
- **Lifecycle Position**: `lifecycle_pct` $= \text{cycle}_t / \max_{k \le t}(\text{cycle}_k)$.
- **Target Clipping**: Piecewise-linear RUL target capped at $125$ cycles.

**Total Columns**: 151 (147 scaled features, 3 metadata/target, 1 unscaled identifier).

---

## 3. NASA IMS Bearing Feature Catalog (Anomaly Detection)

### 3.1 Data Considerations & Drop Justifications
1. **Null Sensor Columns**:
   - `temperature`, `pressure`, `rpm`, `current` are **100% null** in IMS because the physical test rig was instrumented with accelerometers only.
   - **Action**: Dropped completely during feature engineering (never fabricated with synthetic defaults).
2. **Redundant Sensor Columns**:
   - `raw_rms` has an exact Pearson correlation $\rho = 1.0$ with `vibration`.
   - **Action**: Dropped to eliminate collinearity.
3. **Rolling Window Sizing**:
   - The interim dataset contains snapshot summaries across run-to-failure cycles.
   - Window sizes configured to **$[3, 5, 10]$** via `params.yaml` with statistics `["mean", "std"]`.

### 3.2 Primary Vibration Features & Transformations
Primary features analyzed for temporal and spectral trends:
- `vibration` (RMS acceleration, $\rho = +0.662$ with anomaly)
- `raw_peak` (Peak acceleration, $\rho = +0.652$)
- `raw_spectral_energy` (FFT spectral energy, $\rho = +0.668$)
- `raw_crest_factor` (Impact indicator $\text{Peak}/\text{RMS}$)

**Transformations**:
1. **Log Transforms**: $\log(1 + x)$ for `vibration`, `raw_peak`, and `raw_spectral_energy` to reduce severe positive skewness ($\text{skew} > 3.8$).
2. **Rolling Statistics**: Windows $[3, 5, 10]$ for `mean` and `std` grouped by bearing ID (`machine_id`).
3. **EMA Features**: Spans $[5, 10]$ grouped by bearing ID.
4. **Delta Features**: First differences capturing acceleration of degradation.
5. **Spectral Ratios**:
   - `spectral_to_rms_ratio` $= \text{spectral\_energy} / \text{vibration}$
   - `peak_to_energy_ratio` $= \text{raw\_peak} / \text{spectral\_energy}$
6. **Spectral Trend Ratios**:
   - Trend ratio over 5 snapshots: $x_t / \text{rolling\_mean}_5(x)$ for vibration, peak, and spectral energy.

**Total Columns**: 61 (57 scaled features, 4 metadata/label).

---

## 4. AI4I 2020 Feature Catalog (Failure Classification)

### 4.1 Engineered Physical Composites
1. **Mechanical Power**:
   $$\text{power} = \text{vibration} \times \text{rpm} \quad (\text{Torque} \times \text{Rotational Speed})$$
   *Physically correlated with Power Failure (PWF) and Overstrain Failure (OSF).*
2. **Torque-to-Speed Ratio**:
   $$\text{torque\_per\_rpm} = \frac{\text{vibration}}{\max(\text{rpm}, \epsilon)}$$
   *Identifies low-speed high-torque overstrain conditions.*
3. **Thermal Dissipation Differential**:
   $$\text{temp\_diff} = \text{pressure} - \text{temperature} \quad (\text{Process Temp} - \text{Air Temp})$$
   *Correlated with Heat Dissipation Failure (HDF).*
4. **Thermal-Speed Interaction**:
   $$\text{temp\_rpm\_interaction} = \text{temperature} \times \text{rpm}$$
5. **Tool Wear Non-Linearity**:
   $$\text{tool_wear_squared} = (\text{operating\_cycle})^2$$
   *Captures accelerating failure probability as cutting tool reaches end-of-life.*
6. **One-Hot Quality Variants**:
   - `product_type_L`, `product_type_M`, `product_type_H` (categorical product type).

### 4.2 Class Imbalance Handling
- Class imbalance (normal vs 5 rare failure modes) is **not modified** during featurization to avoid introducing synthetic sample bias into feature distributions.
- Imbalance is handled at training time via cost-sensitive learning (`class_weight='balanced'` in LightGBM).

**Total Columns**: 23 (10 scaled continuous features, 3 binary one-hot features, 5 individual binary failure targets, 2 multiclass/label columns, 3 metadata).

---

## 5. Summary Schema & Split Metrics

| Dataset | Split | Row Count | Total Columns | Scaled Feature Count | Missing / Inf Count |
|---|---|---|---|---|---|
| **C-MAPSS** | Train | 9,908 | 151 | 147 | 0 (0.0%) |
| | Val | 1,641 | 151 | 147 | 0 (0.0%) |
| | Test | 2,624 | 151 | 147 | 0 (0.0%) |
| **IMS** | Train | 132 | 61 | 57 | 0 (0.0%) |
| | Val | 28 | 61 | 57 | 0 (0.0%) |
| | Test | 40 | 61 | 57 | 0 (0.0%) |
| **AI4I** | Train | 1,300 | 23 | 10 | 0 (0.0%) |
| | Val | 300 | 23 | 10 | 0 (0.0%) |
| | Test | 400 | 23 | 10 | 0 (0.0%) |

---

## 6. Verification and Determinism Checklist

- [x] **Zero NaN / Infinite Values**: Verified across all 9 output splits.
- [x] **Row Count Integrity**: Processed dataset row counts match featurized row counts exactly.
- [x] **Entity Boundaries Preserved**: Rolling/EMA/Lag strictly isolated per machine / bearing ID.
- [x] **Strict Temporal Order**: Verified strictly increasing chronological cycle index per entity.
- [x] **Zero Data Leakage**: Scaler fit strictly on training partitions and serialized to `scaler.joblib`.
- [x] **Deterministic Re-execution**: Output Parquet files generate identical SHA256 and DataFrame comparisons on repeated pipeline runs.
- [x] **Automated Test Suite**: 114 passing unit and integration tests across [`tests/unit/test_features.py`](file:///Users/nithin/Documents/projects/factorypulse/tests/unit/test_features.py) and [`tests/ml/test_feature_pipeline.py`](file:///Users/nithin/Documents/projects/factorypulse/tests/ml/test_feature_pipeline.py).
