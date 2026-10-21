# MaintenAI— Exploratory Data Analysis

Automated EDA reports for all three canonical datasets used in FactoryPulse.

## Dataset Reports

| Dataset | Purpose | Report |
|---------|---------|--------|
| NASA C-MAPSS (FD001) | RUL Prediction | [View Report](cmapss/cmapss_eda.md) |
| NASA IMS Bearing | Anomaly Detection | [View Report](ims/ims_eda.md) |
| AI4I 2020 | Failure-Mode Classification | [View Report](ai4i/ai4i_eda.md) |

## Generated Artifacts

### NASA C-MAPSS (FD001)

- [correlation](cmapss/cmapss_correlation_heatmap.png)
- [degradation](cmapss/cmapss_degradation_curves.png)
- [distributions](cmapss/cmapss_sensor_distributions.png)
- [lifecycle](cmapss/cmapss_lifecycle_lengths.png)
- [report](cmapss/cmapss_eda.md)
- [rul](cmapss/cmapss_rul_distribution.png)

### NASA IMS Bearing

- [correlation](ims/ims_feature_correlation.png)
- [kurtosis](ims/ims_kurtosis_evolution.png)
- [normal_vs_anomalous](ims/ims_normal_vs_anomalous.png)
- [report](ims/ims_eda.md)
- [rms_trends](ims/ims_vibration_rms_trends.png)
- [spectral_energy](ims/ims_spectral_energy.png)

### AI4I 2020

- [correlation](ai4i/ai4i_correlation_heatmap.png)
- [failure_dist](ai4i/ai4i_failure_distribution.png)
- [failure_features](ai4i/ai4i_failure_feature_comparison.png)
- [features_by_type](ai4i/ai4i_features_by_type.png)
- [imbalance](ai4i/ai4i_class_imbalance.png)
- [report](ai4i/ai4i_eda.md)
