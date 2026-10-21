import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

df_train = pd.read_parquet("data/processed/ai4i_features/train.parquet")
df_test = pd.read_parquet("data/processed/ai4i_features/test.parquet")

exclude_cols = {
    "machine_id",
    "timestamp",
    "failure_label",
    "failure_tool_wear",
    "failure_heat_dissipation",
    "failure_power_failure",
    "failure_overstrain",
    "failure_random",
    "failure_mode",
    "product_type",
}
feature_cols = [c for c in df_train.columns if c not in exclude_cols]

X_train = df_train[feature_cols]
y_train = df_train["failure_label"]
X_test = df_test[feature_cols]
y_test = df_test["failure_label"]

clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
}

metrics_path = Path("ml/failure_classification/metrics_baseline.json")
metrics_path.parent.mkdir(parents=True, exist_ok=True)
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("Baseline AI4I Metrics:", json.dumps(metrics, indent=2))
