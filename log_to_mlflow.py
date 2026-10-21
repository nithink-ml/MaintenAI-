import json

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def log_model(experiment, run_name, metrics_file, dataset):
    mlflow.set_experiment(experiment)
    with open(metrics_file) as f:
        metrics = json.load(f)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("dataset", dataset)
        mlflow.log_param("model", run_name)

        flat_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, dict):
                        for sub_sub_k, sub_sub_v in sub_v.items():
                            flat_metrics[f"{k}_{sub_k}_{sub_sub_k}"] = sub_sub_v
                    else:
                        flat_metrics[f"{k}_{sub_k}"] = sub_v
            elif isinstance(v, list):
                mlflow.log_param(k, str(v))
            else:
                flat_metrics[k] = v
        mlflow.log_metrics(flat_metrics)


try:
    log_model("RUL_Prediction", "XGBoost", "ml/rul/metrics_xgboost.json", "C-MAPSS")
    log_model("RUL_Prediction", "Transformer", "ml/rul/metrics_transformer.json", "C-MAPSS")
    log_model("Anomaly_Detection", "IsolationForest", "ml/anomaly/metrics_iforest.json", "IMS")
    log_model("Anomaly_Detection", "LSTM_Autoencoder", "ml/anomaly/metrics_lstm.json", "IMS")
    log_model(
        "Failure_Classification",
        "LogisticRegression",
        "ml/failure_classification/metrics_baseline.json",
        "AI4I",
    )
    log_model(
        "Failure_Classification", "LightGBM", "ml/failure_classification/metrics_lgbm.json", "AI4I"
    )
    print("MLflow logging completed.")
except Exception as e:
    print(f"Error logging to MLflow: {e}")
