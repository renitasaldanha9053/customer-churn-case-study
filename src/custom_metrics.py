# custom_metrics.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import pandas as pd

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def save_metrics(metrics_dict, output_path):
    if output_path.endswith(".json"):
        with open(output_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
    elif output_path.endswith(".csv"):
        pd.DataFrame([metrics_dict]).to_csv(output_path, index=False)