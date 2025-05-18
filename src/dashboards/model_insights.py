import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

# Set paths
PROCESSED_DIR = os.path.abspath(os.path.join("..", "data", "processed"))
BEFORE_BALANCE = os.path.abspath(os.path.join("..", "..", "data", "processed", "metrics","class_distribution_before.png"))
AFTER_BALANCE = os.path.abspath(os.path.join("..", "..", "data", "processed", "metrics","class_distribution_after.png"))
FEATURE_IMPORTANCE_DIR = os.path.abspath(os.path.join("..", "..", "data", "processed", "feature_importance"))
METRICS_DIR = os.path.abspath(os.path.join("..", "..", "data", "processed","metrics"))
PREDICTIONS_DIR = os.path.abspath(os.path.join("..", "..", "data", "processed","predictions"))
# Dashboard title
st.title("Dashboard 2: Model Insights & Class Imbalance Handling")

# Section 1: Class imbalance visualization
st.header("1. Class Imbalance Handling")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Before Balancing")
    st.image(BEFORE_BALANCE)

with col2:
    st.subheader("After Balancing")
    st.image(AFTER_BALANCE)

st.markdown("Techniques used: **SMOTE** for oversampling the minority class.")

# Section 2: Feature Importance
st.header("2. Feature Importance")
importance_model = st.selectbox("Select model to view importance", ["logreg","rf", "xgb"])
importance_path = os.path.join(FEATURE_IMPORTANCE_DIR, f"importance_{importance_model}.csv")

if os.path.exists(importance_path):
    imp_df = pd.read_csv(importance_path).sort_values(by="importance", ascending=False).head(10)
    st.bar_chart(data=imp_df.set_index("feature"))
else:
    st.warning("Feature importance file not found.")

# Section 3: Evaluation Metrics
st.header("3. Model Evaluation Metrics")
metrics = {}
for model in ["logreg", "rf", "xgb"]:
    path = os.path.join(METRICS_DIR, f"metrics_{model}.json")
    if os.path.exists(path):
        with open(path) as f:
            metrics[model] = json.load(f)

if metrics:
    metric_df = pd.DataFrame(metrics).T.round(4)
    st.dataframe(metric_df)
else:
    st.warning("No evaluation metrics found.")

# Section 4: ROC Curves
st.header("4. ROC Curves")
roc_fig, ax = plt.subplots()
for model in metrics:
    df = pd.read_csv(os.path.join(PREDICTIONS_DIR, f"predicted_{model}.csv"))
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(df["Churn"], df["predicted_prob"])
    ax.plot(fpr, tpr, label=f"{model.upper()} (AUC={metrics[model]['roc_auc']:.2f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend()
ax.grid(True)
st.pyplot(roc_fig)

# Section 5: Confusion Matrix
st.header("5. Confusion Matrix of Best Model")
summary_path = os.path.join(METRICS_DIR, "model_summary.json")
if os.path.exists(summary_path):
    with open(summary_path) as f:
        summary = json.load(f)
    best_model = summary["best_model"]
    df_best = pd.read_csv(os.path.join(PREDICTIONS_DIR, f"predicted_{best_model}.csv"))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df_best["Churn"], df_best["predicted"])
    cm_fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {best_model.upper()}")
    st.pyplot(cm_fig)
else:
    st.warning("Model summary not found.")

# Section 6: Hyperparameter Tuning Results
st.header("6. Hyperparameter Tuning")
params_path = os.path.join(METRICS_DIR, "best_params.json")
if os.path.exists(params_path):
    with open(params_path) as f:
        best_params = json.load(f)
    for model, params in best_params.items():
        st.subheader(f"{model.upper()} Best Parameters")
        st.json(params)
else:
    st.info("Hyperparameter tuning was not performed or results are unavailable.")

# Section 7: Final Insights
st.header("7. Final Insights")
st.markdown("""
1) SMOTE effectively addressed class imbalance.
2) Random Forest or XGBoost often outperform logistic regression in ROC-AUC.
3) ROC Curves show trade-offs in false positive/negative rates.
4) Feature importance provides explainability—top drivers of churn are clear.
5) Best model balances precision/recall with tuning improving generalization.
""")
