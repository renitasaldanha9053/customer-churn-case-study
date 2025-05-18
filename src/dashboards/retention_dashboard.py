# retention_dashboard.py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import json
# Set style
sns.set(style="whitegrid")

# Paths
PROCESSED_DIR = os.path.abspath(os.path.join("..","..", "data", "processed"))
ENGINEERED_PATH = os.path.join(PROCESSED_DIR, "engineered_customer_churn.csv")
PREDICTIONS_DIR = os.path.join(PROCESSED_DIR, "predictions")
METRICS_DIR = os.path.abspath(os.path.join("..", "..", "data", "processed","metrics"))
SUMMARY_PATH = os.path.join(METRICS_DIR, "model_summary.json")
RAW_PATH = os.path.abspath(os.path.join("..","..", "data", "raw","customer_churn.csv"))

# Load Data
df = pd.read_csv(ENGINEERED_PATH)
with open(SUMMARY_PATH, "r") as f:
    summary = json.load(f)
best_model = summary["best_model"]
df_pred = pd.read_csv(os.path.join(PREDICTIONS_DIR, f"predicted_{best_model}.csv"))

# Merge predicted probabilities with original data
df = df.copy()
df["predicted_prob"] = df_pred["predicted_prob"]
df["predicted"] = df_pred["predicted"]
df.head()

st.title("Customer Retention Strategies Dashboard")
st.markdown("""
### Understanding how can Customer Churn be reduced
""")

# ----------- Churn Rate by Tenure Group ------------
TENURE_BINS = [0, 12, 24, 48, 60, 72]
TENURE_LABELS = ["<1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"]
df["TenureGroup"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, right=True)

st.subheader("Churn Rate by Customer Tenure Group")
fig, ax = plt.subplots()
sns.barplot(data=df, x="TenureGroup", y="Churn", estimator=lambda x: sum(x)/len(x), ax=ax)
ax.set_ylabel("Churn Rate")
st.pyplot(fig)

# ----------- Churn by Contract Type ------------
df_churn = pd.read_csv(RAW_PATH)

# Clean Churn column
df_churn["Churn"] = df_churn["Churn"].map({"Yes": 1, "No": 0})
st.subheader("Churn by Contract Type")
fig, ax = plt.subplots()
sns.barplot(data=df_churn, x="Contract", y="Churn", estimator=lambda x: sum(x)/len(x), ax=ax)
ax.set_ylabel("Churn Rate")
st.pyplot(fig)

# ----------- Monthly Charges Distribution ------------
st.subheader("Monthly Charges: Loyal vs Churned")
fig, ax = plt.subplots()
sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", ax=ax, fill=True)
ax.set_title("Distribution by Churn")
st.pyplot(fig)

# ----------- Service Feature: Internet Type ------------
st.subheader("Churn by Internet Service Type")
fig, ax = plt.subplots()
sns.barplot(data=df_churn, x="InternetService", y="Churn", estimator=lambda x: sum(x)/len(x), ax=ax)
ax.set_ylabel("Churn Rate")
st.pyplot(fig)

# ----------- Paperless Billing ------------
st.subheader("Impact of Paperless Billing on Churn")
fig, ax = plt.subplots()
sns.barplot(data=df, x="PaperlessBilling", y="Churn", estimator=lambda x: sum(x)/len(x), ax=ax)
ax.set_ylabel("Churn Rate")
st.pyplot(fig)

# ----------- Demographics: Senior Citizen ------------
st.subheader("Churn Rate by Senior Citizenship")
fig, ax = plt.subplots()
sns.barplot(data=df, x="SeniorCitizen", y="Churn", estimator=lambda x: sum(x)/len(x), ax=ax)
ax.set_xticklabels(["Not Senior", "Senior"])
ax.set_ylabel("Churn Rate")
st.pyplot(fig)

# ----------- Predicted Probability of Churn ------------
st.subheader("Predicted Probability of Churn")
fig, ax = plt.subplots()
sns.histplot(df["predicted_prob"], kde=True, bins=30, ax=ax)
ax.set_title("Predicted Churn Probability Distribution")
ax.set_xlabel("Probability")
st.pyplot(fig)

# ----------- Conclusion ------------
st.markdown("""
### Conclusion:
### 1) Short-tenure, month-to-month customers with high monthly charges churn more.
### 2) Fiber optic internet and **paperless billing** have higher churn risk.
### 3) Senior citizens and customers without partners or dependents are more likely to churn.
### 4)  Retention strategies may include:
###   1. Loyalty discounts or long-term contract offers
###   2. Targeted outreach to high-risk demographics
###   3. Incentives for switching to lower-churn service types
###   4. Personalized engagement based on churn probability
""")
