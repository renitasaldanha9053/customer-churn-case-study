# churn_overview.py
import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Set global styles
sns.set(style="whitegrid")
plt.rcParams.update({'figure.figsize': (10, 5)})

# ----------- Load Data -----------
DATA_PATH = os.path.abspath(os.path.join("..", "..", "data", "processed", "cleaned_customer_churn.csv"))

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.title("Customer Churn Overview Dashboard")
st.markdown("This dashboard explores how customer churn varies by usage patterns and demographics.")

# ----------- Chart 1: Churn Distribution -----------
st.subheader("1. Churn Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='Churn', data=df, palette='Set2', ax=ax1)
ax1.set_title("Churn Distribution (0=No, 1=Yes)")
st.pyplot(fig1)

# ----------- Chart 2: Churn by Contract Type -----------
st.subheader("2. Churn by Contract Type")
fig2, ax2 = plt.subplots()
sns.countplot(x='Contract', hue='Churn', data=df, palette='pastel', ax=ax2)
ax2.set_title("Churn by Contract Type")
ax2.set_xlabel("Contract Type")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# ----------- Chart 3: Monthly Charges by Churn -----------
st.subheader("3. Monthly Charges by Churn")
fig3, ax3 = plt.subplots()
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='muted', ax=ax3)
ax3.set_title("Monthly Charges by Churn")
st.pyplot(fig3)

# ----------- Chart 4: Tenure Distribution with Churn Overlay -----------
st.subheader("4. Tenure Distribution by Churn")
fig4, ax4 = plt.subplots()
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, multiple="stack", ax=ax4)
ax4.set_title("Customer Tenure Distribution by Churn")
ax4.set_xlabel("Tenure (Months)")
st.pyplot(fig4)

# ----------- Chart 5: Churn by Internet Service Type -----------
if 'InternetService' in df.columns:
    st.subheader("5. Churn by Internet Service Type")
    fig5, ax5 = plt.subplots()
    sns.countplot(x='InternetService', hue='Churn', data=df, palette='pastel', ax=ax5)
    ax5.set_title("Churn by Internet Service Type")
    ax5.set_xlabel("Internet Service")
    st.pyplot(fig5)

# ----------- Chart 6: Senior Citizen Churn Rate -----------
if 'SeniorCitizen' in df.columns:
    st.subheader("6. Senior Citizen Churn Rate")
    senior_churn = df.groupby('SeniorCitizen')['Churn'].mean()
    fig6, ax6 = plt.subplots()
    sns.barplot(x=['Not Senior (0)', 'Senior (1)'], y=senior_churn.values, palette='Set2', ax=ax6)
    ax6.set_ylabel("Churn Rate")
    ax6.set_ylim(0, 1)
    ax6.set_title("Churn Rate by Senior Citizen Status")
    st.pyplot(fig6)

# ----------- Chart 7: Dependents vs Churn -----------
if 'Dependents' in df.columns:
    st.subheader("7. Dependents vs Churn")
    dep_churn = pd.crosstab(df['Dependents'], df['Churn'], normalize='index')
    fig7, ax7 = plt.subplots()
    dep_churn.plot(kind='bar', stacked=True, colormap='coolwarm', ax=ax7)
    ax7.set_title("Churn Distribution by Dependents")
    ax7.set_ylabel("Proportion")
    ax7.legend(title="Churn", labels=["No", "Yes"])
    st.pyplot(fig7)

# ----------- Chart 8: Avg Monthly Spend by Churn (Violin) -----------
st.subheader("8. Average Monthly Charges by Churn")
fig8, ax8 = plt.subplots()
sns.violinplot(x='Churn', y='MonthlyCharges', data=df, palette='muted', inner='quart', ax=ax8)
ax8.set_title("Monthly Charges Distribution by Churn")
st.pyplot(fig8)

# ----------- Conclusion -----------
st.markdown("---")
st.subheader("Conclusion")
st.markdown("""
1) Customers on month-to-month contracts are more likely to churn.
2) Higher monthly charges correlate with churn.
3) Shorter tenure customers churn more.
4) Senior citizens and customers without dependents have slightly higher churn rates.
5) Internet service types also impact churn behavior.
""")
