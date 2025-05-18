#feature_engineering.py
import pandas as pd

def engineer_features(df):
    print("Engineering features")

    # Binary Encoding
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Convert target
    if df['Churn'].dtype == object:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # One-Hot Encoding
    ohe_cols = ['InternetService', 'Contract', 'PaymentMethod', 'Gender']
    for col in ohe_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)


    # Binning tenure
    df['tenure_group'] = pd.cut(df['tenure'], 
                                bins=[0, 12, 24, 48, 60, df['tenure'].max()],
                                labels=['0-12m', '12-24m', '24-48m', '48-60m', '60m+'])

    # Encode tenure_group
    df = pd.get_dummies(df, columns=['tenure_group'])

    # TotalCharges to numeric (if not already)
    if df['TotalCharges'].dtype == object:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # New Feature: Average Monthly Spend
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)

    return df
