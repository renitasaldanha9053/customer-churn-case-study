#data_cleaning.py
import os
import pandas as pd

# ----------- Path Setup -----------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove rows with missing TotalCharges (which are blank strings)
    df['TotalCharges'] = df['TotalCharges'].replace(" ", pd.NA)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)

    # Drop customerID - not useful for modeling
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # Encode churn column as 0/1 if not already done
    if df['Churn'].dtype == object:
        df['Churn'] = df['Churn'].map({"Yes": 1, "No": 0})

    # Fill remaining blanks or NAs if any
    df.ffill(inplace=True)

    return df

