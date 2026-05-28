import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)

RAW_PATH = "data/raw/BankChurners.csv"
PROCESSED_PATH = "data/processed/clean_customers.csv"

os.makedirs("data/processed", exist_ok=True)


def load_data(path):
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw data shape: {df.shape}")
    return df


def drop_junk_columns(df):
    junk = [
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]
    df = df.drop(columns=[c for c in junk if c in df.columns])
    logger.info("Dropped junk classifier columns")
    return df


def handle_nulls(df):
    null_counts = df.isnull().sum()
    logger.info(f"Null values before cleaning:\n{null_counts[null_counts > 0]}")

    # Fill categorical nulls with mode
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Fill numeric nulls with median
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    logger.info("Null values handled")
    return df


def feature_engineering(df):
    # 1. Utilization Ratio — how much credit is being used
    df['Utilization_Ratio'] = (
        df['Total_Revolving_Bal'] / df['Credit_Limit']
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # 2. Avg transaction value
    df['Avg_Transaction_Value'] = (
        df['Total_Trans_Amt'] / df['Total_Trans_Ct']
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # 3. Months inactive ratio
    df['Inactivity_Ratio'] = df['Months_Inactive_12_mon'] / 12

    # 4. Encode target column as binary
    df['Churn'] = (
        df['Attrition_Flag'] == 'Attrited Customer'
    ).astype(int)

    logger.info("Feature engineering complete — new columns added: Utilization_Ratio, Avg_Transaction_Value, Inactivity_Ratio, Churn")
    return df


def save_data(df, path):
    df.to_csv(path, index=False)
    logger.info(f"Cleaned data saved to {path}")
    logger.info(f"Final shape: {df.shape}")


def run_pipeline():
    logger.info("========== PIPELINE STARTED ==========")
    start = datetime.now()

    df = load_data(RAW_PATH)
    df = drop_junk_columns(df)
    df = handle_nulls(df)
    df = feature_engineering(df)
    save_data(df, PROCESSED_PATH)

    duration = (datetime.now() - start).seconds
    logger.info(f"========== PIPELINE COMPLETE in {duration}s ==========")


if __name__ == "__main__":
    run_pipeline()
