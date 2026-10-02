# -*- coding: utf-8 -*-
"""
Rising Waters: AI-Powered Flood Prediction System
Pipeline Stage 03: Data Cleaning, Outlier Capping & Scaling

This module processes raw historical data: handles missing values, clips outliers
using the Interquartile Range (IQR) method, subsets features to match the
production web application input, fits a StandardScaler, and splits the data.
"""

import logging
import os
from typing import Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("DataPreprocessor")

# Core prediction features collected from the user interface
FEATURE_COLS = [
    'ANNUAL_RAINFALL', 
    'CLOUD_COVERAGE', 
    'JUN-SEP', 
    'MAR-MAY', 
    'OCT-DEC', 
    'JAN-FEB'
]

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "flood_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads raw records from the dataset file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source dataset not found at: {file_path}")
    df = pd.read_csv(file_path)
    logger.info("Raw dataset loaded. Initial Shape: %s", df.shape)
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in numerical columns with their median,
    and categorical columns with their mode.
    """
    logger.info("Executing missing value audit and imputation...")
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Impute numeric columns with median
    for col in numeric_cols:
        null_sum = df[col].isnull().sum()
        if null_sum > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info("Imputed missing values for '%s' (%d nulls) with median: %.2f", col, null_sum, median_val)

    # Impute categorical columns with mode
    for col in categorical_cols:
        null_sum = df[col].isnull().sum()
        if null_sum > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info("Imputed missing values for '%s' (%d nulls) with mode: %s", col, null_sum, mode_val)
            
    logger.info("Imputation pipeline complete.")
    return df


def cap_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Caps extreme values outside of the IQR boundary range:
    [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
    """
    logger.info("Executing IQR outlier capping on features...")
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        low_count = (df[col] < lower_bound).sum()
        high_count = (df[col] > upper_bound).sum()
        
        # Apply clipping to boundary values
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        if low_count + high_count > 0:
            logger.info(
                "Feature '%s': Capped %d low and %d high outlier values to [%.2f, %.2f]",
                col, low_count, high_count, lower_bound, upper_bound
            )
            
    logger.info("Outlier capping complete.")
    return df


def split_and_scale(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    target_col: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Subsets the data, fits a StandardScaler on the features, and splits into train/test sets.
    """
    logger.info("Extracting core features and target variables...")
    X = df[feature_cols]
    y = df[target_col]
    
    logger.info("Fitting feature standardization pipeline...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    logger.info("Partitioning data (train_size=%.1f, test_size=%.1f)...", 1.0 - test_size, test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    logger.info("Training features shape: %s | Test features shape: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test, scaler


def save_scaler(scaler: StandardScaler, save_path: str) -> None:
    """Exports the fitted scaler to the models directory."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(scaler, save_path)
    logger.info("StandardScaler pipeline successfully saved to: %s", save_path)


def main() -> None:
    """Core process controller."""
    try:
        df = load_dataset(DATASET_PATH)
        df_imputed = impute_missing_values(df)
        df_clean = cap_outliers_iqr(df_imputed, FEATURE_COLS)
        
        X_train, X_test, y_train, y_test, scaler = split_and_scale(
            df_clean, FEATURE_COLS, 'FLOODS'
        )
        
        scaler_save_path = os.path.join(MODELS_DIR, "scaler.pkl")
        save_scaler(scaler, scaler_save_path)
        
        logger.info("Data preprocessing stage completed successfully.")
        
    except Exception as err:
        logger.error("Preprocessing pipeline execution failed: %s", err)


if __name__ == "__main__":
    main()
