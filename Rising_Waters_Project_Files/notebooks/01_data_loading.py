# -*- coding: utf-8 -*-
"""
Rising Waters: AI-Powered Flood Prediction System
Pipeline Stage 01: Data Ingestion & Structural Inspection

This script loads the historical meteorological dataset, inspects its dimensions,
checks for missing values, and prints descriptive statistical summaries.
"""

import logging
import os
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("DataIngestion")

# Relative paths for dataset loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "flood_data.csv")


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads raw CSV meteorological records from disk.

    Args:
        file_path: Absolute or relative path to the csv dataset.

    Returns:
        Pandas DataFrame containing raw records.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source dataset not found at: {file_path}")
        
    logger.info("Loading raw meteorological dataset from: %s", file_path)
    df = pd.read_csv(file_path)
    logger.info("Ingestion completed. Total Shape: %s", df.shape)
    return df


def inspect_metadata(df: pd.DataFrame) -> None:
    """
    Prints high-level metadata statistics including data types and column structures.
    """
    print("\n" + "=" * 80)
    print("                      METADATA & STRUCTURAL INTEGRITY")
    print("=" * 80)
    print(f"Total Records (Rows)   : {df.shape[0]}")
    print(f"Total Columns (Features): {df.shape[1]}")
    
    print("\n--- Column Definitions & Types ---")
    print(df.dtypes)
    print("=" * 80 + "\n")


def check_missing_data(df: pd.DataFrame) -> pd.Series:
    """
    Scans columns for null values and outputs missing count summaries.

    Args:
        df: Input DataFrame.

    Returns:
        Pandas Series listing missing counts per column.
    """
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    print("=" * 80)
    print("                         MISSING DATA AUDIT")
    print("=" * 80)
    print(null_counts[null_counts > 0] if total_nulls > 0 else "No missing values detected.")
    print(f"\nAggregate Null Count: {total_nulls}")
    print("=" * 80 + "\n")
    return null_counts


def inspect_distributions(df: pd.DataFrame) -> None:
    """
    Calculates summary statistics and target variable balance.
    """
    print("=" * 80)
    print("                       TARGET VARIABLE BALANCE")
    print("=" * 80)
    if 'FLOODS' in df.columns:
        counts = df['FLOODS'].value_counts()
        mean_rate = df['FLOODS'].mean()
        print(f"Safe Instances (Class 0): {counts.get(0, 0)}")
        print(f"Flood Instances (Class 1): {counts.get(1, 0)}")
        print(f"Historical Flood Occurrence Rate: {mean_rate:.2%}")
    else:
        print("Target variable 'FLOODS' not present in the dataset.")
    print("=" * 80 + "\n")

    print("=" * 80)
    print("                         DESCRIPTIVE STATS")
    print("=" * 80)
    print(df.describe().T)
    print("=" * 80 + "\n")


def main() -> None:
    """Core process controller."""
    try:
        df = load_dataset(DATASET_PATH)
        
        # Display sample head and tail
        print("\n--- Dataset Sample Head (First 5 Rows) ---")
        print(df.head())
        
        inspect_metadata(df)
        check_missing_data(df)
        inspect_distributions(df)
        
    except Exception as err:
        logger.error("Data loading inspection pipeline failed: %s", err)


if __name__ == "__main__":
    main()
