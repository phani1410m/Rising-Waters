# -*- coding: utf-8 -*-
"""
Rising Waters: AI-Powered Flood Prediction System
Pipeline Stage 04: Model Training, Benchmarking & Serialization

This module runs the complete model training workflow: preprocesses input records,
trains K-Nearest Neighbors, Decision Tree, Random Forest, and XGBoost Classifiers,
compares metrics (accuracy, precision, recall, F1, confusion matrices), plots
evaluation charts (model comparisons and feature importances), and serializes the
best classifier and scaler for application server use.
"""

import logging
import os
from typing import Dict, Any, Tuple, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ModelTrainer")

# Feature set selection
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
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "plots")


def load_and_preprocess(file_path: str) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Loads raw meteorological dataset, handles missing values via median/mode,
    clips outliers with the IQR method, scales the selected 6 features,
    and returns features (X), labels (y), and the fitted StandardScaler.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source dataset not found at: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # IQR outlier capping on features
    for col in FEATURE_COLS:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # Subset and scale
    X = df[FEATURE_COLS]
    y = df['FLOODS']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    return X_scaled, y, scaler


def evaluate_model(
    model: Any, 
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series
) -> Tuple[float, np.ndarray, str]:
    """
    Fits a model and evaluates its performance metrics.

    Returns:
        A tuple of (accuracy, confusion_matrix, classification_report).
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    return acc, cm, cr


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str) -> None:
    """Saves a bar plot comparing validation accuracy across trained classifiers."""
    logger.info("Generating model comparison accuracy bar chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Highlight the best model with green and others with blue
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(comparison_df))]
    
    bars = ax.bar(
        comparison_df['Model'], 
        comparison_df['Accuracy'], 
        color=colors,
        edgecolor='black', 
        linewidth=0.8
    )

    # Add accuracy values on top of the bars
    for bar, acc in zip(bars, comparison_df['Accuracy']):
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            bar.get_height() + 0.01,
            f'{acc:.2%}', 
            ha='center', 
            va='bottom', 
            fontweight='bold', 
            fontsize=11
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Validation Accuracy', fontsize=13)
    ax.set_title('Classifier Accuracy Comparison — Flood Prediction', fontsize=15, fontweight='bold')
    ax.set_xlabel('Classification Algorithms', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", save_path)


def plot_feature_importance(best_model: Any, feature_names: List[str], save_path: str) -> None:
    """Generates and saves the feature importance bar chart of the champion model."""
    logger.info("Generating feature importance plot for champion model...")
    if not hasattr(best_model, "feature_importances_"):
        logger.warning("Selected best model does not export feature importances. Skipping plot.")
        return
        
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, ax=ax, palette='viridis')
    
    ax.set_title('Feature Importance Analysis (Champion Model)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Relative Importance Score', fontsize=13)
    ax.set_ylabel('Meteorological Parameters', fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", save_path)


def main() -> None:
    """Core process controller."""
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load and preprocess features
        X_scaled, y, scaler = load_and_preprocess(DATASET_PATH)
        
        # Train-Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        logger.info("Data splits complete. Train: %s, Test: %s", X_train.shape, X_test.shape)

        # Define classifiers to evaluate
        classifiers = {
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
        }

        results = {}
        for name, clf in classifiers.items():
            logger.info("Training and evaluating model: %s", name)
            acc, cm, cr = evaluate_model(clf, X_train, X_test, y_train, y_test)
            results[name] = {
                'accuracy': acc, 
                'confusion_matrix': cm, 
                'classification_report': cr, 
                'model_object': clf
            }
            
            print("\n" + "=" * 70)
            print(f" MODEL PERFORMANCE: {name}")
            print("=" * 70)
            print(f"Validation Accuracy: {acc:.4%}")
            print("\nConfusion Matrix:")
            print(cm)
            print("\nClassification Report:")
            print(cr)
            print("=" * 70 + "\n")

        # Compile comparison results
        comparison_data = {
            'Model': list(results.keys()),
            'Accuracy': [res['accuracy'] for res in results.values()]
        }
        comparison_df = pd.DataFrame(comparison_data).sort_values(
            'Accuracy', ascending=False
        ).reset_index(drop=True)
        
        print("+" * 50)
        print("             FINAL METRIC COMPARISON")
        print("+" * 50)
        print(comparison_df.to_string(index=False))
        print("+" * 50 + "\n")

        # Select the champion model
        best_name = comparison_df.iloc[0]['Model']
        best_acc = comparison_df.iloc[0]['Accuracy']
        best_model = results[best_name]['model_object']
        
        logger.info("CHAMPION MODEL SELECTED: %s with %.2f%% accuracy", best_name, best_acc * 100)

        # Generate plots
        model_comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
        plot_model_comparison(comparison_df, model_comparison_path)
        
        feature_importance_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
        plot_feature_importance(best_model, FEATURE_COLS, feature_importance_path)

        # Save artifacts
        model_save_path = os.path.join(MODELS_DIR, "floods.save")
        scaler_save_path = os.path.join(MODELS_DIR, "scaler.pkl")
        
        joblib.dump(best_model, model_save_path)
        joblib.dump(scaler, scaler_save_path)
        
        logger.info("Successfully serialized champion model to: %s", model_save_path)
        logger.info("Successfully serialized StandardScaler to: %s", scaler_save_path)

    except Exception as err:
        logger.error("Model training pipeline execution failed: %s", err)


if __name__ == "__main__":
    main()
