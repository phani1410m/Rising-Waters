# -*- coding: utf-8 -*-
"""
Rising Waters: AI-Powered Flood Prediction System
Pipeline Stage 02: Exploratory Data Analysis & Visualizations

This module conducts univariate, multivariate, and correlation analysis of the meteorological
parameters. All resulting figures are saved to the project documentation directory.
"""

import logging
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("DataVisualizer")

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "flood_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "plots")

# Set consistent plotting styles
sns.set_style('darkgrid')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']


def load_data(file_path: str) -> pd.DataFrame:
    """Loads raw records for visualization."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source dataset not found at: {file_path}")
    return pd.read_csv(file_path)


def generate_univariate_distributions(df: pd.DataFrame, save_dir: str) -> None:
    """Generates and saves histograms representing key meteorological factors."""
    logger.info("Plotting univariate distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution of Key Meteorological Features', fontsize=16, fontweight='bold')

    # Subplot 1: Annual Rainfall
    sns.histplot(df['ANNUAL_RAINFALL'], kde=True, bins=30, color='steelblue', ax=axes[0, 0])
    axes[0, 0].set_title('Annual Rainfall Distribution', fontweight='semibold')
    axes[0, 0].set_xlabel('Annual Rainfall (mm)')

    # Subplot 2: Cloud Coverage
    sns.histplot(df['CLOUD_COVERAGE'], kde=True, bins=30, color='coral', ax=axes[0, 1])
    axes[0, 1].set_title('Cloud Coverage Distribution', fontweight='semibold')
    axes[0, 1].set_xlabel('Cloud Coverage (%)')

    # Subplot 3: Monsoon Rainfall (JUN-SEP)
    sns.histplot(df['JUN-SEP'], kde=True, bins=30, color='seagreen', ax=axes[1, 0])
    axes[1, 0].set_title('Monsoon Rainfall (JUN-SEP) Distribution', fontweight='semibold')
    axes[1, 0].set_xlabel('JUN-SEP Rainfall (mm)')

    # Subplot 4: Pre-Monsoon Rainfall (MAR-MAY)
    sns.histplot(df['MAR-MAY'], kde=True, bins=30, color='goldenrod', ax=axes[1, 1])
    axes[1, 1].set_title('Pre-Monsoon Rainfall (MAR-MAY) Distribution', fontweight='semibold')
    axes[1, 1].set_xlabel('MAR-MAY Rainfall (mm)')

    plt.tight_layout()
    output_path = os.path.join(save_dir, 'univariate_distributions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", output_path)


def generate_multivariate_pairplot(df: pd.DataFrame, save_dir: str) -> None:
    """Creates a pairplot matrix colored by binary flood occurrences."""
    logger.info("Plotting pairwise feature relationships...")
    key_features = ['ANNUAL_RAINFALL', 'JUN-SEP', 'MAR-MAY', 'CLOUD_COVERAGE', 'FLOODS']
    
    # Filter dataset columns to key features
    pair_df = df[key_features].copy()
    pair_df['FLOODS'] = pair_df['FLOODS'].astype(str)

    g = sns.pairplot(
        pair_df, 
        hue='FLOODS', 
        palette={'0': '#3498db', '1': '#e74c3c'}, 
        diag_kind='kde',
        plot_kws={'alpha': 0.5, 's': 20, 'edgecolor': 'none'}
    )
    g.figure.suptitle('Pairwise Distribution Matrix Grouped by Flood Status', y=1.02, fontsize=16, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(save_dir, 'pairplot.png')
    g.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", output_path)


def generate_rainfall_cloud_scatter(df: pd.DataFrame, save_dir: str) -> None:
    """Generates a scatter plot mapping annual rainfall vs cloud cover."""
    logger.info("Plotting Annual Rainfall vs Cloud Coverage...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    scatter = ax.scatter(
        df['ANNUAL_RAINFALL'], 
        df['CLOUD_COVERAGE'],
        c=df['FLOODS'], 
        cmap='coolwarm', 
        alpha=0.6, 
        edgecolors='k', 
        linewidths=0.3, 
        s=35
    )
    
    ax.set_xlabel('Annual Rainfall (mm)', fontsize=12)
    ax.set_ylabel('Cloud Coverage (%)', fontsize=12)
    ax.set_title('Correlation Scatter: Annual Rainfall vs Cloud Coverage', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Flood Occurrence Status (1 = Flood, 0 = Safe)')

    plt.tight_layout()
    output_path = os.path.join(save_dir, 'scatter_rainfall_cloud.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", output_path)


def generate_feature_boxplots(df: pd.DataFrame, save_dir: str) -> None:
    """Generates boxplots showing the distributions of rainfall features by target classes."""
    logger.info("Plotting target-grouped boxplots...")
    rainfall_features = ['ANNUAL_RAINFALL', 'JUN-SEP', 'MAR-MAY', 'OCT-DEC', 'JAN-FEB']

    df_box = df.copy()
    df_box['FLOODS'] = pd.to_numeric(df_box['FLOODS'], errors='coerce').fillna(0).astype(int)

    fig, axes = plt.subplots(1, len(rainfall_features), figsize=(22, 6))
    fig.suptitle('Precipitation Feature Distributions Grouped by Flood Status', fontsize=16, fontweight='bold')

    for i, feat in enumerate(rainfall_features):
        sns.boxplot(
            x='FLOODS', 
            y=feat, 
            data=df_box, 
            ax=axes[i],
            palette={0: '#3498db', 1: '#e74c3c', '0': '#3498db', '1': '#e74c3c'},
            width=0.6
        )
        axes[i].set_title(feat, fontweight='semibold')
        axes[i].set_xlabel('Flood Status')

    plt.tight_layout()
    output_path = os.path.join(save_dir, 'boxplots_by_floods.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", output_path)


def generate_correlation_heatmap(df: pd.DataFrame, save_dir: str) -> None:
    """Plots and exports the correlation heatmap of numeric parameters."""
    logger.info("Plotting correlation matrix heatmap...")
    numeric_df = df.select_dtypes(include=[np.number])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = numeric_df.corr()
    
    # Generate upper triangle mask to prevent symmetry clutter
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True, 
        fmt='.2f', 
        cmap='RdBu_r',
        center=0, 
        linewidths=0.5, 
        ax=ax, 
        annot_kws={'size': 9}
    )
    ax.set_title('Meteorological Features Correlation Heatmap', fontsize=16, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(save_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", output_path)


def generate_class_distribution_plot(df: pd.DataFrame, save_dir: str) -> None:
    """Generates a bar plot showing the distribution of safe vs flood target classes."""
    logger.info("Plotting class distribution bar chart...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Cast to integer for display formatting
    df_plot = df.copy()
    df_plot['FLOODS'] = pd.to_numeric(df_plot['FLOODS'], errors='coerce').fillna(0).astype(int)
    
    # Count classes
    class_counts = df_plot['FLOODS'].value_counts()
    
    bars = ax.bar(
        ['Safe (Class 0)', 'Flood Risk (Class 1)'], 
        [class_counts.get(0, 0), class_counts.get(1, 0)], 
        color=['#3498db', '#e74c3c'],
        edgecolor='black',
        width=0.5
    )
    
    # Add count labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            height + 10,
            f'{height} ({height/len(df_plot):.1%})',
            ha='center', 
            va='bottom', 
            fontweight='bold'
        )
        
    ax.set_ylim(0, max(class_counts) * 1.15)
    ax.set_ylabel('Record Count', fontsize=12)
    ax.set_title('Target Variable Distribution (FLOODS)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, 'class_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", output_path)


def main() -> None:
    """Core process controller."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df = load_data(DATASET_PATH)
        
        generate_univariate_distributions(df, OUTPUT_DIR)
        generate_multivariate_pairplot(df, OUTPUT_DIR)
        generate_rainfall_cloud_scatter(df, OUTPUT_DIR)
        generate_feature_boxplots(df, OUTPUT_DIR)
        generate_correlation_heatmap(df, OUTPUT_DIR)
        generate_class_distribution_plot(df, OUTPUT_DIR)
        
        logger.info("Exploratory Data Analysis completed. All charts saved.")
    except Exception as err:
        logger.error("Visualization pipeline execution failed: %s", err)


if __name__ == '__main__':
    main()
