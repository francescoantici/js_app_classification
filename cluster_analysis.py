#!/usr/bin/env python3
"""
Cluster Analysis Script

Analyzes clustering results from CSV or Parquet files and generates statistics
and a summary of top applications per cluster.
"""

import argparse
import pandas as pd
import numpy as np
from collections import Counter


def load_data(input_file: str) -> pd.DataFrame:
    """Load data from CSV or Parquet file."""
    if input_file.endswith('.csv'):
        return pd.read_csv(input_file)
    elif input_file.endswith('.parquet'):
        return pd.read_parquet(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file}. Use .csv or .parquet")


def analyze_clusters(df: pd.DataFrame, cluster_col: str, app_col:str) -> None:
    """Analyze clustering results and print statistics."""

    if cluster_col not in df.columns:
        raise ValueError(f"Column '{cluster_col}' not found in file. Available columns: {list(df.columns)}")

    # Check for noise points (cluster = -1)
    cluster_series = df[cluster_col]

    # Get unique clusters (-1 is noise)
    unique_clusters = cluster_series.unique()
    n_clusters = len([c for c in unique_clusters if c >= 0])
    n_noise = int((cluster_series == -1).sum())

    print("=" * 60)
    print("CLUSTER ANALYSIS REPORT")
    print("=" * 60)
    print(f"\nTotal rows: {len(df)}")
    print(f"Number of clusters (excluding noise): {n_clusters}")
    print(f"Number of noise points (cluster=-1): {n_noise} ({n_noise / len(df) * 100:.1f}%)")

    # Cluster size distribution
    print("\n" + "-" * 40)
    print("CLUSTER SIZE DISTRIBUTION")
    print("-" * 40)

    cluster_counts = cluster_series[cluster_series >= 0].value_counts().sort_values(ascending=False)

    if len(cluster_counts) > 0:
        print(f"Biggest cluster: {cluster_counts.index[0]} (size: {cluster_counts.iloc[0]})")
        print(f"Smallest cluster: {cluster_counts.index[-1]} (size: {cluster_counts.iloc[-1]})")
        print(f"Average cluster size: {cluster_counts.mean():.2f}")
        print(f"Median cluster size: {cluster_counts.median():.0f}")

        # Top 10 biggest clusters
        print("\nTop 10 biggest clusters:")
        for i, (cluster_id, count) in enumerate(cluster_counts.head(10).items(), 1):
            print(f"  {i}. Cluster {cluster_id}: {count} samples ({count / len(df) * 100:.1f}%)")

        # Histogram of cluster sizes
        print("\nCluster size histogram:")
        bins = [0, 5, 10, 20, 50, 100, 200, 500, 1000, float('inf')]
        bin_labels = ['0-5', '6-10', '11-20', '21-50', '51-100', '101-200', '201-500', '501-1000', '>1000']
        size_hist = pd.cut(cluster_counts, bins=bins, labels=bin_labels)
        hist_counts = size_hist.value_counts().reindex(bin_labels, fill_value=0)

        for bin_label, count in hist_counts.items():
            bar = '#' * min(count, 40)
            print(f"  {bin_label:12s}: {count:5d} clusters {bar}")
    else:
        print("No clusters found (only noise points or empty data)")

    # Applications per cluster
    if app_col in df.columns:
        print("\n" + "-" * 40)
        print("TOP 10 APPLICATIONS PER CLUSTER")
        print("-" * 40)

        # Prepare results
        results = []

        for cluster_id in sorted(unique_clusters):
            if cluster_id == -1:
                continue

            cluster_df = df[df[cluster_col] == cluster_id]
            app_counts = Counter(cluster_df[app_col])

            top_10 = app_counts.most_common(10)
            total_in_cluster = len(cluster_df)

            print(f"\nCluster {cluster_id} (size: {total_in_cluster}):")

            for i, (app, count) in enumerate(top_10):
                pct = count / total_in_cluster * 100
                print(f"  {i+1}. {app}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze clustering results from CSV or Parquet files"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV or Parquet file path"
    )

    parser.add_argument(
        "--cluster_col",
        default="cluster",
        help="Column name containing cluster IDs (default: cluster)"
    )

    parser.add_argument(
            "--app_col",
            default="application",
            help="Column name containing application name (default: application)"
        )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from '{args.input}'...")
    df = load_data(args.input)
    print(f"Loaded {len(df)} rows")

    # Analyze
    analyze_clusters(df, args.cluster_col, args.app_col)


if __name__ == "__main__":
    main()
