"""
DBSCAN Clustering on app_label field from results/labels.csv

Pipeline:
  1. Load   — read app_label column from CSV
  2. Strip  — clean / normalise label strings
  3. Embed  — encode with embedding-emma (via sentence-transformers)
  4. Cluster— DBSCAN on the dense embedding matrix
  5. Output — clustered CSV
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

def strip_label(text: str) -> str:
    """
    Clean a raw app_label string:
    • strip leading/trailing whitespace
    • collapse internal whitespace runs to a single space
    • remove non-printable / control characters
    • lower-case for consistent embedding
    """
    text = str(text)
    text = text.strip()
    text = re.sub(r"[^\x20-\x7E]", " ", text)   # keep printable ASCII only
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    text = text.lower()
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DBSCAN clustering on app labels")

    # Data config
    parser.add_argument("--input_csv", default="results/labels.csv",
                        help="Input CSV file path (default: results/labels.csv)")
    parser.add_argument("--output_csv", default="results/labels_clustered.csv",
                        help="Output CSV file path (default: results/labels_clustered.csv)")
    parser.add_argument("--label_column", default="app_label",
                        help="Column name containing labels (default: app_label)")

    # Clustering config
    parser.add_argument("--min_cluster_size", type=int, default=15,
                        help="Minimum cluster size for DBSCAN (default: 15)")
    parser.add_argument("--eps", type=float, default=0.01,
                        help="EPS parameter for DBSCAN (default: 0.01)")
    parser.add_argument("--metric", default="cosine",
                        help="Distance metric for DBSCAN (default: cosine)")

    # Embedding config
    parser.add_argument("--embedding_model", default="google/embeddinggemma-300m",
                        help="Sentence transformer model for embeddings (default: google/embeddinggemma-300m)")
    parser.add_argument("--truncate_dim", type=int, default=256,
                        help="Dimension to truncate embeddings (default: 256)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for embedding (default: 64)")
    parser.add_argument("--device", default=None,
                        help="Device for embedding (default: auto-detect)")

    args = parser.parse_args()

    # Data config
    INPUT_CSV    = args.input_csv
    OUTPUT_CSV   = args.output_csv
    LABEL_COLUMN = args.label_column

    # Clustering config
    MIN_CLUSTER_SIZE = args.min_cluster_size
    EPS = args.eps
    METRIC = args.metric

    # Embedding config
    EMBEDDING_MODEL=args.embedding_model
    TRUNCATE_DIM = args.truncate_dim
    BATCH_SIZE  = args.batch_size
    DEVICE      = args.device

    # LOAD DATA
    print(f"Loading data from '{INPUT_CSV}' ...")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Could not find '{INPUT_CSV}'. "
            "Make sure the file exists relative to where you run this script."
        )

    df = pd.read_csv(INPUT_CSV)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{LABEL_COLUMN}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    original_len = len(df)
    df = df.dropna(subset=[LABEL_COLUMN]).reset_index(drop=True)
    print(f"  Rows loaded              : {original_len}")
    print(f"  Rows after dropping nulls: {len(df)}")

    # Strip labels
    df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(strip_label)

    # Drop rows that became empty after stripping
    before = len(df)
    df = df[df[LABEL_COLUMN].str.len() > 0].reset_index(drop=True)
    print(f"  Rows removed (empty after strip): {before - len(df)}")
    print(f"  Labels ready for embedding      : {len(df)}")

    # Extract app names
    app_names = df[LABEL_COLUMN].unique().to_numpy()
    print(f"Found {len(app_names)} unique app names")

    # EMBED WITH embedding-emma
    print(f"\nLoading embedding model '{EMBEDDING_MODEL}' ...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE, truncate_dim=TRUNCATE_DIM)

    print(f"Encoding {len(app_names)} labels (batch_size={BATCH_SIZE}) ...")
    embeddings = model.encode(
        app_names,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalise → cosine distance = euclidean / 2
        convert_to_numpy=True,
    )
    print(f"  Embedding matrix shape: {embeddings.shape}")

    # Save label to embedding mapping
    emb_map = {app_names[i]:embeddings[i] for i in range(len(app_names))}

    # DBSCAN CLUSTERING
    print(f"\nRunning DBSCAN (eps={EPS}, min_samples={MIN_CLUSTER_SIZE}) ...")
    # Save weights to make the operations less memory intensive
    sample_weight = [len(df[df[LABEL_COLUMN] == app]) for app in app_names]

    # Cluster
    db = DBSCAN(eps=EPS, min_samples=MIN_CLUSTER_SIZE, metric="cosine", n_jobs=-1)
    cluster_labels = db.fit_predict(embeddings, sample_weight=sample_weight)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise    = int((cluster_labels == -1).sum())

    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise points   : {n_noise}  ({n_noise / len(df) * 100:.1f}%)")

    label_map = {app_names[i]:cluster_labels[i] for i in range(len(app_names))}

    # SAVE RESULTS
    df["cluster"] = df[LABEL_COLUMN].apply(label_map.get)
    df.to_csv(OUTPUT_CSV, index=False)
    