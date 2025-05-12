# File: multi_omics_integration.py

import argparse
import logging
import os
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)


def load_data(file_paths: List[str]) -> pd.DataFrame:
    """
    Load and concatenate multiple omics CSV files.

    Args:
        file_paths: List of paths to CSV files. Rows should represent samples.
    Returns:
        A single DataFrame with samples as rows and features from all omics concatenated.
    """
    dfs = []
    for path in file_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path, index_col=0)
        dfs.append(df)
        logger.info(f"Loaded {df.shape[0]} samples and {df.shape[1]} features from {path}")

    # Ensure consistent sample ordering
    common_index = set(dfs[0].index)
    for df in dfs[1:]:
        common_index &= set(df.index)
    if not common_index:
        raise ValueError("No common samples (row indices) found across all files.")
    common_index = sorted(common_index)

    aligned = [df.loc[common_index] for df in dfs]
    combined = pd.concat(aligned, axis=1)
    logger.info(f"Combined data shape: {combined.shape}")
    return combined


def preprocess_data(
    df: pd.DataFrame,
    variance_threshold: float = 0.0
) -> np.ndarray:
    """
    Impute missing values, remove low-variance features, and scale data.

    Args:
        df: Combined omics DataFrame.
        variance_threshold: Features with variance <= threshold will be removed.
    Returns:
        A NumPy array of preprocessed data.
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('var_thresh', VarianceThreshold(threshold=variance_threshold)),
        ('scaler', StandardScaler()),
    ])
    processed = pipeline.fit_transform(df.values)
    logger.info(
        f"After preprocessing: {processed.shape[1]} features (variance > {variance_threshold})"
    )
    return processed


def integrate_data(
    data: np.ndarray,
    n_components: int = 50,
    random_state: int = 42
) -> np.ndarray:
    """
    Perform PCA dimensionality reduction.

    Args:
        data: Preprocessed data array.
        n_components: Number of PCA components or <=1 for variance ratio cutoff.
        random_state: Random seed for reproducibility.
    Returns:
        PCA-transformed data array.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    embedding = pca.fit_transform(data)

    # Log variance explained
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    logger.info(
        f"PCA explained variance by component: {pca.explained_variance_ratio_}"  
    )
    logger.info(
        f"Cumulative explained variance: {cum_var}"  
    )
    return embedding


def cluster_data(
    embedding: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42
) -> np.ndarray:
    """
    Cluster the PCA embedding with KMeans and compute silhouette score.

    Args:
        embedding: PCA-transformed data.
        n_clusters: Number of clusters for KMeans.
        random_state: Random seed.
    Returns:
        Cluster labels array.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embedding)

    if embedding.shape[0] >= n_clusters * 2:
        score = silhouette_score(embedding, labels)
        logger.info(f"Silhouette score for {n_clusters} clusters: {score:.3f}")
    else:
        logger.warning(
            "Not enough samples to compute silhouette score (need at least 2 samples per cluster)."
        )
    return labels


def visualize_clusters(
    embedding: np.ndarray,
    labels: np.ndarray,
    out_prefix: str = 'multi_omics'
) -> None:
    """
    Create an interactive 2D scatter plot of the first two PCA components colored by cluster.

    Args:
        embedding: PCA-transformed data.
        labels: Cluster labels.
        out_prefix: Prefix for output files.
    """
    df_vis = pd.DataFrame({
        'PC1': embedding[:, 0],
        'PC2': embedding[:, 1],
        'Cluster': labels.astype(str)
    })
    fig = px.scatter(
        df_vis, x='PC1', y='PC2', color='Cluster',
        title='Multi-Omics Clustering',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
    )
    html_path = f"{out_prefix}_clusters.html"
    fig.write_html(html_path)
    logger.info(f"Interactive cluster plot saved to {html_path}")

    # Optionally save static image
    try:
        png_path = f"{out_prefix}_clusters.png"
        fig.write_image(png_path)
        logger.info(f"Static cluster plot saved to {png_path}")
    except Exception:
        logger.warning(
            "Failed to save static image; ensure Orca or Kaleido is installed."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Integrate multi-omics data via PCA + KMeans clustering"
    )
    parser.add_argument(
        'files', nargs='+', help='Paths to omics CSV files (samples as rows, features as columns)'
    )
    parser.add_argument(
        '--components', type=int, default=50,
        help='Number of PCA components (default: 50)'
    )
    parser.add_argument(
        '--clusters', type=int, default=3,
        help='Number of clusters for KMeans (default: 3)'
    )
    parser.add_argument(
        '--variance_thresh', type=float, default=0.0,
        help='Variance threshold for feature filtering (default: 0.0)'
    )
    parser.add_argument(
        '--out_prefix', default='multi_omics',
        help='Prefix for output files (default: multi_omics)'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        df = load_data(args.files)
        data = preprocess_data(df, variance_threshold=args.variance_thresh)
        embedding = integrate_data(data, n_components=args.components)
        labels = cluster_data(embedding, n_clusters=args.clusters)
        np.save(f"{args.out_prefix}_embedding.npy", embedding)
        np.save(f"{args.out_prefix}_labels.npy", labels)
        visualize_clusters(embedding, labels, out_prefix=args.out_prefix)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        exit(1)


if __name__ == '__main__':
    main()
