# File: multi_omics_integration.py

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
def load_data(file_paths):
    data = [pd.read_csv(file) for file in file_paths]
    return data

def preprocess_data(data):
    preprocessed_data = []
    for df in data:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        preprocessed_data.append(scaled_data)
    return preprocessed_data

# Integrate data using PCA
def integrate_data(data, n_components=50):
    pca = PCA(n_components=n_components)
    integrated_data = pca.fit_transform(np.hstack(data))
    return integrated_data

# Cluster integrated data
def cluster_data(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters

# Visualize clustering results
def visualize_clusters(data, clusters):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=clusters, palette='viridis')
    plt.title('Clusters of Integrated Multi-Omics Data')
    plt.show()

if __name__ == "__main__":
    file_paths = ['path/to/genomics.csv', 'path/to/transcriptomics.csv', 'path/to/proteomics.csv']
    data = load_data(file_paths)
    preprocessed_data = preprocess_data(data)
    integrated_data = integrate_data(preprocessed_data)

    clusters = cluster_data(integrated_data)
    visualize_clusters(integrated_data, clusters)
    
    # Save integrated data and clusters
    np.save('integrated_multi_omics_data.npy', integrated_data)
    np.save('multi_omics_clusters.npy', clusters)
