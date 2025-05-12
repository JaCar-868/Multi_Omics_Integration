## Multi-Omics Integration Project

This repository provides a configurable pipeline for integrating multi-omics data (e.g., genomics, transcriptomics, proteomics) using PCA for dimensionality reduction and KMeans for clustering, with interactive Plotly visualizations and robust preprocessing.

## Project Structure

multi_omics_integration.py: Main Python script with a command-line interface.

data/ (optional): Folder for organizing input CSV files, each with samples as rows and features as columns.

results/ (optional): Directory for storing generated embeddings, labels, and plots.

README.md: Project documentation.

## Requirements

Python 3.7+

numpy

pandas

scikit-learn

plotly

kaleido (for static image export)

## Install dependencies via:

pip install numpy pandas scikit-learn plotly kaleido

## Data Format

Each CSV file represents one omics layer and must share the same sample identifiers (row indices):

# rows = sample IDs, columns = features
sample_id,feature1,feature2,...
sampleA,0.5,1.2,...
sampleB,0.3,0.8,...

All files passed to the script will be aligned on their common samples before integration.

##Usage

python multi_omics_integration.py \
  path/to/genomics.csv \
  path/to/transcriptomics.csv \
  path/to/proteomics.csv \
  --components 50 \
  --clusters 3 \
  --variance_thresh 0.0 \
  --out_prefix results/multi_omics

## Key Arguments

files (positional): List of input CSV file paths.

--components: Number of PCA components (default: 50).

--clusters: Number of clusters for KMeans (default: 3).

--variance_thresh: Variance threshold for filtering low-variance features (default: 0.0).

--out_prefix: Prefix for naming output files (default: multi_omics).

## Produced Outputs

<out_prefix>_embedding.npy: NumPy array of shape (n_samples, n_components) containing PCA embeddings.

<out_prefix>_labels.npy: NumPy array of cluster labels (length = n_samples).

<out_prefix>_clusters.html: Interactive Plotly dashboard for exploring cluster assignments.

<out_prefix>_clusters.png: Static image export (requires Kaleido/Orca).

## Logging

The script uses Python's built-in logging module to report:

File loading summaries and shapes.

Number of features retained after variance filtering.

Explained variance ratio per PCA component and cumulative summary.

Silhouette score for clustering quality.

## Advanced Customization

Reproducibility: Modify random_state parameters in PCA and KMeans for consistent results.

##A lternative embeddings: Swap PCA for UMAP or t-SNE in integrate_data.

##Hyperparameter tuning: Wrap pipeline in GridSearchCV to optimize n_components, n_clusters, and variance_thresh.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/JaCar-868/Disease-Progression/blob/main/LICENSE) file for details.
