# Clustering Toolbox

An interactive Streamlit dashboard for exploring and applying clustering algorithms to your own data. Upload any CSV, preprocess it, and experiment with multiple clustering methods and visualizations — no coding required.

---

## Features

### Data Ingestion & Preprocessing
- Upload any CSV file and preview it instantly
- Optionally designate an ID column that is excluded from clustering but preserved in the output
- Automatic detection of non-numeric columns, with a one-click option to remove them
- Missing value detection with an option to drop incomplete rows
- Automatic standardization (zero mean, unit variance) before clustering

### Dimensionality Reduction
- Optional PCA with a configurable number of components
- Explained variance bar chart and cumulative variance breakdown per component
- Cluster on the raw features or on the PCA-reduced space

### Clustering Algorithms

| Algorithm | Tuning Aids |
|---|---|
| KMeans | Elbow method plot |
| Gaussian Mixture Model | AIC / BIC plot, choice of covariance type |
| DBSCAN | k-distance plot to guide eps selection |
| Agglomerative Clustering | Interactive dendrogram with truncation control |
| Spectral Clustering | Configurable number of nearest neighbors |

### Cluster Analysis
- Cluster centroid means and within-cluster variances on the original (non-standardized) scale
- DBSCAN noise-point count reported separately
- Download the full dataset with cluster labels as a CSV

### Visualizations
- **Pairwise Scatter Plot** — all feature pairs colored by cluster label
- **PCA 2D Scatter Plot** — fast two-component projection of the clustering space
- **t-SNE** — non-linear 2D embedding with configurable perplexity, learning rate, and iterations

---

## Getting Started

### Prerequisites
- Python 3.8 or higher

### Installation

```bash
git clone https://github.com/<your-username>/clustering-toolbox.git
cd clustering-toolbox
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run Streamlit_Clustering_Toolbox.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage

1. **Upload** a CSV file using the file uploader.
2. **Select an ID column** if your data has a row identifier you want to keep but not cluster on.
3. **Remove non-numeric columns** if the dataset contains categorical or text fields.
4. **Select features** to include in clustering using the multiselect widget.
5. **Remove missing rows** if your data has NaN values.
6. Optionally **apply PCA** to reduce dimensionality before clustering.
7. **Choose a clustering algorithm** and tune its parameters using the interactive controls.
8. Inspect the **cluster labels**, **centroid statistics**, and **visualizations**.
9. **Download** the labeled dataset as a CSV.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Dashboard framework |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Clustering algorithms, PCA, t-SNE, scaling |
| `scipy` | Hierarchical linkage and dendrogram rendering |
| `matplotlib` | Plot rendering |
| `seaborn` | Statistical visualizations |
