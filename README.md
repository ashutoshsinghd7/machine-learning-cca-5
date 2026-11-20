# Student Performance Pattern Analysis — PCA & K-Means Clustering

A machine learning mini-project that explores patterns in student performance data using **Principal Component Analysis (PCA)** for dimensionality reduction and **K-Means Clustering** to group students with similar characteristics. Includes an interactive **Gradio** dashboard for live exploration.

> Exploratory/unsupervised analysis only — no grade prediction is made. Final grades (G1/G2/G3) are excluded from clustering and compared against the clusters only afterward.

## Overview

Each student in the dataset is described by 30+ demographic, family, study, and lifestyle attributes. Looking at these features one at a time — or plotting just two of them — hides how they vary together and misses the bigger picture.

This project:
1. Reduces the feature space to 2 principal components with PCA, so it can be visualized in 2‑D.
2. Groups students into clusters of similar profiles with K-Means.
3. Compares each cluster's academic performance (final grade, `G3`) to see whether the natural groupings line up with outcomes.

**Key question:** Can we see natural groups of similar students in one compact view, and how do their final grades differ?

## Dataset

- **Source:** [UCI Machine Learning Repository — Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez & Silva, 2008)
- **File:** `student-mat.csv` (semicolon-delimited), loaded directly from a GitHub-hosted CSV mirror, with an automatic backup URL if the primary source fails
- **Size:** 395 students, 33 original attributes, 0 missing values
- **Features used for clustering** (13, numeric, excluding grades):
  `age`, `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`
- `G1`, `G2`, `G3` (grades) are kept aside and used only afterward to interpret the clusters — never as inputs to PCA or K-Means.

## Approach / Pipeline

```
Load Dataset → Inspect & Clean → EDA → Select Features → Standardize (z-score)
     → PCA (2 components) → K-Means (k=3) → Evaluate & Visualize → Gradio Dashboard
```

1. **Load & inspect** the dataset with Pandas (shape, dtypes, missing values).
2. **EDA** — grade distribution, gender split, study time vs. grade, correlation heatmap.
3. **Feature selection & scaling** — select 13 numeric features, standardize with `StandardScaler`.
4. **PCA** — reduce to 2 components for visualization.
5. **K-Means** — determine `k` with the elbow method, then cluster on the *standardized features* (not the PCA scores).
6. **Evaluate** — silhouette score, cluster-wise feature profiles, grade distribution per cluster.
7. **Dashboard** — an interactive Gradio + Plotly app to re-run clustering for different values of `k`.

## Results

| Metric | Value |
|---|---|
| Students analyzed | 395 |
| Features used | 13 |
| PCA variance explained (PC1 + PC2) | 32.6% |
| Number of clusters (k) | 3 |
| Silhouette score | 0.105 |

| Cluster | Students | Avg. Final Grade (G3) | Min / Max |
|---|---|---|---|
| 0 | 154 | 9.47 | 0 / 19 |
| 1 | 169 | 11.64 | 0 / 20 |
| 2 | 72 | 9.56 | 0 / 18 |

Cluster 1 — students with higher parental education and fewer past failures — has the highest average final grade. The silhouette score (0.105) indicates the clusters overlap more than they separate, which is expected for weakly-correlated, survey-style lifestyle data; this is association, not causation.

## Tech Stack

- **Language:** Python (Google Colab / Jupyter)
- **Data handling:** Pandas, NumPy
- **ML:** scikit-learn (`StandardScaler`, `PCA`, `KMeans`, `silhouette_score`)
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Dashboard:** Gradio

## Getting Started

### Run in Google Colab (recommended)
Open `Student_Performance_PCA_KMeans_project.ipynb` in Google Colab and run all cells top to bottom. The first cell installs the only extra dependencies (`gradio`, `plotly`); everything else is pre-installed in Colab.

### Run locally
```bash
git clone <your-repo-url>
cd <your-repo-name>

pip install pandas numpy matplotlib seaborn scikit-learn gradio plotly

jupyter notebook Student_Performance_PCA_KMeans_project.ipynb
```

Running the notebook end-to-end will:
- Download the dataset automatically
- Generate all EDA and clustering plots inline
- Launch a Gradio dashboard (prints a local + shareable public URL)
- Export clustering results to `Student_Performance_Clustering_Results.csv`

## Project Structure

```
.
├── Student_Performance_PCA_KMeans_project.ipynb   # Main notebook — full pipeline
└── Student_Performance_Clustering_Results.csv      # Generated after running the notebook
```

## Limitations & Future Scope

**Limitations**
- Low silhouette score (0.105) — clusters overlap and are not sharply separated.
- PCA retains only 32.6% of the total variance; a 2-D view does not capture all structure.
- Clustering shows association with grades, not a causal relationship.

**Future Scope**
- Try other clustering methods (DBSCAN, hierarchical clustering).
- Compare linear PCA against Kernel PCA / t-SNE / UMAP.
- Validate the approach on other student-performance datasets.

## References

1. Cortez, P. & Silva, A. (2008). *Using data mining to predict secondary school student performance.* UCI Machine Learning Repository, Student Performance dataset.
2. Pearson, K. (1901). *On lines and planes of closest fit to systems of points in space.* Philosophical Magazine, 2(11), 559–572.
3. MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations.* Proc. 5th Berkeley Symposium on Mathematical Statistics and Probability, 1, 281–297.
4. [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) — PCA, KMeans, StandardScaler.
5. [Gradio documentation](https://www.gradio.app/docs)

## Authors

Machine Learning Mini-Project (CCE IV)

- Om Mahulkar — TY2526AIDC205
- Naitik Wake — TY2526AIDC208
- Parth Patil — TY2526AIDC213
- Ashutosh Singh — TY2526AIDC223
