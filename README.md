## Disclaimer

### Dataset cannot be upload due to size limit

### Sampling Strategy

Due to hardware limitations (8GB RAM) and the quadratic memory complexity of Hierarchical Agglomerative Clustering (HAC), clustering was performed on a sampled subset of the dataset instead of the full dataset.

Key points:

- A single shared random sample was used for both KMeans and HAC to ensure fair comparison.
- The sample size was dynamically limited to avoid memory overflow.
- Results (Silhouette Score and cluster assignments) reflect the structure of the sampled data, not the entire dataset.
- Different random samples may produce slightly different clustering results.

This approach ensures computational feasibility while preserving representative data structure.

---

### Step 2 – Feature Engineering Assumptions

Feature transformations were applied to improve clustering quality:

- Time-based features were converted into cyclical representations (sin/cos transformation) to reflect their periodic nature.
- Log transformations were applied to skewed count-based features to reduce the influence of extreme values.
- Redundant features were reduced to minimize multicollinearity.

These preprocessing decisions directly influence clustering outcomes. Different feature engineering strategies may lead to different optimal cluster structures and Silhouette Scores.

---

### Step 6 – Clustering Interpretation

The number of clusters (k) was initially estimated using the Elbow Method, but cluster quality was evaluated using the Silhouette Score.

Important considerations:

- The Elbow Method does not guarantee optimal separation.
- Silhouette Scores observed in this project (~0.11–0.16) indicate weak to moderate cluster structure.
- Low Silhouette Scores may suggest that the dataset does not contain strongly separable natural clusters.
- KMeans and HAC may assign different cluster labels because they rely on different optimization principles.

Clustering results should therefore be interpreted as exploratory rather than definitive segmentation.
