# Feature_Engineering
Clustering and Predict  Entrepreneur

Based on association rules mining (vela.py)

Based on hierarchical clustering and decision tree (Hierarchical Founder Analysis)


# README (for Hierarchical Founder Analysis)

## Overview

This program performs a hierarchical analysis on entrepreneur data to discover and understand patterns that lead to success. It first identifies main clusters, then within each main cluster, uses a decision tree to find and characterize subclusters. Unlike black-box models, this approach is inherently **interpretable**: the decision trees produce human-readable rules that clearly explain why certain entrepreneurs fall into particular subgroups. By examining these rules and associated metrics, you can understand the key features driving cluster formation and subcluster differentiation.

## Features

1. **Main Cluster Extraction**: Automatically clusters the dataset into main clusters.
2. **Decision Tree Training**: For each main cluster, trains a decision tree to identify subclusters.
3. **Threshold Value Restoration**: If features were scaled, the program restores decision tree thresholds to their original feature scale for easier interpretation.
4. **Leaf Node Analysis**: Examines each leaf node (subcluster) to compute success rates, normalized success rates, and identify features significantly different from the overall population.
5. **Output**:
   - `decision_tree_cluster_X.png`: A decision tree visualization for each main cluster (X is the cluster number).
   - `decision_rules_cluster_X.txt`: A text file containing decision rules for each main cluster.
   - `founder_clusters_analysis.xlsx`: An Excel file summarizing main clusters, subclusters, and combined clusters.
   - Results in memory: Arrays/lists of main clusters, subclusters, and labels returned by the `fit_transform` method.
6. **Classifying a New Entrepreneur**: After fitting, you can apply the model to a new entrepreneur’s feature data to determine which main cluster and subcluster they would belong to, leveraging the explicit, interpretable rules from the decision trees.

## Interpretability

This method is highly interpretable:
- Each subcluster is defined by transparent, human-readable decision rules derived from decision trees.
- Visualized decision trees and exported rule files help you understand the logical path from features to cluster membership.
- Identified significant features help explain why certain groups differ from the overall population.

## Requirements

- Python 3.x
- Required packages:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

To install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Input Data Requirements

1. **Dataset `df`**: A `pandas.DataFrame` containing all relevant feature columns and a binary target column indicated by `success_column`.
2. **Parameters**:
   - `success_column`: The name of the binary column indicating success/failure.
   - `n_main_clusters`: The number of main clusters to identify.
   - `min_subcluster_size`: The minimum number of samples required in each subcluster (leaf node).
   - `real_world_success_rate`: A rate used to normalize the success rate in the results.

## Usage Steps

1. **Prepare Your Data**:
   - Ensure `df` contains all required features and a binary success/failure column.

2. **Instantiate and Configure**:
   - Create an instance of `TwoStageFounderAnalysis`.
   - Set `success_column`, `n_main_clusters`, `min_subcluster_size`, and `real_world_success_rate`.

3. **Run the Program**:
   - Call the `fit_transform(df)` method on your analyzer instance.
   - The program will:
     - Identify main clusters.
     - Train decision trees for each main cluster.
     - Generate visualization images, decision rule text files, and the Excel summary.
     - Return `main_clusters`, `subclusters`, and `labels`.

4. **Check the Results**:
   - After execution, you will find files like `decision_tree_cluster_1.png`, `decision_rules_cluster_1.txt`, and `founder_clusters_analysis.xlsx` in the current directory.
   - The returned `main_clusters`, `subclusters`, and `labels` provide information on the hierarchical structure discovered.

5. **Classify a New Entrepreneur**:
   - After fitting, you can use the trained model (stored within the analyzer) to predict which cluster a new entrepreneur would belong to by passing their feature data through the decision trees.  
   - For example:  
     ```python
     new_entrepreneur = pd.DataFrame([{
         "feature1": 0.5,
         "feature2": 2.0,
         "feature3": -1.0
     }])
     results = analyzer.classify_new_founder(new_founder)
     ```
   - This will tell you exactly which main cluster and subcluster they are most similar to, and you can refer to the decision rules to understand why.

## Notes

- If a main cluster has fewer samples than `min_subcluster_size`, it is skipped.
- If the target column has no variability within a main cluster (e.g., all successes or all failures), no decision tree splitting is performed for that cluster.
- If scaling was applied, ensure that the scaler is fitted on the training data before running the analysis so that threshold values can be correctly inverted.

## Example

```python
import pandas as pd
from Hierarchical Founder Analysis import TwoStageFounderAnalysis

df = pd.read_csv("Data.csv")
analyzer = TwoStageFounderAnalysis(
    success_column="is_success",
    n_main_clusters=5,
    min_subcluster_size=15,
    real_world_success_rate=0.019
)

main_clusters, subclusters, labels = analyzer.fit_transform(df)

# Classify a new entrepreneur
new_entrepreneur = pd.DataFrame([{
    "feature1": 1.2,
    "feature2": 0.3,
    "feature3": 5.0
}])
results = analyzer.classify_new_founder(new_founder)
```

By following these steps, you can easily apply hierarchical founder analysis to your dataset, understand the logic behind cluster assignments, and classify new individuals in a fully interpretable manner.
