# Feature_Engineering 
# Please use the program in Founder_Analysis_after_resampling!!!!!
Clustering and Predict  Entrepreneur

1. Based on hierarchical clustering and decision tree (Hierarchical Founder Analysis.py in the Founder_Analysis_after_resampling dir.)

2. Based on association rules mining (main.py)

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
```

# Example:  

```python
new_entrepreneur = pd.DataFrame([{
   "feature1": 0.5,
   "feature2": 2.0,
   "feature3": -1.0
}])
results = analyzer.classify_new_founder(new_founder)
```

### Example Output

When you run the classifier, you'll get detailed results like this:

```python
Classification Results:
Main Cluster: 2
Decision Path: previous_startup_funding_experience_as_ceo <= 3.50 AND education_institution <= 3.50 AND nasdaq_leadership <= 0.50 AND personal_branding > -0.49
Leaf Node Statistics:
- Success rate in leaf: 56.0%
```

This output tells you:
1. The founder belongs to Main Cluster 2
2. The specific decision path taken through the tree
3. The success rate for similar founders in this leaf node (56.0%)

The decision path can be interpreted as:
- Raised less than 50M USD as CEO
- Not from a top-20 ranked university
- No leadership role in NASDAQ companies
- Above average personal branding

Using these results, you can understand both the classification and the reasoning behind it, making it valuable for both prediction and insight generation.

# README (For Association Rules Mining)

This project is designed to analyze founder characteristics and predict success probabilities using frequent pattern mining and clustering techniques. It implements the Apriori algorithm for finding frequent itemsets in founder data and uses statistical analysis to identify patterns associated with founder success.

## Features

- Frequent pattern mining using Apriori algorithm
- Cluster analysis of founder characteristics
- Success probability prediction
- Real-world probability scaling
- Confidence interval calculations
- Detailed clustering analysis with visualization
- Evaluation of prediction accuracy

## Project Structure

```
.
├── config.py               # Configuration parameters
├── founder_clustering.py   # Clustering analysis implementation
├── main.py                # Main analysis pipeline
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd founder-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- mlxtend
- scikit-learn
- tqdm
- tabulate

## Usage

### Basic Analysis

```python
from config import AnalysisConfig
from main import FounderAnalyzer

# Configure analysis parameters
config = AnalysisConfig(
    base_feature=None,           # Filter by specific feature
    feature_value=None,          # Value of the base feature
    exclude_features=None,       # Features to exclude
    persona=None,               # Filter by persona
    feature_combination=1,       # Max number of features in combinations
    min_sample=30,              # Minimum sample size
    sample_size=8800,           # Total sample size
    decreasing_prob=True,       # Sort by decreasing probability
    include_negative=False,      # Include negative indicators
    cluster_weights=[5, 3, 0, 0, 0, 0]  # Weights for different clusters
)

# Initialize analyzer and run analysis
analyzer = FounderAnalyzer(config)
analyzer.analyze("your_data.csv")
```

### Saving and Loading Clusters

```python
# Save clustering results
analyzer.save_clusters('cluster_results.json')

# Load existing clustering results
analyzer.load_clusters('cluster_results.json')
```

### Predicting New Founder Success

```python
# Example founder features
founder_features = {
    'feature1': 'value1',
    'feature2': 'value2',
    # ... more features
}

analyzer.predict_new_founder(founder_features)
```

### Evaluating Predictions

```python
results = analyzer.evaluate_predictions("your_data.csv", start_idx=0, end_idx=8800)
```

## Configuration Options

### AnalysisConfig Parameters

- `base_feature`: Filter analysis by a specific feature
- `feature_value`: Value of the base feature to filter by
- `exclude_features`: List of features to exclude from analysis
- `persona`: Filter founders by specific persona
- `feature_combination`: Maximum number of features to combine (1-3 recommended)
- `min_sample`: Minimum sample size for pattern consideration
- `sample_size`: Total sample size to analyze
- `num_results`: Number of top results to display
- `decreasing_prob`: Sort by decreasing probability if True
- `confidence_level`: Confidence level for intervals (default: 0.95)
- `real_world_scaling`: Scaling factor for real-world probabilities
- `include_negative`: Include negative indicators in analysis
- `cluster_weights`: Weights for different success clusters [extremely_high, very_high, high, low, very_low, extremely_low]

## Data Format

The input CSV file should contain founder data with the following columns:
- `founder_uuid`: Unique identifier for each founder
- `name`: Founder name
- `org_name`: Organization name
- `success`: Binary indicator of success (0 or 1)
- Additional feature columns containing founder characteristics

## Output

The analysis provides:
1. Frequent patterns in founder characteristics
2. Success probabilities with confidence intervals
3. Cluster analysis results
4. Real-world scaled probabilities
5. Detailed cluster statistics and visualization

## Notes

- Adjust `min_sample` and `sample_size` based on your dataset size
- The `cluster_weights` parameter can be tuned to adjust the importance of different success clusters in predictions
- Use `include_negative=True` if you want to consider negative indicators in the analysis
- The real-world scaling factor can be adjusted based on your domain knowledge


