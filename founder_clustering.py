import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from typing import Dict, List
import json
from pathlib import Path

from config import AnalysisConfig


class FounderClusterAnalyzer:
    def __init__(self, frequent_itemsets: pd.DataFrame, founders_data: pd.DataFrame, config: AnalysisConfig):
        self.frequent_itemsets = frequent_itemsets
        self.founders_data = founders_data
        self.config = config
        self.itemsets_with_success = None
        self.clusters = None

    def analyze_clusters(self) -> Dict:
        """Analyze the frequent itemsets within each success rate range"""
        cluster_analysis = {}

        for cluster_name, itemsets in self.clusters.items():
            # Statistical information
            success_rates = [item['success_rate'] for item in itemsets]
            founder_counts = [item['num_founders'] for item in itemsets]

            # Count feature frequencies
            feature_freq = defaultdict(int)
            for item in itemsets:
                for feature in item['itemset']:
                    feature_freq[feature] += 1

            cluster_analysis[cluster_name] = {
                'size': len(itemsets),
                'avg_success_rate': np.mean(success_rates),
                'success_rate_range': (min(success_rates), max(success_rates)),
                'total_founders': sum(founder_counts),
                'common_features': sorted(
                    [(f, c / len(itemsets)) for f, c in feature_freq.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                'representative_patterns': sorted(
                    itemsets,
                    key=lambda x: x['success_rate'],
                    reverse=True
                )[:5]
            }

        return cluster_analysis

    def calculate_success_rate(self, itemset) -> float:
        """
        Calculate the success rate of founders who meet the specific frequent itemset
        """
        # Find founders who satisfy all features in the frequent itemset
        mask = pd.Series(True, index=self.founders_data.index)
        for feature in itemset:
            feature_name, feature_value = feature.rsplit('_', 1)
            if feature_name in self.founders_data:
                mask &= (self.founders_data[feature_name].astype(str) == feature_value)

        # Calculate the success rate of these founders
        filtered_founders = self.founders_data[mask]
        if len(filtered_founders) > 0:
            return (filtered_founders['success'] == 1).mean()
        return 0.0

    def calculate_itemsets_success_rates(self) -> pd.DataFrame:
        """Calculate the success rate for each frequent itemset"""
        success_rates = []

        for itemset in self.frequent_itemsets['itemsets']:
            # Find founders who satisfy all features in the frequent itemset
            mask = pd.Series(True, index=self.founders_data.index)
            for feature in itemset:
                feature_name, feature_value = feature.rsplit('_', 1)
                if feature_name in self.founders_data:
                    mask &= (self.founders_data[feature_name].astype(str) == feature_value)

            # Calculate success rate
            filtered_founders = self.founders_data[mask]
            if len(filtered_founders) > 0:
                success_rate = (filtered_founders['success'] == 1).mean()
            else:
                success_rate = 0.0

            success_rates.append({
                'itemset': itemset,
                'success_rate': success_rate,
                'num_founders': len(filtered_founders)
            })

        self.itemsets_with_success = pd.DataFrame(success_rates)
        return self.itemsets_with_success

    def cluster_by_success_rate(self, method='density', n_bins=5, bin_method='equal_width', **kwargs) -> Dict:

        if self.itemsets_with_success is None:
            self.calculate_itemsets_success_rates()

        if method == 'density':
            return self._cluster_by_density(**kwargs)
        else:
            return self._cluster_by_bins(n_bins=n_bins, method=bin_method)

    def _cluster_by_density(self, eps=0.05, min_samples=5) -> Dict:
        """Use DBSCAN to perform density-based clustering on success rates"""
        success_rates = self.itemsets_with_success['success_rate'].values.reshape(-1, 1)

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(success_rates)

        # Organize clustering results
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:  # Exclude noise points
                clusters[f"success_cluster_{label}"].append({
                    'itemset': self.itemsets_with_success.iloc[idx]['itemset'],
                    'success_rate': self.itemsets_with_success.iloc[idx]['success_rate'],
                    'num_founders': self.itemsets_with_success.iloc[idx]['num_founders']
                })

        self.clusters = dict(clusters)
        return self.clusters

    def _cluster_by_bins(self, n_bins=5, method='equal_width') -> Dict:
        """Use binning methods to group success rates"""
        success_rates = self.itemsets_with_success['success_rate']

        if method == 'equal_freq':
            # Equal frequency binning
            bins = pd.qcut(success_rates, q=n_bins, labels=False)
        else:
            # Equal width binning
            bins = pd.cut(success_rates, bins=n_bins, labels=False)

        # Organize clustering results
        clusters = defaultdict(list)
        for idx, bin_label in enumerate(bins):
            clusters[f"success_cluster_{bin_label}"].append({
                'itemset': self.itemsets_with_success.iloc[idx]['itemset'],
                'success_rate': self.itemsets_with_success.iloc[idx]['success_rate'],
                'num_founders': self.itemsets_with_success.iloc[idx]['num_founders']
            })

        self.clusters = dict(clusters)
        return self.clusters

    def calculate_feature_success_rates(self) -> pd.DataFrame:
        """Calculate the average success rate for each feature"""
        # Collect all unique features
        all_features = set()
        for itemset in self.frequent_itemsets['itemsets']:
            all_features.update(itemset)

        feature_stats = []
        for feature in all_features:
            # Find all frequent itemsets that include this feature
            containing_itemsets = self.frequent_itemsets[
                self.frequent_itemsets['itemsets'].apply(lambda x: feature in x)
            ]

            # Calculate the average success rate of frequent itemsets containing this feature
            success_rates = []
            total_founders = 0

            for itemset in containing_itemsets['itemsets']:
                # Calculate the success rate of this frequent itemset
                mask = pd.Series(True, index=self.founders_data.index)
                for feat in itemset:
                    feat_name, feat_value = feat.rsplit('_', 1)
                    if feat_name in self.founders_data:
                        mask &= (self.founders_data[feat_name].astype(str) == feat_value)

                filtered_founders = self.founders_data[mask]
                if len(filtered_founders) > 0:
                    success_rate = (filtered_founders['success'] == 1).mean()
                    success_rates.append(success_rate)
                    total_founders += len(filtered_founders)

            if success_rates:
                avg_success_rate = np.mean(success_rates)
            else:
                avg_success_rate = 0.0

            feature_stats.append({
                'feature': feature,
                'avg_success_rate': avg_success_rate,
                'num_patterns': len(containing_itemsets),
                'total_founders': total_founders
            })

        self.feature_success_rates = pd.DataFrame(feature_stats)
        return self.feature_success_rates

    def cluster_features(self, n_bins=5) -> Dict:
        """Categorize features based on their average success rates"""
        if self.feature_success_rates is None:
            self.calculate_feature_success_rates()

        # Use equal frequency binning to categorize features
        self.feature_success_rates['cluster'] = pd.qcut(
            self.feature_success_rates['avg_success_rate'],
            q=n_bins,
            labels=[f'cluster_{i}' for i in range(n_bins)]
        )

        # Organize clustering results
        clusters = defaultdict(list)
        for _, row in self.feature_success_rates.iterrows():
            clusters[row['cluster']].append({
                'feature': row['feature'],
                'avg_success_rate': row['avg_success_rate'],
                'num_patterns': row['num_patterns'],
                'total_founders': row['total_founders']
            })

        self.clusters = dict(clusters)
        return self.clusters

    def cluster_itemsets(self, n_std=2) -> Dict:

        # Calculate success rates
        itemsets_stats = []

        for itemset in self.frequent_itemsets['itemsets']:
            mask = pd.Series(True, index=self.founders_data.index)
            for feature in itemset:
                feature_name, feature_value = feature.rsplit('_', 1)
                if feature_name in self.founders_data:
                    mask &= (self.founders_data[feature_name].astype(str) == feature_value)

            filtered_founders = self.founders_data[mask]
            if len(filtered_founders) > 0:
                success_rate = (filtered_founders['success'] == 1).mean()
            else:
                success_rate = 0.0

            itemsets_stats.append({
                'itemset': itemset,
                'success_rate': success_rate,
                'num_founders': len(filtered_founders)
            })

        self.itemsets_with_success = pd.DataFrame(itemsets_stats)

        # Calculate mean and standard deviation of success rates
        mean_success = self.itemsets_with_success['success_rate'].mean()
        std_success = self.itemsets_with_success['success_rate'].std()

        # Define boundaries for 6 intervals
        boundaries = [
            float('-inf'),
            mean_success - 2 * std_success,
            mean_success - std_success,
            mean_success,
            mean_success + std_success,
            mean_success + 2 * std_success,
            float('inf')
        ]

        # Define interval labels
        labels = [
            'extremely_low_success',  # < μ-2σ
            'very_low_success',  # μ-2σ to μ-σ
            'low_success',  # μ-σ to μ
            'high_success',  # μ to μ+σ
            'very_high_success',  # μ+σ to μ+2σ
            'extremely_high_success'  # > μ+2σ
        ]

        # Use pd.cut for binning
        self.itemsets_with_success['cluster'] = pd.cut(
            self.itemsets_with_success['success_rate'],
            bins=boundaries,
            labels=labels,
            include_lowest=True
        )

        print(f"\nClustering Statistics:")
        print(f"Mean Success Rate (μ): {mean_success:.2%}")
        print(f"Standard Deviation (σ): {std_success:.2%}")
        print(f"\nCluster Boundaries:")
        print(f"- Extremely Low: < {boundaries[1]:.2%} (< μ-2σ)")
        print(f"- Very Low: {boundaries[1]:.2%} to {boundaries[2]:.2%} (μ-2σ to μ-σ)")
        print(f"- Low: {boundaries[2]:.2%} to {boundaries[3]:.2%} (μ-σ to μ)")
        print(f"- High: {boundaries[3]:.2%} to {boundaries[4]:.2%} (μ to μ+σ)")
        print(f"- Very High: {boundaries[4]:.2%} to {boundaries[5]:.2%} (μ+σ to μ+2σ)")
        print(f"- Extremely High: > {boundaries[5]:.2%} (> μ+2σ)")

        # Organize clustering results
        clusters = defaultdict(list)
        for _, row in self.itemsets_with_success.iterrows():
            clusters[row['cluster']].append({
                'itemset': row['itemset'],
                'success_rate': row['success_rate'],
                'num_founders': row['num_founders']
            })

        self.clusters = dict(clusters)
        return self.clusters

    def analyze_feature_clusters(self) -> Dict:
        """Analyze the characteristics of each feature cluster"""
        cluster_analysis = {}

        for cluster_name, features in self.clusters.items():
            # Calculate statistical information
            success_rates = [f['avg_success_rate'] for f in features]

            cluster_analysis[cluster_name] = {
                'size': len(features),
                'avg_success_rate': np.mean(success_rates),
                'success_rate_range': (min(success_rates), max(success_rates)),
                'features': sorted(features, key=lambda x: x['avg_success_rate'], reverse=True)
            }

        return cluster_analysis

    def calculate_similarity(self, founder_features: pd.Series, itemset: set) -> float:

        # Convert founder features to a set format
        founder_feature_set = set()
        for feat_name, feat_value in founder_features.items():
            # Skip unnecessary columns
            if feat_name in ['founder_uuid', 'name', 'org_name', 'success']:
                continue
            # Convert to string to ensure format matches
            feature = f"{feat_name}_{str(feat_value)}"
            founder_feature_set.add(feature)

        # Calculate similarity
        intersection = len(founder_feature_set.intersection(itemset))
        if intersection == 0:
            return 0

        # Use improved similarity calculation
        # Considering that frequent itemsets are usually smaller than individual features, use the size of the frequent itemset as a baseline
        return intersection / len(itemset)

    def predict_success_probability(self, founder_features: pd.Series, similarity_threshold: float = 0.05) -> dict:
        """
        Predict the success rate of a founder using configured cluster weights
        """
        cluster_probabilities = defaultdict(float)
        cluster_weights = defaultdict(float)

        # Get cluster weights
        cluster_importance_weights = {
            'extremely_high_success': self.config.cluster_weights[0],
            'very_high_success': self.config.cluster_weights[1],
            'high_success': self.config.cluster_weights[2],
            'low_success': self.config.cluster_weights[3],
            'very_low_success': self.config.cluster_weights[4],
            'extremely_low_success': self.config.cluster_weights[5]
        }

        # Calculate similarity with frequent itemsets in each cluster
        for cluster_name, patterns in self.clusters.items():
            total_similarity = 0
            weighted_success_rate = 0

            # Get the weight for this cluster
            cluster_weight = cluster_importance_weights.get(cluster_name, 1.0)

            for pattern in patterns:
                similarity = self.calculate_similarity(founder_features, pattern['itemset'])

                if similarity > similarity_threshold:
                    # Apply cluster weight
                    weighted_similarity = similarity * cluster_weight
                    total_similarity += weighted_similarity
                    weighted_success_rate += weighted_similarity * pattern['success_rate']

            if total_similarity > 0:
                cluster_probabilities[cluster_name] = total_similarity
                cluster_weights[cluster_name] = weighted_success_rate / total_similarity

        # Normalize probabilities
        total_probability = sum(cluster_probabilities.values())
        if total_probability > 0:
            normalized_probabilities = {
                k: v / total_probability
                for k, v in cluster_probabilities.items()
            }
        else:
            normalized_probabilities = {
                k: 0 for k in self.clusters.keys()
            }

        # Calculate weighted average success rate
        final_success_rate = sum(
            prob * cluster_weights[cluster]
            for cluster, prob in normalized_probabilities.items()
        )

        return {
            'cluster_probabilities': normalized_probabilities,
            'cluster_success_rates': dict(cluster_weights),
            'predicted_success_rate': final_success_rate
        }

