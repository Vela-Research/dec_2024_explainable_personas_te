import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from typing import Optional, List, Dict, Tuple
import json
from pathlib import Path
from tqdm import tqdm

from founder_clustering import FounderClusterAnalyzer
from config import AnalysisConfig


class FounderAnalyzer:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.founders_data = None
        self.categorical_data_encoded = None
        self.random_success_prob = None
        self.cluster_analyzer = None

    def _preprocess_column(self, column: pd.Series) -> pd.Series:
        """
        Preprocess a column to handle list-type data and other data types appropriately.

        When we encounter a list in a column, we convert it to a string representation
        so that pandas can properly handle it during the encoding process.
        """
        if column.dtype == 'object':
            # Check if the column contains any lists
            if column.apply(lambda x: isinstance(x, list)).any():
                # For list-type data, convert to a consistent string representation
                return column.apply(lambda x: ','.join(sorted(x)) if isinstance(x, list) else str(x))
        return column

    def load_and_prepare_data(self, file_path: str) -> None:
        """Load and prepare the founders data for analysis"""
        # Load the data
        self.founders_data = pd.read_csv(file_path)

        # Take a sample
        sample_data = self.founders_data.sample(
            self.config.sample_size,
            random_state=1
        )

        # Handle persona filtering if specified
        if self.config.persona:
            sample_data = self._filter_by_persona(sample_data)

        self.founders_data = self._prepare_categorical_data(sample_data)

    def _filter_by_persona(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data based on specified persona"""
        # Convert string representation of lists to actual lists
        data["persona"] = data["persona"].apply(
            lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x
        )

        filtered_data = data[
            data["persona"].apply(
                lambda persona_list: any(
                    p.startswith(self.config.persona) for p in persona_list
                )
            )
        ]
        print(f"Found {len(filtered_data)} entrepreneurs with persona: {self.config.persona}")
        return filtered_data

    def _prepare_categorical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare categorical data for analysis, with optional filtering of negative indicators.
        The method processes the data differently based on the include_negative configuration:
        - If True: Keeps all features including negative indicators (_0, _False, _nope)
        - If False: Filters out negative indicators to focus on positive characteristics
        """
        # Remove non-feature columns
        exclude_cols = ["founder_uuid", "name", "org_name"]
        if self.config.exclude_features:
            exclude_cols.extend(self.config.exclude_features)

        categorical_data = data.drop(columns=exclude_cols, errors='ignore')

        # Select categorical and boolean columns
        categorical_data = categorical_data.select_dtypes(
            include=["int64", "bool", "object"]
        ).copy()

        # Preprocess each column to handle lists and other special data types
        for column in categorical_data.columns:
            if column != 'success':  # Don't process the target variable
                categorical_data[column] = self._preprocess_column(categorical_data[column])

        # Create dummy variables
        self.categorical_data_encoded = pd.get_dummies(
            categorical_data.drop(columns=["success"]),
            columns=categorical_data.drop(columns=["success"]).columns
        ).astype(bool)

        # Filter out negative indicators if include_negative is False
        if not self.config.include_negative:
            negative_patterns = ['_0$', '_False$', '_nope$']
            columns_to_keep = [
                col for col in self.categorical_data_encoded.columns
                if not any(col.endswith(pattern.replace('$', '')) for pattern in negative_patterns)
            ]
            self.categorical_data_encoded = self.categorical_data_encoded[columns_to_keep]
            print(
                f"Filtered out {len(self.categorical_data_encoded.columns) - len(columns_to_keep)} negative indicator features")

        return categorical_data

    def find_patterns(self) -> pd.DataFrame:
        """
        Find patterns using Apriori algorithm including all possible feature combinations
        """
        min_support = self.config.min_sample / len(self.categorical_data_encoded)

        # Get frequent itemsets including all features
        frequent_itemsets = apriori(
            self.categorical_data_encoded,
            min_support=min_support,
            use_colnames=True,
            verbose=0,
            max_len=self.config.feature_combination
        )

        if frequent_itemsets.empty:
            raise ValueError("No frequent itemsets found. Try adjusting min_support.")

        # Filter by base feature if specified
        if (self.config.base_feature and self.config.base_feature != "None" and
                self.config.feature_value and self.config.feature_value != "None"):

            target_feature = f"{self.config.base_feature}_{self.config.feature_value}"
            filtered_itemsets = frequent_itemsets[
                frequent_itemsets["itemsets"].apply(lambda x: target_feature in x)
            ]

            if filtered_itemsets.empty:
                print(f"No patterns found containing {target_feature}")
                return frequent_itemsets

            return filtered_itemsets

        return frequent_itemsets

    def calculate_success_metrics(self, frequent_itemsets: pd.DataFrame) -> pd.DataFrame:
        """Calculate success metrics for frequent itemsets with properly scaled confidence intervals"""
        self.random_success_prob = (self.founders_data["success"] == 1).mean() * 100

        metrics = []
        for itemset in frequent_itemsets["itemsets"]:
            filtered_data = self.founders_data[
                self.categorical_data_encoded[list(itemset)].all(axis=1)
            ]

            if len(filtered_data) > 0:
                # Calculate base success probability
                success_prob = (filtered_data["success"].mean() * 100)

                # Calculate real world probability
                real_world_prob = success_prob * (self.config.real_world_scaling / self.random_success_prob)

                # Calculate confidence interval for the real world probability
                confidence_interval = self._calculate_confidence_interval(
                    success_prob / 100,
                    len(filtered_data),
                    self.config.real_world_scaling / self.random_success_prob
                )
            else:
                success_prob = 0
                real_world_prob = 0
                confidence_interval = (0, 0)

            metrics.append({
                "itemsets": tuple(sorted(itemset)),  # Sort items for consistent display
                "success_probability": success_prob,
                "sample_count": len(filtered_data),
                "likelihood_of_success": success_prob / self.random_success_prob,
                "real_world_prob": real_world_prob,
                "confidence_interval_95": confidence_interval
            })

        results_df = pd.DataFrame(metrics)

        # Set display options for better visibility
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)

        return results_df

    def _calculate_confidence_interval(
            self,
            probability: float,
            sample_size: int,
            scaling_factor: float
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for real world probability

        Parameters:
        probability: The base success probability (as a proportion)
        sample_size: Number of samples
        scaling_factor: Factor to scale the confidence interval to real world values
        """
        z_score = {0.95: 1.96, 0.99: 2.576}[self.config.confidence_level]
        std_err = np.sqrt((probability * (1 - probability)) / sample_size)
        margin = z_score * std_err * 100 * scaling_factor

        base = probability * 100 * scaling_factor
        return (
            round(max(0, base - margin), 2),
            round(min(100, base + margin), 2)
        )

    def format_results(self, results: pd.DataFrame) -> None:
        """
        Format and display results in a simple table with dashed lines separating each feature set.
        """
        # Create a copy of results for formatting
        formatted_results = results.copy()

        # Rename columns for clearer display
        formatted_results.columns = [
            'Feature Sets',
            'Success Rate(%)',
            'Sample Size',
            'Success Likelihood',
            'Real World Prob(%)',
            '95% Conf. Interval'
        ]

        # Format numeric columns
        formatted_results['Success Rate(%)'] = formatted_results['Success Rate(%)'].round(2)
        formatted_results['Success Likelihood'] = formatted_results['Success Likelihood'].round(2)
        formatted_results['Real World Prob(%)'] = formatted_results['Real World Prob(%)'].round(2)

        # Format feature combinations - one feature per line
        formatted_results['Feature Sets'] = formatted_results['Feature Sets'].apply(
            lambda x: '\n'.join([str(item) for item in x])
        )

        # Format confidence intervals
        formatted_results['95% Conf. Interval'] = formatted_results['95% Conf. Interval'].apply(
            lambda x: f"({x[0]:.2f}, {x[1]:.2f})"
        )

        # Create and display table using tabulate with dashed line separator
        from tabulate import tabulate

        print("\nFounder Success Analysis Results:")

        print(tabulate(
            formatted_results,
            headers='keys',
            tablefmt='grid',  # Using 'grid' format for dashed lines between rows
            showindex=False,
            numalign='right',
            stralign='left'
        ))

    def perform_clustering(self) -> Dict:
        """Perform clustering analysis"""
        if self.cluster_analyzer is None:
            frequent_itemsets = self.find_patterns()
            self.cluster_analyzer = FounderClusterAnalyzer(frequent_itemsets, self.founders_data, self.config)

        # Classify using 1 standard deviation
        clusters = self.cluster_analyzer.cluster_itemsets(n_std=1)

        # Analyze clusters
        cluster_results = self.cluster_analyzer.analyze_clusters()

        # Output results
        print("\nFrequent Itemsets Clustering Results:")
        sorted_clusters = sorted(
            cluster_results.items(),
            key=lambda x: x[1]['avg_success_rate'],
            reverse=True
        )

        for cluster_name, analysis in sorted_clusters:
            print(f"\n{cluster_name}:")
            print(f"Number of patterns: {analysis['size']}")
            print(f"Average success rate: {analysis['avg_success_rate']:.2%}")
            print(
                f"Success rate range: {analysis['success_rate_range'][0]:.2%} - {analysis['success_rate_range'][1]:.2%}")
            print(f"Total founders covered: {analysis['total_founders']}")

            print("\nMost common features in patterns:")
            for feature, proportion in analysis['common_features']:
                print(f"  - {feature}: {proportion:.2%}")

            print("\nAll patterns in this cluster (sorted by success rate):")
            all_patterns = sorted(
                self.cluster_analyzer.clusters[cluster_name],
                key=lambda x: x['success_rate'],
                reverse=True
            )
            for i, pattern in enumerate(all_patterns, 1):
                print(f"\n  {i}. {pattern['itemset']}")
                print(f"     Success rate: {pattern['success_rate']:.2%}")
                print(f"     Founders: {pattern['num_founders']}")

        return cluster_results

    def analyze(self, file_path: str) -> None:
        """Extended analysis method"""
        self.load_and_prepare_data(file_path)
        frequent_itemsets = self.find_patterns()
        results = self.calculate_success_metrics(frequent_itemsets)

        # Sort and get results
        results_sorted = results.sort_values(
            by="real_world_prob",
            ascending=not self.config.decreasing_prob
        ).head(self.config.num_results)

        # Display results
        self.format_results(results_sorted)

        # Perform clustering analysis and save results
        self.cluster_results = self.perform_clustering()

    def save_clusters(self, file_path: str = 'cluster_results.json'):
        """Save clustering results to a JSON file"""
        if not hasattr(self, 'cluster_results') or not hasattr(self, 'cluster_analyzer'):
            print("Please run cluster analysis first")
            return

        def convert_to_serializable(obj):
            """Convert non-JSON-serializable objects to a serializable form"""
            if isinstance(obj, (frozenset, set)):
                return list(obj)
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        # Prepare data to save
        save_data = {
            'cluster_results': convert_to_serializable(self.cluster_results),
            'clusters': {
                cluster_name: [
                    {
                        'itemset': list(pattern['itemset']),  # Convert frozenset to list
                        'success_rate': pattern['success_rate'],
                        'num_founders': pattern['num_founders']
                    }
                    for pattern in patterns
                ]
                for cluster_name, patterns in self.cluster_analyzer.clusters.items()
            },
            'statistics': {
                'mean_success': float(self.cluster_analyzer.itemsets_with_success['success_rate'].mean()),
                'std_success': float(self.cluster_analyzer.itemsets_with_success['success_rate'].std())
            }
        }

        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2)

        print(f"Clustering results have been saved to: {file_path}")

    def load_clusters(self, file_path: str = 'cluster_results.json'):
        """Load clustering results from a JSON file"""
        if not Path(file_path).exists():
            print(f"Clustering results file not found: {file_path}")
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.cluster_results = data['cluster_results']
            clusters = {
                cluster_name: [
                    {
                        'itemset': frozenset(pattern['itemset']),
                        'success_rate': pattern['success_rate'],
                        'num_founders': pattern['num_founders']
                    }
                    for pattern in patterns
                ]
                for cluster_name, patterns in data['clusters'].items()
            }

            if not hasattr(self, 'cluster_analyzer'):
                # Pass the config parameter
                self.cluster_analyzer = FounderClusterAnalyzer(None, None, self.config)

            self.cluster_analyzer.clusters = clusters

            print("Clustering results loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading clustering results: {e}")
            return False

    def predict_new_founder(self, founder_features: dict) -> None:
        """Predict the success probability of a new founder"""
        if not hasattr(self, 'cluster_results'):
            # Attempt to load clustering results
            if not self.load_clusters():
                print("Please run cluster analysis first or provide a valid clustering results file")
                return

        # Use saved clustering results for prediction
        prediction = self.cluster_analyzer.predict_success_probability(founder_features)

        # Output results
        print("\nFounder Success Prediction Results:")
        print("\nCluster Probabilities:")
        sorted_probs = sorted(
            prediction['cluster_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for cluster, prob in sorted_probs:
            if prob > 0:
                print(f"{cluster}:")
                print(f"  Probability: {prob:.2%}")
                print(f"  Cluster Success Rate: {prediction['cluster_success_rates'][cluster]:.2%}")

        print(f"\nOverall Predicted Success Rate: {prediction['predicted_success_rate']:.2%}")

    def evaluate_predictions(self, data_file: str, start_idx: int = 0, end_idx: int = 8800):
        """Evaluate prediction performance"""
        fd = pd.read_csv(data_file)

        # Store prediction results
        success_predictions = []  # Predicted probabilities for successful founders
        failure_predictions = []  # Predicted probabilities for failed founders

        # Use a single progress bar
        with tqdm(range(start_idx, end_idx), desc="Evaluating predictions") as pbar:
            for i in pbar:
                founder = fd.iloc[i]
                prediction = self.cluster_analyzer.predict_success_probability(founder)

                pred_prob = prediction['predicted_success_rate']
                actual_success = founder['success']

                if actual_success == 1:
                    success_predictions.append(pred_prob)
                else:
                    failure_predictions.append(pred_prob)

        # Calculate statistical results
        avg_success_pred = np.mean(success_predictions) if success_predictions else 0
        avg_failure_pred = np.mean(failure_predictions) if failure_predictions else 0

        print("\nPrediction Evaluation Results:")
        print(f"Number of actual successful founders: {len(success_predictions)}")
        print(f"Number of actual failed founders: {len(failure_predictions)}")
        print(f"Average predicted probability for successful founders: {avg_success_pred:.2%}")
        print(f"Average predicted probability for failed founders: {avg_failure_pred:.2%}")
        print(f"Difference in predicted probabilities: {avg_success_pred - avg_failure_pred:.2%}")

        return {
            'success_predictions': success_predictions,
            'failure_predictions': failure_predictions,
            'avg_success_pred': avg_success_pred,
            'avg_failure_pred': avg_failure_pred
        }


# Example usage
if __name__ == "__main__":
    # Configure analysis parameters
    config = AnalysisConfig(
        base_feature=None,
        feature_value=None,
        exclude_features=None,
        persona=None,
        feature_combination=2,
        min_sample=30,
        sample_size=8800,
        decreasing_prob=True,
        include_negative=False,
        cluster_weights=[5, 3, 0, 0, 0, 0]
    )

    # Initialize analyzer and run analysis
    analyzer = FounderAnalyzer(config)
    analyzer.analyze("(December 2024)_ Founders data - feature_engineered.csv")
    analyzer.save_clusters('cluster_results.json')

    # Test Data
    results = analyzer.evaluate_predictions("(December 2024)_ Founders data - feature_engineered.csv")
