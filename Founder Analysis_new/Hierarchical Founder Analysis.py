import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt
from copy import deepcopy
class TwoStageFounderAnalysis:
    def __init__(self, n_main_clusters=5, min_subcluster_size=15,
                 real_world_success_rate=0.019, success_column='success'):
        self.n_main_clusters = n_main_clusters
        self.min_subcluster_size = min_subcluster_size
        self.real_world_success_rate = real_world_success_rate
        self.success_column = success_column
        self.dataset_success_rate = None
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

        self.cluster_trees = {}
        self.main_cluster_labels = None
        self.X = None

    def preprocess_data(self, df):
        """Preprocess the data and calculate dataset statistics"""
        print("Preprocessing data...")

        # Verify success column exists
        if self.success_column not in df.columns:
            raise ValueError(f"Success column '{self.success_column}' not found in dataset")

        # Calculate dataset success rate
        self.dataset_success_rate = df[self.success_column].mean()
        print(f"Dataset success rate: {self.dataset_success_rate:.1%}")

        # Select numerical features
        numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
        numerical_features = numerical_features[numerical_features != self.success_column]

        self.feature_names = numerical_features.tolist()
        print(f"Selected {len(self.feature_names)} features")

        # Handle missing values and scale
        X = self.imputer.fit_transform(df[numerical_features])
        X = self.scaler.fit_transform(X)

        return X, df

    def create_main_clusters(self, X):
        """Create main cluster groups using hierarchical clustering"""
        print("Creating main clusters...")
        clustering = AgglomerativeClustering(
            n_clusters=self.n_main_clusters,
            metric='euclidean',
            linkage='ward'
        )
        main_cluster_labels = clustering.fit_predict(X)
        linkage_matrix = linkage(X, method='ward')

        return main_cluster_labels, linkage_matrix

    def analyze_main_clusters(self, df, cluster_labels):
        """Analyze characteristics and success rates of main clusters"""
        print("Analyzing main clusters...")
        results = []

        for cluster in range(self.n_main_clusters):
            mask = cluster_labels == cluster
            cluster_data = df[mask]

            if len(cluster_data) < self.min_subcluster_size:
                continue

            success_rate = cluster_data[self.success_column].mean()
            total_samples = len(cluster_data)
            successful_samples = cluster_data[self.success_column].sum()

            feature_stats = []
            for feature in self.feature_names:
                feature_mean = cluster_data[feature].mean()
                overall_mean = df[feature].mean()
                feature_std = df[feature].std()

                if len(cluster_data) > 1:
                    z_score = (feature_mean - overall_mean) / (feature_std / np.sqrt(len(cluster_data)))

                    if abs(z_score) > 1.96:  # 95% confidence level
                        feature_stats.append({
                            'feature': feature,
                            'diff': feature_mean - overall_mean,
                            'z_score': z_score
                        })

            results.append({
                'cluster_id': cluster + 1,
                'size': total_samples,
                'success_count': successful_samples,
                'success_rate': success_rate,
                'normalized_success_rate': success_rate * (self.real_world_success_rate / self.dataset_success_rate),
                'significant_features': sorted(feature_stats, key=lambda x: abs(x['z_score']), reverse=True)[:5]
            })

        return pd.DataFrame(results)

    def create_subclusters(self, X, df, main_cluster_labels):
        """Create interpretable subclusters within each main cluster"""
        print("Creating subclusters...")
        results = []
        self.X = X
        self.main_cluster_labels = main_cluster_labels

        for cluster in range(self.n_main_clusters):

            subcluster_counter = 1

            # Get data for this main cluster
            mask = main_cluster_labels == cluster
            X_cluster = X[mask]
            y_cluster = df[self.success_column].values[mask]
            df_cluster = df[mask]

            if len(X_cluster) < self.min_subcluster_size:
                continue

            # Create decision tree
            tree = DecisionTreeClassifier(
                min_samples_leaf=self.min_subcluster_size,
                max_depth=3,
                random_state=42
            )

            if len(np.unique(y_cluster)) < 2:
                continue

            tree.fit(X_cluster, y_cluster)

            self.cluster_trees[cluster] = tree

            plt.figure(figsize=(20, 10))

            def transform_threshold(value, feature_idx):
                return value * self.scaler.scale_[feature_idx] + self.scaler.mean_[feature_idx]

            tree_copy = deepcopy(tree)
            for i in range(tree_copy.tree_.node_count):
                if tree_copy.tree_.feature[i] >= 0:
                    feature_idx = tree_copy.tree_.feature[i]
                    tree_copy.tree_.threshold[i] = transform_threshold(
                        tree_copy.tree_.threshold[i],
                        feature_idx
                    )

            plot_tree(tree_copy,
                      feature_names=self.feature_names,
                      class_names=['Failure', 'Success'],
                      filled=True,
                      rounded=True,
                      fontsize=10)
            plt.savefig(f'decision_tree_cluster_{cluster + 1}.png', bbox_inches='tight', dpi=300)
            plt.close()

            tree_rules = export_text(tree,
                                     feature_names=self.feature_names,
                                     spacing=3)
            with open(f'decision_rules_cluster_{cluster + 1}.txt', 'w') as f:
                f.write(tree_rules)

            # Analyze leaf nodes
            leaf_ids = tree.apply(X_cluster)
            unique_leaves = np.unique(leaf_ids)

            leaf_sizes = [(leaf, (leaf_ids == leaf).sum()) for leaf in unique_leaves]
            sorted_leaves = sorted(leaf_sizes, key=lambda x: x[1], reverse=True)
            print(len(sorted_leaves))
            for leaf, _ in sorted_leaves:
                leaf_mask = leaf_ids == leaf
                leaf_samples = X_cluster[leaf_mask]
                leaf_y = y_cluster[leaf_mask]
                leaf_df = df_cluster[leaf_mask]

                if len(leaf_samples) < self.min_subcluster_size:
                    continue

                # Calculate basic metrics
                success_rate = leaf_y.mean()

                # Calculate feature characteristics relative to main cluster
                feature_stats = []
                for feature in self.feature_names:
                    feature_mean = leaf_df[feature].mean()
                    overall_mean = df[feature].mean()
                    feature_std = df[feature].std()

                    if len(leaf_df) > 1:
                        z_score = (feature_mean - overall_mean) / (feature_std / np.sqrt(len(leaf_df)))

                        if abs(z_score) > 1.96:  # 95% confidence level
                            feature_stats.append({
                                'feature': feature,
                                'diff': feature_mean - overall_mean,
                                'z_score': z_score
                            })

                subcluster_id = f"{cluster + 1}.{subcluster_counter}"
                subcluster_counter += 1

                results.append({
                    'main_cluster_id': cluster + 1,
                    'subcluster_id': subcluster_id,
                    'size': len(leaf_samples),
                    'success_count': leaf_y.sum(),
                    'success_rate': success_rate,
                    'normalized_success_rate': success_rate * (
                            self.real_world_success_rate / self.dataset_success_rate),
                    'significant_features': sorted(feature_stats, key=lambda x: abs(x['z_score']), reverse=True)[:5]
                })

        return pd.DataFrame(results)

    def classify_new_founder(self, new_founder_data):
        """
        Classify a new founder into main cluster and subcluster with detailed explanation
        """
        print("Starting classification process for new founder...")

        original_values = new_founder_data[self.feature_names].iloc[0]

        new_founder_features = new_founder_data[self.feature_names]
        new_founder_features = self.imputer.transform(new_founder_features)
        new_founder_features = self.scaler.transform(new_founder_features)

        # 2. Calculate distances to each main cluster centroid
        distances = {}
        cluster_sizes = {}
        for cluster in range(self.n_main_clusters):
            # Get cluster data
            mask = self.main_cluster_labels == cluster
            cluster_data = self.X[mask]

            # Calculate centroid
            centroid = cluster_data.mean(axis=0)

            # Calculate Euclidean distance
            distance = np.linalg.norm(new_founder_features - centroid)
            distances[cluster + 1] = distance
            cluster_sizes[cluster + 1] = len(cluster_data)

        # Print distances to all clusters
        print("\nDistances to main clusters:")
        for cluster, distance in sorted(distances.items(), key=lambda x: x[1]):
            print(f"Cluster {cluster}: {distance:.3f} (size: {cluster_sizes[cluster]} samples)")

        # Find closest cluster
        closest_cluster = min(distances.items(), key=lambda x: x[1])[0]
        print(f"\nClosest cluster: Cluster {closest_cluster}")

        # 3. Use decision tree to find subcluster
        print("\nFollowing decision tree path:")
        mask = self.main_cluster_labels == (closest_cluster - 1)
        cluster_tree = self.cluster_trees[closest_cluster - 1]

        node = 0
        path = []
        while True:
            if node == -1:
                break

            feature_idx = cluster_tree.tree_.feature[node]
            threshold = cluster_tree.tree_.threshold[node]
            feature_name = self.feature_names[feature_idx]

            scaled_value = new_founder_features[0, feature_idx]
            original_value = original_values[feature_name]

            original_threshold = (
                    threshold * self.scaler.scale_[feature_idx] +
                    self.scaler.mean_[feature_idx]
            )

            print(f"\nChecking {feature_name}:")
            print(f"Value = {original_value:.2f} (scaled: {scaled_value:.2f})")
            print(f"Threshold = {original_threshold:.2f} (scaled: {threshold:.2f})")

            if scaled_value <= threshold:
                print(f"Decision: {original_value:.2f} <= {original_threshold:.2f}, going left")
                path.append(f"{feature_name} <= {original_threshold:.2f}")
                node = cluster_tree.tree_.children_left[node]
            else:
                print(f"Decision: {original_value:.2f} > {original_threshold:.2f}, going right")
                path.append(f"{feature_name} > {original_threshold:.2f}")
                node = cluster_tree.tree_.children_right[node]

        # Get leaf node statistics
        leaf_id = node
        leaf_values = cluster_tree.tree_.value[leaf_id][0]
        total_samples = sum(leaf_values)
        success_rate = leaf_values[1] / total_samples if total_samples > 0 else 0

        print("\nClassification Results:")
        print(f"Main Cluster: {closest_cluster}")
        print(f"Decision Path: {' AND '.join(path)}")
        print(f"Leaf Node Statistics:")
        print(f"- Success rate in leaf: {success_rate:.1%}")

        return {
            'main_cluster': closest_cluster,
            'distances': distances,
            'decision_path': path,
            'leaf_statistics': {
                'total_samples': total_samples,
                'success_rate': success_rate
            }
        }

    def generate_summary_tables(self, main_clusters, subclusters):
        """Generate summary tables with all statistics"""
        print("Generating summary tables...")
        # Calculate totals
        total_samples = main_clusters['size'].sum()
        total_successes = main_clusters['success_count'].sum()

        # Process main clusters
        main_table = main_clusters.copy()
        main_table['success_probability'] = main_table['success_count'] / main_table['size']
        main_table['relative_success_pct'] = (main_table['success_count'] / total_successes)
        main_table['relative_count_pct'] = (main_table['size'] / total_samples)

        # Format features for main clusters
        def format_features(features):
            feature_strs = []
            for f in features:
                direction = "↑" if f['diff'] > 0 else "↓"
                feature_strs.append(f"{f['feature']} {direction} ({abs(f['diff']):.2f})")
            return "\n".join(feature_strs)

        main_table['key_features'] = main_table['significant_features'].apply(format_features)

        # Create main summary
        main_summary = main_table[[
            'cluster_id', 'size', 'success_count', 'success_probability',
            'normalized_success_rate', 'relative_success_pct', 'relative_count_pct',
            'key_features'
        ]].rename(columns={
            'cluster_id': 'Cluster ID',
            'size': 'Total Count',
            'success_count': 'Success Count',
            'success_probability': 'Success Probability',
            'normalized_success_rate': 'Normalized Success Rate',
            'relative_success_pct': 'Relative % of Success',
            'relative_count_pct': 'Relative % of Total',
            'key_features': 'Key Characteristics'
        })

        # Process subclusters
        sub_table = subclusters.copy()
        sub_table['success_probability'] = sub_table['success_count'] / sub_table['size']
        sub_table['relative_success_pct'] = (sub_table['success_count'] / total_successes)
        sub_table['relative_count_pct'] = (sub_table['size'] / total_samples)

        # Add formatted features to subclusters
        sub_table['key_features'] = sub_table['significant_features'].apply(format_features)

        # Create subcluster summary
        sub_summary = sub_table[[
            'main_cluster_id', 'subcluster_id', 'size', 'success_count',
            'success_probability', 'normalized_success_rate', 'relative_success_pct',
            'relative_count_pct', 'key_features'
        ]].rename(columns={
            'main_cluster_id': 'Main Cluster',
            'subcluster_id': 'Subcluster ID',
            'size': 'Total Count',
            'success_count': 'Success Count',
            'success_probability': 'Success Probability',
            'normalized_success_rate': 'Normalized Success Rate',
            'relative_success_pct': 'Relative % of Success',
            'relative_count_pct': 'Relative % of Total',
            'key_features': 'Key Characteristics'
        })

        # Format numeric columns
        format_cols = ['Success Probability', 'Normalized Success Rate',
                       'Relative % of Success', 'Relative % of Total']

        for col in format_cols:
            main_summary[col] = main_summary[col].apply(lambda x: f"{x:.1%}")
            sub_summary[col] = sub_summary[col].apply(lambda x: f"{x:.1%}")

        # Sort both summaries by success probability
        main_summary = main_summary.sort_values('Success Probability',
                                                ascending=False,
                                                key=lambda x: pd.to_numeric(x.str.rstrip('%')) / 100)

        sub_summary = sub_summary.sort_values('Success Probability',
                                              ascending=False,
                                              key=lambda x: pd.to_numeric(x.str.rstrip('%')) / 100)

        print(f"Generated main summary shape: {main_summary.shape}")
        print(f"Generated sub summary shape: {sub_summary.shape}")

        return main_summary, sub_summary

    def get_decision_path(self, tree, feature_names, leaf_id):
        """Get decision path to a specific leaf node with sample counts and purity at each step"""
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        values = tree.tree_.value
        gini = tree.tree_.impurity

        # Find path from root to target leaf node
        path = []
        node_id = 0  # Start from root
        current_path = []

        while node_id != leaf_id:
            # Get current node information
            fail_samples = int(values[node_id][0][0])
            success_samples = int(values[node_id][0][1])
            current_gini = gini[node_id]

            # Get feature name and threshold for current node
            feature_idx = feature[node_id]
            feature_name = feature_names[feature_idx]

            original_threshold = (
                    threshold[node_id] * self.scaler.scale_[feature_idx] +
                    self.scaler.mean_[feature_idx]
            )

            # Create node info
            node_info = (f"{feature_name} (fail: {fail_samples}, success: {success_samples}, "
                         f"gini: {current_gini:.3f})")
            current_path.append(node_info)

            # Determine path direction
            if leaf_id in self._get_descendants(children_left, node_id):
                path.append(current_path[-1] + f" <= {original_threshold:.2f}")
                node_id = children_left[node_id]
            else:
                path.append(current_path[-1] + f" > {original_threshold:.2f}")
                node_id = children_right[node_id]

        # Add leaf node information
        fail_samples = int(values[leaf_id][0][0])
        success_samples = int(values[leaf_id][0][1])
        leaf_gini = gini[leaf_id]
        path.append(f"Final Node: fail: {fail_samples}, success: {success_samples}, gini: {leaf_gini:.3f}")

        return " → ".join(path)

    def _get_descendants(self, children_array, node_id):
        """Get all descendant nodes of a node"""
        descendants = []
        to_check = [children_array[node_id]]

        while to_check:
            current = to_check.pop(0)
            if current != -1:
                descendants.append(current)
                if current < len(children_array):
                    to_check.append(children_array[current])

        return descendants
    def save_summary_tables(self, main_summary, sub_summary):
        """Save summary tables to files and console"""
        print("\nSaving results...")
        with pd.ExcelWriter('founder_clusters_analysis.xlsx') as writer:
            main_summary.to_excel(writer, sheet_name='Main Clusters', index=False)
            sub_summary.to_excel(writer, sheet_name='Subclusters', index=False)
            hierarchical_df = self.generate_hierarchical_table(main_summary, sub_summary)
            hierarchical_df.to_excel(writer, sheet_name='Hierarchical View', index=False)

            # Format hierarchical view
            worksheet = writer.sheets['Hierarchical View']
            workbook = writer.book

            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D9E1F2',
                'border': 1,
                'align': 'center'
            })

            main_format = workbook.add_format({
                'bold': True,
                'bg_color': '#E6E6FA'
            })

            sub_format = workbook.add_format({
                'indent': 2
            })

            # Apply formats
            for col_num, col in enumerate(hierarchical_df.columns):
                worksheet.write(0, col_num, col, header_format)
                worksheet.set_column(col_num, col_num, len(col) + 2)

            for row_num in range(1, len(hierarchical_df) + 1):
                if hierarchical_df.iloc[row_num - 1]['Cluster Level'] == 'Main':
                    worksheet.set_row(row_num, None, main_format)
                elif hierarchical_df.iloc[row_num - 1]['Cluster Level'] == 'Sub':
                    worksheet.set_row(row_num, None, sub_format)

        print("Results saved to 'founder_clusters_analysis.xlsx'")

        # Print to console
        print("\nMain Clusters Summary:")
        print("=" * 120)
        print(main_summary.to_string(index=False))

        print("\nSubclusters Summary:")
        print("=" * 120)
        print(sub_summary.to_string(index=False))

    def generate_hierarchical_table(self, main_summary, sub_summary):
        """Generate a hierarchical view of clusters and their subclusters"""
        print("\nGenerating hierarchical cluster view...")

        hierarchical_rows = []
        sorted_main = main_summary.sort_values('Success Probability',
                                               ascending=False,
                                               key=lambda x: pd.to_numeric(x.str.rstrip('%')) / 100)

        for _, main_cluster in sorted_main.iterrows():
            # Add main cluster
            hierarchical_rows.append({
                'Cluster Level': 'Main',
                'Cluster ID': f"Cluster {main_cluster['Cluster ID']}",
                'Total Count': main_cluster['Total Count'],
                'Success Count': main_cluster['Success Count'],
                'Success Probability': main_cluster['Success Probability'],
                'Normalized Success Rate': main_cluster['Normalized Success Rate'],
                'Relative % of Success': main_cluster['Relative % of Success'],
                'Relative % of Total': main_cluster['Relative % of Total'],
                'Key Characteristics': main_cluster['Key Characteristics']
            })

            # Add subclusters
            cluster_subs = sub_summary[sub_summary['Main Cluster'] == main_cluster['Cluster ID']]
            sorted_subs = cluster_subs.sort_values('Success Probability',
                                                   ascending=False,
                                                   key=lambda x: pd.to_numeric(x.str.rstrip('%')) / 100)

            for _, sub in sorted_subs.iterrows():
                hierarchical_rows.append({
                    'Cluster Level': 'Sub',
                    'Cluster ID': f"└── {sub['Subcluster ID']}",
                    'Total Count': sub['Total Count'],
                    'Success Count': sub['Success Count'],
                    'Success Probability': sub['Success Probability'],
                    'Normalized Success Rate': sub['Normalized Success Rate'],
                    'Relative % of Success': sub['Relative % of Success'],
                    'Relative % of Total': sub['Relative % of Total'],
                    'Key Characteristics': sub['Key Characteristics']
                })

            # Add blank row between clusters
            hierarchical_rows.append({key: '' for key in hierarchical_rows[0].keys()})

        return pd.DataFrame(hierarchical_rows)

    def visualize_clusters(self, linkage_matrix):
        """Create visualization of cluster hierarchy"""
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.savefig('cluster_dendrogram.png')
        plt.close()

    def fit_transform(self, df):
        """Run the complete analysis pipeline"""
        print("\nStarting two-stage founder analysis...")

        # Preprocess data
        X, original_df = self.preprocess_data(df)

        # Create main clusters
        main_cluster_labels, linkage_matrix = self.create_main_clusters(X)

        # Analyze clusters
        main_cluster_results = self.analyze_main_clusters(original_df, main_cluster_labels)
        subcluster_results = self.create_subclusters(X, original_df, main_cluster_labels)
        print(subcluster_results.shape)
        # Generate summaries
        main_summary, sub_summary = self.generate_summary_tables(
            main_cluster_results, subcluster_results)

        # Generate hierarchical view and save all results
        _ = self.generate_hierarchical_table(main_summary, sub_summary)
        self.save_summary_tables(main_summary, sub_summary)

        # Create visualization
        self.visualize_clusters(linkage_matrix)

        print("\nAnalysis complete!")
        return main_cluster_results, subcluster_results, main_cluster_labels

    def extract_decision_path(self, tree, feature_names, leaf_id):
        """Extract the decision path leading to a specific leaf node"""
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        # Find the path to the leaf
        node_id = 0  # Start from root
        path = []
        while node_id != leaf_id:
            # Get feature and threshold for this node
            feature_name = feature_names[feature[node_id]]
            threshold_value = threshold[node_id]

            # Check if we should go left or right
            if leaf_id in self._get_all_children(children_left, node_id):
                path.append(f"{feature_name} <= {threshold_value:.2f}")
                node_id = children_left[node_id]
            else:
                path.append(f"{feature_name} > {threshold_value:.2f}")
                node_id = children_right[node_id]

        return " AND ".join(path)

    def _get_all_children(self, children_array, node_id):
        """Helper function to get all children of a node"""
        children = []
        to_visit = [children_array[node_id]]

        while to_visit:
            current = to_visit.pop()
            if current != -1:  # -1 indicates leaf
                children.append(current)
                if current < len(children_array):
                    to_visit.append(children_array[current])

        return children


df = pd.read_csv("(December 2024)_ Founders data - feature_engineered.csv")

analyzer = TwoStageFounderAnalysis(
    n_main_clusters=5,
    min_subcluster_size=15,
    real_world_success_rate=0.019
)

main_clusters, subclusters, labels = analyzer.fit_transform(df)

# Test a new founder
new_founder = pd.DataFrame([df.iloc[5]])

results = analyzer.classify_new_founder(new_founder)