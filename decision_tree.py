import numpy as np
from ucimlrepo import fetch_ucirepo
from load_adult import get_clean_adult_data


class Node:
    def __init__(self, left=None, right=None, feature_index=None, threshold=None, label=None):
        self.left = left
        self.right = right
        self.feature_index = feature_index
        self.threshold = threshold
        self.label = label


class DecisionTree:
    def __init__(self, max_depth=float('inf'), min_size=0):
        self.max_depth = max_depth
        self.min_size = min_size
        self.pa_indices = None
        self.root = None

    # Split the dataset based on label values relative to threashold (left: <=, right: >)
    def _split(self, X, feature_index, threshold, is_category):
        # Threshold: to the left holds smaller values and to the right holds greater values
        feature_values = X.iloc[:, feature_index]
        if is_category:
            mask = feature_values == threshold
        else:
            mask = feature_values <= threshold
        return mask

    # Compute gini index for left and right group
    def _compute_gini_impurity(self, val, w, mask):
        total_w = np.size(w)
        gini = 0.0
        for indices in mask, ~mask:
            sub_tree_w = np.sum(w[indices])
            if sub_tree_w == 0:
                continue
            score = 0.0
            # Count the total weight of each class label
            class_counts = np.bincount(val[indices], w[indices])
            for count in class_counts:
                proportion = count / sub_tree_w
                score += proportion ** 2
            gini += (1.0 - score) * (sub_tree_w / total_w)
        return gini

    # Find the best split for the dataset in question
    def _find_best_split(self, X, y, w):
        n_features = X.shape[1]
        best_gini, best_split = float('inf'), None
        is_category = None
        for feature_index in range(n_features):
            feature_values = X.iloc[:, feature_index]
            thresholds = np.unique(feature_values)
            is_category = isinstance(feature_values.iloc[0], str)
            for threshold in thresholds:
                mask = self._split(X, feature_index, threshold, is_category)
                gini = self._compute_gini_impurity(y, w, mask)
                for pa_index in self.pa_indices:
                    gini -= self._compute_gini_impurity(
                        X.iloc[:, pa_index].values.flatten(), w, mask)
                # Need better tie-breaking method - currently favors features / thresholds that appear first
                if gini < best_gini:
                    best_gini, best_split = gini, {
                        'feature': feature_index, 'threshold': threshold, 'mask': mask}
        return best_split

    # Find the most frequent label
    def _get_most_frequent_label(self, labels):
        return np.bincount(labels).argmax()

    # Recursively build decision tree
    def _build_tree(self, X, y, w, depth):
        n_samples = X.shape[0]
        leaf_node = Node(label=self._get_most_frequent_label(y))
        if depth >= self.max_depth or n_samples <= self.min_size:
            return leaf_node
        best_split = self._find_best_split(X, y, w)
        mask = best_split['mask']
        if np.size(y[mask]) == 0 or np.size(y[~mask]) == 0:
            return leaf_node
        return Node(self._build_tree(X[mask], y[mask], w[mask], depth+1), self._build_tree(X[~mask], y[~mask], w[~mask], depth+1), best_split['feature'], best_split['threshold'])

    # Train the model
    def fit(self, X, y, w, pa_indices):
        self.pa_indices = pa_indices
        self.root = self._build_tree(X, y, w, 0)

    def _predict(self, node, row):
        if node.label is not None:
            return node.label
        if row.iloc[node.feature_index] <= node.threshold:
            return self._predict(node.left, row)
        else:
            return self._predict(node.right, row)

    def predict(self, X):
        return X.apply(lambda row: self._predict(self.root, row), axis=1)


data = get_clean_adult_data()
X, y = data['X'], data['y']

# Construct decision tree
tree = DecisionTree(3, 1)

# Train on head
head_X, head_y = X[:10], y[:10]
w = np.full(10, 1/10)
tree.fit(head_X, head_y, w, [8, 9])
print('train 10 records: success')
print(tree.predict(X.iloc[0:10, :]))

# Train on entire dataset
w = np.full(X.shape[0], (1 / X.shape[0]))
tree.fit(X, y, w, [8, 9])
print('train all records: success')
