import numpy as np
from ucimlrepo import fetch_ucirepo


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
        self.root = None

    # Split the dataset based on label values relative to threashold (left: <=, right: >)
    def _split(self, X, y, feature_index, threshold, is_category):
        # Threshold: to the left holds smaller values and to the right holds greater values
        feature_values = X.iloc[:, feature_index]
        if is_category:
            return X[feature_values == threshold], y[feature_values == threshold], X[feature_values != threshold], y[feature_values != threshold]
        else:
            return X[feature_values <= threshold], y[feature_values <= threshold], X[feature_values > threshold], y[feature_values > threshold]

    # Compute gini index for left and right group
    def _compute_gini_impurity(self, left_y, right_Y, classes):
        n_samples = left_y.shape[0] + right_Y.shape[0]
        gini = 0.0
        for sub_tree in left_y, right_Y:
            sub_tree_size = sub_tree.shape[0]
            if sub_tree_size == 0:
                continue
            score = 0.0
            for label in classes:
                proportion = np.count_nonzero(
                    sub_tree == label) / sub_tree_size
                score += proportion ** 2
            gini += (1.0 - score) * (sub_tree_size / n_samples)
        return gini

    # Find the best split for the dataset in question
    def _find_best_split(self, X, y):
        n_features = X.shape[1]
        classes = np.unique(y)
        best_gini, best_split = float('inf'), None
        is_category = None
        for feature_index in range(n_features):
            feature_values = X.iloc[:, feature_index]
            thresholds = np.unique(feature_values)
            is_category = isinstance(feature_values.iloc[0], str)
            for threshold in thresholds:
                left_X, left_y, right_X, right_y = self._split(
                    X, y, feature_index, threshold, is_category)
                gini = self._compute_gini_impurity(left_y, right_y, classes)
                # Need better tie-breaking method - currently favors features / thresholds that appear first
                if gini < best_gini:
                    best_gini, best_split = gini, {
                        'feature': feature_index, 'threshold': threshold, 'groups': [left_X, left_y, right_X, right_y]}
        return best_split

    # Find the most frequent label
    def _get_most_frequent_label(self, labels):
        return np.bincount(labels.iloc[:, 0]).argmax()

    # Recursively build decision tree
    def _build_tree(self, X, y, depth):
        n_samples = X.shape[0]
        leaf_node = Node(label=self._get_most_frequent_label(y))
        if depth >= self.max_depth or n_samples <= self.min_size:
            return leaf_node
        best_split = self._find_best_split(X, y)
        left_X, left_y, right_X, right_y = best_split['groups']
        if left_X.shape[0] == 0 or right_X.shape[0] == 0:
            return leaf_node
        return Node(self._build_tree(left_X, left_y, depth+1), self._build_tree(right_X, right_y, depth+1), best_split['feature'], best_split['threshold'])

    # Train the model
    def fit(self, X, y):
        self.root = self._build_tree(X, y, 1)

    def _predict(self, node, row):
        if node.label is not None:
            return node.label
        if row.iloc[node.feature_index] <= node.threshold:
            return self._predict(node.left, row)
        else:
            return self._predict(node.right, row)

    def predict(self, X):
        return X.apply(lambda row: self._predict(self.root, row), axis=1)


# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# drop na and convert target to binary
X = X.dropna()
y = y.loc[X.index]
y.replace(to_replace={r'<=50K.*': 0, r'>50K.*': 1}, regex=True, inplace=True)
y = y.infer_objects(copy=False)
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

# Construct decision tree
tree = DecisionTree(3, 1)

# Train on head
head_X, head_y = X[:10], y[:10]
tree.fit(head_X, head_y)
print('train 10 records (without missing data): success')
print(tree.predict(X.iloc[0:5, :]))

# Train on entire dataset
tree.fit(X, y)
print('train all records (without missing data): success')
