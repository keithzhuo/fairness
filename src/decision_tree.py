import numpy as np
import pandas as pd
from adult_dataset import load_adult_data


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
        self.pa = None
        self.root = None

    # split the dataset based on label values relative to threashold (left: <=, right: >)
    def _split(self, X: pd.DataFrame, feature_index: np.ndarray, threshold: int):
        # Threshold: to the left holds smaller values and to the right holds greater values
        feature_values = X.iloc[:, feature_index]
        return feature_values <= threshold

    # compute gini index for left and right group
    def _compute_gini_impurity(self, val: np.ndarray, w: np.ndarray, mask):
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

    # find the best split for the dataset in question
    def _find_best_split(self, X: pd.DataFrame, y: np.ndarray, w: np.ndarray):
        n_features = X.shape[1]
        best_gini, best_split = float('inf'), None
        for feature_index in range(n_features):
            feature_values = X.iloc[:, feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                mask = self._split(X, feature_index, threshold)
                gini = self._compute_gini_impurity(y, w, mask)
                for pa in self.pa:
                    gini -= self._compute_gini_impurity(
                        X.loc[:, pa].values.flatten().astype(np.int64), w, mask)
                if gini < best_gini:
                    best_gini, best_split = gini, {
                        'feature': feature_index, 'threshold': threshold, 'mask': mask}
        return best_split

    # find the most frequent label
    def _get_most_frequent_label(self, labels: np.ndarray):
        return np.bincount(labels).argmax()

    # recursively build decision tree
    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, w: np.ndarray, depth: int):
        n_samples = X.shape[0]
        leaf_node = Node(label=self._get_most_frequent_label(y))
        if depth >= self.max_depth or n_samples <= self.min_size:
            return leaf_node
        best_split = self._find_best_split(X, y, w)
        mask = best_split['mask']
        if np.size(y[mask]) == 0 or np.size(y[~mask]) == 0:
            return leaf_node
        return Node(self._build_tree(X[mask], y[mask], w[mask], depth+1), self._build_tree(X[~mask], y[~mask], w[~mask], depth+1), best_split['feature'], best_split['threshold'])

    # train the model
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, w: np.ndarray, pa):
        y = y.astype(np.int64)
        self.pa = pa
        self.root = self._build_tree(X, y, w, 0)

    def _predict(self, node: Node, row: pd.DataFrame):
        if node.label is not None:
            return node.label
        if row.iloc[node.feature_index] <= node.threshold:
            return self._predict(node.left, row)
        else:
            return self._predict(node.right, row)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X.apply(lambda row: self._predict(self.root, row), axis=1)


data = load_adult_data()
pa = data['pa']
X, y = data['X'], data['y']

# construct decision tree
tree = DecisionTree(3, 1)

# train on head
head_X, head_y = X[:10], y[:10]
w = np.full(10, 1/10)
tree.fit(head_X, head_y, w, pa)
print('train 10 records for dt: success')
print(tree.predict(X.iloc[0:10, :]))

# train on entire dataset
w = np.full(X.shape[0], (1 / X.shape[0]))
tree.fit(X, y, w, pa)
print('train all records for dt: success')
