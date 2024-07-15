import numpy as np
from ucimlrepo import fetch_ucirepo
from decision_tree import DecisionTree


class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        feature_values = X.iloc[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values >= self.threshold] = -1
        return predictions


class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Convert y to numpy array
        y = y.values.flatten()
        # Initialize equal weights
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            # Train a decision stump
            for feature_i in range(n_features):
                feature_values = X.iloc[:, feature_i]
                thresholds = np.unique(feature_values)
                for threshold in thresholds:
                    # Predict with polarity 1
                    p = 1
                    predictions = np.ones(n_samples)
                    if isinstance(feature_values.iloc[0], str):
                        predictions[feature_values != threshold] = -1
                    else:
                        predictions[feature_values < threshold] = -1
                    # Calculate misclassification error
                    error = sum(w[y != predictions])

                    # If error is greater than 0.5, flip polarity
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # Store the best stump
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error

            # Calculate alpha - amount of say for each stump
            clf.alpha = 0.5 * np.log((1.0 - min_error) / (min_error + 1e-10))

            # Update weights
            predictions = clf.predict(X)
            # e^(amount of say) > 1, increases the sample weight; e^(-amount of say) decreases
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            # Save the classifier
            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sign(np.sum(clf_preds, axis=0))
        return y_pred


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

# Create a adaboost
clf = AdaBoost(n_clf=5)

# Train on head
head_X, head_y = X[:10], y[:10]
clf.fit(head_X, head_y)
print('train 10 records (without missing data): success')
print(clf.predict(X.iloc[0:5, :]))

# Train on entire dataset
clf.fit(X, y)
print('train all records (without missing data): success')
