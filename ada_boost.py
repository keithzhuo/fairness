import numpy as np
from decision_tree import DecisionTree
from load_adult import get_clean_adult_data


class AdaBoost:
    def __init__(self, n_clf=5, base_estimator=None):
        self.n_clf = n_clf
        self.base_estimator = base_estimator
        self.clfs = []

    def fit(self, X, y, pa_indices):
        n_samples = X.shape[0]
        # Initialize equal weights
        w = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_clf):
            clf = self.base_estimator(max_depth=1)
            clf.fit(X, y, w, pa_indices)

            # Make predictions and compute error
            predictions = clf.predict(X)
            error = np.sum(w * (predictions != y)) / np.sum(w)

            # Calculate alpha - amount of say for each stump
            alpha = 0.5 * np.log((1.0 - error) / (error + 1e-10))
            clf.alpha = alpha

            # Update weights - e^(+/-amount of say) > 1, increases/decreases the sample weight
            w *= np.exp(-clf.alpha * y * clf.predict(X))
            w /= np.sum(w)

            # Save the classifier
            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sign(np.sum(clf_preds, axis=0))
        return y_pred


data = get_clean_adult_data()
X, y = data['X'], data['y']

# Create a adaboost
clf = AdaBoost(n_clf=5, base_estimator=DecisionTree)

# Train on head
head_X, head_y = X[:10], y[:10]
clf.fit(head_X, head_y, [8, 9])
print('train 10 records: success')
print(clf.predict(X.iloc[0:10, :]))

# Train on entire dataset
clf.fit(X, y)
print('train all records: success')
