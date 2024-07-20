import numpy as np
import pandas as pd
from decision_tree import DecisionTree


class AdaBoost:
    def __init__(self, n_clf=5, base_estimator=DecisionTree):
        self.n_clf = n_clf
        self.base_estimator = base_estimator
        self.clfs = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, pa: list):
        y = y.astype(np.int64)
        n_samples = X.shape[0]
        # initialize equal weights
        w = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_clf):
            clf = self.base_estimator(max_depth=1)
            clf.fit(X, y, w, pa)

            # make predictions and compute error
            predictions = clf.predict(X)
            error = np.sum(w * (predictions != y)) / np.sum(w)

            # calculate alpha - amount of say for each stump
            alpha = 0.5 * np.log((1.0 - error) / (error + 1e-10))
            clf.alpha = alpha

            # update weights - e^(+/-amount of say) > 1, increases/decreases the sample weight
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            # save the classifier
            self.clfs.append(clf)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sign(np.sum(clf_preds, axis=0))
        return y_pred
