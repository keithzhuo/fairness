import numpy as np
import pandas as pd
from decision_tree import DecisionTree


class AdaBoost:
    def __init__(self, n_clf=5, base_estimator=DecisionTree):
        self.n_clf = n_clf
        self.base_estimator = base_estimator
        self.clfs = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, pa: list, max_depth: int = 3, ratio: int = 1):
        y = y.astype(np.int64)
        n_samples = X.shape[0]
        # initialize equal weights
        w = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_clf):
            clf = self.base_estimator(max_depth)
            clf.fit(X, y, w, pa)

            # make predictions, save incorrect rows, and compute error
            predictions = clf.predict(X)
            incorrect = (predictions != y)
            error = np.sum(w * incorrect) / np.sum(w)

            # calculate alpha - amount of say for each stump
            alpha = 0.5 * np.log((1.0 - error) / (error + 1e-10))
            clf.alpha = alpha

            # update weights - e^(+/-amount of say) increases/decreases the sample weight
            # (2*y-1) * (2*predictions-1) == 1 iff predicition matches true value
            w *= np.exp(-clf.alpha * (2*y-1) * (2*predictions-1))

            # adjust weights for fairness
            for attr in pa:
                privileged = X[attr] == 1
                w *= np.where((incorrect & ~privileged & (y == 1)) |
                              (incorrect & privileged & (y == 0)), ratio, 1.0)

            # normalize weights
            w /= np.sum(w)

            # save the classifier
            self.clfs.append(clf)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        clf_preds = np.zeros(X.shape[0])
        for clf in self.clfs:
            predictions = clf.predict(X)
            clf_preds += clf.alpha * (2*predictions-1)
        y_pred = (clf_preds >= 0).astype(int)
        return y_pred
