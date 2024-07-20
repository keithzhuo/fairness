import numpy as np
import pandas as pd
from decision_tree import DecisionTree
from adult_dataset import load_adult_data
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric


class AdaBoost:
    def __init__(self, n_clf=5, base_estimator=DecisionTree):
        self.n_clf = n_clf
        self.base_estimator = base_estimator
        self.clfs = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, pa):
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


data = load_adult_data()
pa = data['pa']
X, y = data['X'], data['y']

# create a adaboost
clf = AdaBoost(n_clf=5)

# train on head
head_X, head_y = X[:10], y[:10]
clf.fit(head_X, head_y, pa)
print('train 10 records for adaboost: success')
print(clf.predict(X.iloc[0:10, :]))

# train on entire dataset
train_X, train_y = X[:30000], y[:30000]
clf.fit(train_X, train_y, pa)
print('train all records for adaboost: success')

test_X = X[30000:].reset_index(drop=True)
predictions = clf.predict(test_X)
pred_df = pd.DataFrame(predictions, columns=['target'])
df = pd.concat([test_X, pred_df], axis=1)
pred_data = BinaryLabelDataset(
    df=df,
    label_names=['target'],
    protected_attribute_names=pa
)

# Create BinaryLabelDatasetMetric for evaluation
metric = BinaryLabelDatasetMetric(pred_data, privileged_groups=[
                                  {pa[0]: 1}], unprivileged_groups=[{pa[0]: 0}])

# Print fairness metrics
print('Statistical Parity Difference:', metric.statistical_parity_difference())
print('Disparate Impact:', metric.disparate_impact())
print('Average Odds Difference:', metric.mean_difference())
print('Equal Opportunity Difference:',
      metric.smoothed_empirical_differential_fairness())
