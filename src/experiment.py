import pandas as pd
from sklearn.model_selection import train_test_split
from ada_boost import AdaBoost
from adult_dataset import load_adult_data
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_exp(X: pd.DataFrame, y: pd.DataFrame, pa: list, seed=42):
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed)

    # create a adaboost
    clf = AdaBoost(n_clf=5)

    # train on entire dataset
    clf.fit(X_train, y_train, pa)
    print('train all records for adaboost: success')

    predictions = clf.predict(X_test)
    # be careful: X_test preserved original indices from X
    pred_df = pd.concat([X_test.reset_index(drop=True),
                        pd.Series(predictions, name='labels')], axis=1)
    pred_data = BinaryLabelDataset(
        df=pred_df,
        label_names=['labels'],
        protected_attribute_names=pa
    )

    # create BinaryLabelDataset for true data
    true_df = pd.concat([X_test.reset_index(drop=True),
                        y_test.reset_index(drop=True)], axis=1)
    true_data = BinaryLabelDataset(
        df=true_df,
        label_names=['labels'],
        protected_attribute_names=pa
    )

    # create BinaryLabelDatasetMetric for evaluation
    binary_label_metric = BinaryLabelDatasetMetric(pred_data, privileged_groups=[
        {pa[0]: 1}], unprivileged_groups=[{pa[0]: 0}])

    # create ClassificationMetric instance
    classification_metric = ClassificationMetric(true_data, pred_data, privileged_groups=[
        {pa[0]: 1}], unprivileged_groups=[{pa[0]: 0}])

    # print fairness metrics
    print('Statistical Parity Difference:',
          binary_label_metric.statistical_parity_difference())
    print('Disparate Impact:', binary_label_metric.disparate_impact())
    print('Average Odds Difference:',
          classification_metric.average_odds_difference())
    print('Equal Opportunity Difference:',
          classification_metric.equal_opportunity_difference())

    # print performance metrics
    print('Accuracy Score:', accuracy_score(y_test, predictions))
    print('Precision Score:', precision_score(y_test, predictions))
    print('Recall Score:', recall_score(y_test, predictions))
    print('F1 Score:', f1_score(y_test, predictions))


data = load_adult_data()
X, y = data['X'], data['y']
pa = data['pa']
run_exp(X, y, pa)
