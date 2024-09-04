import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_d2h(X: pd.DataFrame, y_optimal: pd.DataFrame, predictions: np.ndarray, pa: list[str]):
    # use list to pass on the reference rather than a copy
    d2h: list[float] = [0.0]

    # be careful: X_test preserved original indices from X
    pred_df = pd.concat([X.reset_index(drop=True),
                        pd.Series(predictions, name='labels').reset_index(drop=True)], axis=1)

    pred_data = BinaryLabelDataset(
        df=pred_df,
        label_names=['labels'],
        protected_attribute_names=pa
    )

    # create BinaryLabelDataset for true data
    true_df = pd.concat([X.reset_index(drop=True),
                        y_optimal.reset_index(drop=True)], axis=1)
    true_data = BinaryLabelDataset(
        df=true_df,
        label_names=['labels'],
        protected_attribute_names=pa
    )

    # create BinaryLabelDatasetMetric for evaluation
    binary_label_metric = BinaryLabelDatasetMetric(pred_data, privileged_groups=[
        {pa[0]: 1, pa[1]: 1}], unprivileged_groups=[{pa[0]: 0, pa[1]: 0}])

    # create ClassificationMetric instance
    classification_metric = ClassificationMetric(true_data, pred_data, privileged_groups=[
        {pa[0]: 1, pa[1]: 1}], unprivileged_groups=[{pa[0]: 0, pa[1]: 0}])

    metrics: dict[str, tuple[float, float]] = {
        # fairness metrics
        'Statistical Parity Difference:': (0, binary_label_metric.statistical_parity_difference()),
        'Disparate Impact:': (1, binary_label_metric.disparate_impact()),
        'Average Odds Difference:': (0, classification_metric.average_odds_difference()),
        'Equal Opportunity Difference:':  (0, classification_metric.equal_opportunity_difference()),
        # performance metrics
        'Accuracy Score:': (1, accuracy_score(y_optimal, predictions)),
        'Precision Score:': (1, precision_score(y_optimal, predictions)),
        'Recall Score:': (1, recall_score(y_optimal, predictions)),
        'F1 Score:': (1, f1_score(y_optimal, predictions))
    }
    print_and_all_all(d2h, metrics)
    return d2h[0]


def print_and_all_all(d2h: list[float], metrics: dict[str, tuple[float, float]]):
    for message, (optimum, metric) in metrics.items():
        print_and_add(d2h, message, optimum, metric)


def print_and_add(d2h: list[float], message: str, optimum: float, metric: float):
    d2h[0] += (optimum-metric) ** 2
    print(message, metric)
