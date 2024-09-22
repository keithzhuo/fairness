import pandas as pd
import numpy as np
from math import *
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_d2h(X: pd.DataFrame, y_optimal: pd.DataFrame, predictions: np.ndarray, pa: list[str], should_print: bool = False):
    metrics = get_metrics(X, y_optimal, predictions, pa)
    return calculate_d2h_from_metrics(metrics, should_print)


def get_metrics(X: pd.DataFrame, y_optimal: pd.DataFrame, predictions: np.ndarray, pa: list[str]):
    # be careful: X_test preserved original indices from X
    pred_df = pd.concat([X.reset_index(drop=True),
                        pd.Series(predictions, name='Probability').reset_index(drop=True)], axis=1)

    pred_data = BinaryLabelDataset(
        df=pred_df,
        label_names=['Probability'],
        protected_attribute_names=pa
    )

    # create BinaryLabelDataset for true data
    true_df = pd.concat([X.reset_index(drop=True),
                        y_optimal.reset_index(drop=True)], axis=1)
    true_data = BinaryLabelDataset(
        df=true_df,
        label_names=['Probability'],
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
    return metrics


def calculate_d2h_from_metrics(metrics: dict[str, tuple[float, float]], should_print: bool):
    d2h: float = 0.0
    for message, (optimum, metric) in metrics.items():
        if should_print:
            print(message, metric)
        d2h += (optimum-metric) ** 2
    return sqrt(d2h)
