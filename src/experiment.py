import pandas as pd
from sklearn.model_selection import train_test_split
from ada_boost import AdaBoost
from adult_dataset import load_adult_data
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from util import calculate_d2h

# just for control group
from decision_tree import DecisionTree
import numpy as np


def run_exp_ada_boost(X: pd.DataFrame, y: pd.DataFrame, pa: list[str], seed=42):
    # split data: 70% train, 10% validation, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed)
    X_validate, X_test, y_validate, y_test = train_test_split(
        X_test, y_test, test_size=0.67, random_state=seed
    )

    param_grid: dict[str, float] = {
        # 'max_depth': [2, 3, 4, 5, 6],
        'max_depth': [3],
        # 'ratio': [0.1, 1, 10]
        'ratio': [2]
    }
    best_params: dict[str, float] = {
        'max_depth': None,
        'ratio': None,
        'd2h': float('inf')
    }
    for ratio in param_grid['ratio']:
        for max_depth in param_grid['max_depth']:
            # create a adaboost
            clf = AdaBoost(n_clf=5)

            # train on the training dataset
            clf.fit(X_train, y_train, pa, max_depth, ratio)

            # make prediction on the validation dataset
            predictions = clf.predict(X_validate)
            d2h = calculate_d2h(X_validate, y_validate, predictions, pa)
            if d2h < best_params['d2h']:
                best_params = {
                    'max_depth': max_depth,
                    'ratio': ratio,
                    'd2h': d2h
                }

    ratio_unit = best_params['ratio'] / 10
    for i in range(1, 11):
        ratio = best_params['ratio'] + ratio_unit * i
        # create a adaboost
        clf = AdaBoost(n_clf=5)

        # train on the training dataset
        clf.fit(X_train, y_train, pa, best_params['max_depth'], ratio)

        # make prediction on the validation dataset
        predictions = clf.predict(X_validate)
        d2h = calculate_d2h(X_validate, y_validate, predictions, pa)
        if d2h < best_params['d2h']:
            best_params['d2h'] = d2h
            best_params['ratio'] = ratio

    print('********* best params *********')
    print(best_params)

    clf.fit(X_train, y_train, pa,
            best_params['max_depth'], best_params['ratio'])
    print('train all records for adaboost: success')
    predictions = clf.predict(X_test)
    d2h = calculate_d2h(X_test, y_test, predictions, pa)


data = load_adult_data()
X, y = data['X'], data['y']
pa = data['pa']
# use seed from 0 to 9 here to ensure robustness? check best params or d2h?
run_exp_ada_boost(X, y, pa)
