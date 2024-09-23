import pandas as pd
from sklearn.model_selection import train_test_split
from ada_boost import AdaBoost
from adult_dataset import load_data
from util import *


def run_exp_ada_boost(X: pd.DataFrame, y: pd.DataFrame, pa: list[str], seed=42):
    # split data: 70% train, 10% validation, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed)
    X_validate, X_test, y_validate, y_test = train_test_split(
        X_test, y_test, test_size=0.67, random_state=seed
    )

    param_grid: dict[str, float] = {
        'max_depth': [2, 3, 4],
        'ratio': [0.2, 2]
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
            d2h = get_d2h(X_validate, y_validate, predictions, pa)
            if d2h < best_params['d2h']:
                best_params = {
                    'max_depth': max_depth,
                    'ratio': ratio,
                    'd2h': d2h
                }

    clf.fit(X_train, y_train, pa,
            best_params['max_depth'], best_params['ratio'])
    predictions = clf.predict(X_test)
    d2h_test = get_d2h(X_test, y_test, predictions, pa, True)
    for attr in pa:
        print(f'flip rate for {attr}: ', flip_rate(clf, X, attr))

    return {
        'best_params': best_params,
        'd2h_test': d2h_test,
    }


def run_exp_ada_boost_n_times_on_dataset(n: int, dataset: str):
    print(f'********* {dataset} *********')
    data = load_data(dataset)
    X, y = data['X'], data['y']
    pa = data['pa']
    for i in range(len(pa)):
        best_ratios, best_max_depth, best_d2h = [], [], []
        single_pa = [pa[i]]
        for seed in range(n):
            res = run_exp_ada_boost(X, y, single_pa, seed)
            best_ratios.append(res['best_params']['ratio'])
            best_max_depth.append(res['best_params']['max_depth'])
            best_d2h.append(res['d2h_test'])
        print('bests:', best_ratios, best_max_depth, best_d2h)


def flip_rate(clf: AdaBoost, X: pd.DataFrame, attr: str):
    X_flip = X.copy()
    X_flip[attr] = np.where(X_flip[attr] == 1, 0, 1)
    a = np.array(clf.predict(X))
    b = np.array(clf.predict(X_flip))
    total = X.shape[0]
    same = np.count_nonzero(a == b)
    return (total-same)/total


def main():
    datasets = ['adult', 'bank', 'compas', 'german', 'h181', 'heart']
    for dataset in datasets:
        run_exp_ada_boost_n_times_on_dataset(10, dataset)


if __name__ == "__main__":
    main()
