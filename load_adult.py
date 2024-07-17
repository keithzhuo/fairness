from ucimlrepo import fetch_ucirepo


def get_clean_adult_data():
    # fetch dataset
    adult = fetch_ucirepo(id=2)

    # data (as pandas dataframes)
    X = adult.data.features
    y = adult.data.targets

    # drop na and convert target to binary
    X = X.dropna()
    y = y.loc[X.index]
    y.replace(to_replace={r'<=50K.*': 0, r'>50K.*': 1},
              regex=True, inplace=True)
    y = y.infer_objects(copy=False)
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    return {'X': X, 'y': y}
