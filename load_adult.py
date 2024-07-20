from ucimlrepo import fetch_ucirepo


def get_clean_adult_data():
    """
    Returns
    -------
    X
        pd.DataFrame
    y
        np.ndarray
    """
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
    X['race'] = X['race'].astype('category').cat.codes
    X['sex'] = X['sex'].astype('category').cat.codes
    y = y.values.flatten()

    return {'X': X, 'y': y}


data = get_clean_adult_data()
X, y = data['X'], data['y']
head_X, head_y = X[:10], y[:10]
print(head_X, head_y)
