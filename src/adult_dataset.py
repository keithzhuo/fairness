from aif360.datasets import AdultDataset
import pandas as pd


def update_adult_data() -> None:
    # by default use one hot encoder to transform each category into a new column
    adult_data = AdultDataset()
    X, y = pd.DataFrame(
        adult_data.features, columns=adult_data.feature_names), pd.DataFrame(adult_data.labels, columns=['labels'])
    pd.DataFrame([adult_data.protected_attribute_names]).to_csv(
        'data/adult_pa.csv', index=False, header=False)
    pd.concat([X, y], axis=1).to_csv(
        'data/processed_adult.csv', index=False)


def load_adult_data() -> dict[str, pd.DataFrame]:
    data = pd.read_csv('data/processed_adult.csv')
    pa = pd.read_csv('data/adult_pa.csv', header=None)
    return {'X': data.drop(columns=['labels']), 'y': data['labels'], 'pa': pa.iloc[0].tolist()}


update_adult_data()
