from aif360.datasets import AdultDataset
import pandas as pd


class Adult:
    def __init__(self) -> None:
        self.pa = []

    def update_adult_data(self) -> None:
        # by default use one hot encoder to transform each category into a new column
        adult_data = AdultDataset()
        self.pa = adult_data.protected_attribute_names
        X, y = pd.DataFrame(
            adult_data.features, columns=adult_data.feature_names), pd.DataFrame(adult_data.labels, columns=['target'])
        pd.concat([X, y], axis=1).to_csv(
            'data/processed_adult.csv', index=False)

    def load_adult_data(self) -> dict[str, pd.DataFrame]:
        data = pd.read_csv('data/processed_adult.csv')
        return {'X': data.drop(columns=['target']), 'y': data['target']}


adult = Adult()
adult.update_adult_data()
