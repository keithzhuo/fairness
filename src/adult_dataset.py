import pandas as pd


def load_data(dataset: str) -> dict[str, pd.DataFrame]:
    data = pd.read_csv(f'data/{dataset}_processed.csv')
    pa = pd.read_csv(f'data/{dataset}_pa.csv', header=None)
    return {'X': data.drop(columns=['Probability']), 'y': data['Probability'], 'pa': pa.iloc[0].tolist()}
