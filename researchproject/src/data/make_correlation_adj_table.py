import pandas as pd
from tqdm import tqdm

from src.data.utils import create_connection_psql
from src import PG_CREDENTIALS


def make_corr(window=100):
    seconds = window * 86400
    engine = create_connection_psql(PG_CREDENTIALS)
    df = pd.read_sql_query('SELECT date, ticker, returns FROM tickerdata', engine)
    df = df.pivot(index='date', columns='ticker')
    for date in tqdm(df.index.unique()):
        corr = df[(df.index <= date) & (df.index > date - seconds)].corr()
        corr = corr.stack()
        corr.dropna(inplace=True)
        if corr.empty:
            continue
        corr.index.rename(['drop', 'a', 'b'], inplace=True)
        corr.reset_index(inplace=True)
        corr.drop('drop', 1, inplace=True)
        corr = corr[corr['a'] < corr['b']]
        corr['date'] = date
        corr.to_sql('returns_correlations', engine, if_exists='append', index=False)


if __name__ == "__main__":
    make_corr()