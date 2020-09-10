import pandas as pd
from tqdm import tqdm
from sqlalchemy import text
from src.data.utils import create_connection_psql
from src import PG_CREDENTIALS, QUERIES


def make_avg(ticker, engine):
    df = pd.read_sql_query(f"SELECT * FROM tickerdata WHERE ticker = '{ticker}'", engine)
    df['5-day'] = df['adjClose'].rolling(5).mean()
    df['10-day'] = df['adjClose'].rolling(10).mean()
    df['20-day'] = df['adjClose'].rolling(20).mean()
    df['30-day'] = df['adjClose'].rolling(30).mean()
    df.to_sql('tickerdata_withavg', engine, if_exists='append', index=False)

def run():
    engine = create_connection_psql(PG_CREDENTIALS)
    with open((QUERIES / 'psql' / 'get_distinct_tickers.q'), 'r') as f:
        q = f.read()
    resultset = engine.execute(text(q), startdate=0, enddate=999999999999)
    results = resultset.fetchall()
    tickers = [result[0] for result in results]
    for ticker in tqdm(tickers):
        make_avg(ticker, engine)


if __name__ == "__main__":
    run()