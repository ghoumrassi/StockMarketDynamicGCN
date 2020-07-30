from src import WD_OUTPUT, NYT_OUTPUT, TIINGO_DATA, FM_OUTPUT, SQLITE_DB
from src.data.utils import create_connection
import pandas as pd
import datetime as dt
import pickle
import numpy as np
import tqdm


def populate_subsidiaries(conn):
    subsidiaries_file = WD_OUTPUT / "subsidiaries.csv"
    df = pd.read_csv(subsidiaries_file)
    df = df[['altLabel', 'subsidiaryLabel', 'parentLabel', 'ticker', 'exchangeLabel']].drop_duplicates()
    df.to_sql('subsidiaries', conn, if_exists='replace')


def populate_industry(conn):
    subsidiaries_file = WD_OUTPUT / "subsidiaries.csv"
    df = pd.read_csv(subsidiaries_file)
    df = df[['subsidiaryLabel', 'parentLabel', 'industryLabel']].drop_duplicates()
    df.to_sql('industry', conn, if_exists='replace')


def populate_dates(conn):
    subsidiaries_file = WD_OUTPUT / "subsidiaries.csv"
    df = pd.read_csv(subsidiaries_file)
    df = df[['subsidiaryLabel', 'parentLabel', 'startdateLabel', 'enddateLabel']].drop_duplicates()
    df.to_sql('dates', conn, if_exists='replace')


def populate_summaries(conn):
    summaries_file = NYT_OUTPUT / "summaries.csv"
    df = pd.read_csv(summaries_file)
    df['Date'] = df['Date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").timestamp())
    df.to_sql('summaries', conn, if_exists='replace')


def populate_mapper(conn):
    with open((FM_OUTPUT / 'nasdaq_mapper.p'), 'rb') as f:
        data = pickle.load(f)
        df = pd.Series(data).to_frame(name='company').reset_index()
        df.to_sql('companymapper', conn, if_exists='replace')


def populate_ticker(conn):
    num_files = len([f for f in (TIINGO_DATA / 'nasdaq').iterdir()])
    pbar = tqdm.tqdm((TIINGO_DATA / 'nasdaq').iterdir(), total=num_files)
    for i, ticker_file in enumerate(pbar):
        df = pd.read_csv(ticker_file)
        df = df[df['date'].notna()]
        df['date'] = df['date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").timestamp())
        df['returns'] = np.log(df['adjClose']) - np.log(df['adjClose'].shift(1))
        df = df[df['returns'].notna()]
        if i == 0:
            df.to_sql('tickerdata', conn, if_exists='replace')
        else:
            df.to_sql('tickerdata', conn, if_exists='append')
        pbar.update()


if __name__ == "__main__":
    try:
        conn = create_connection(SQLITE_DB)
        populate_summaries(conn)
        populate_industry(conn)
        populate_dates(conn)
        populate_mapper(conn)
        populate_subsidiaries(conn)
        populate_ticker(conn)
    finally:
        conn.close()
