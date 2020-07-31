import json
import requests
import pandas as pd
from io import StringIO
from tqdm import tqdm
import datetime as dt
import time

from researchproject.src import TIINGO_KEY, TIINGO_DATA, NASDAQ_TICKERS, COMPANY_DATA


class TiingoData:
    """Gets data on historic stock prices from Tiingo and exports them to csv."""
    def __init__(self):
        with open(TIINGO_KEY, 'r') as fn:
            api_key = json.load(fn)['API_KEY']
        self.headers = {'Content-Type': 'application/json', 'Authorization': f'Token {api_key}'}

    def get(self, tickers, exchange, start_date="2000-01-01", end_date="today"):
        if end_date == 'today':
            end_date = dt.datetime.now().strftime("%Y-%m-%d")
        params = {
            'startDate': start_date,
            'endDate':   end_date,
            'format':    'csv'
        }
        data_dir = TIINGO_DATA / exchange
        data_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(tickers)
        for ticker in pbar:
            pbar.set_description(f"Current ticker: {ticker}")
            data_file = data_dir / f"{ticker}.csv"
            if data_file.exists():
                cutoff_date = pd.to_datetime(pd.read_csv(data_file)['date'], format="%Y-%m-%d").max()
            else:
                cutoff_date = pd.Timestamp(1990, 1, 1, 12)
            if cutoff_date > dt.datetime.strptime(start_date, '%Y-%m-%d'):
                start_date = (cutoff_date + pd.DateOffset(days=1)).strftime('%Y-%m-%d')

            response = requests.get(
                f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
                params=params,
                headers=self.headers)

            # TODO: Error handling here could definitely be better....
            if response.status_code != 200:
                print(f"There was an error for ticker {ticker}. Status code: {response.status_code}")
                continue
            if len(response.text) <= 2:
                print(f"There was no data for the ticker {ticker}.")
                continue
            if response.text[:5] == "Error":
                print(f"There was an error for the ticker {ticker}.")
                continue

            df = pd.read_csv(StringIO(response.text))
            df['ticker'] = ticker
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            df = df[df['date'] > cutoff_date]

            with open(data_file, 'a') as f:
                df.to_csv(f, header=f.tell() == 0, index=False)
            time.sleep(1)


if __name__ == "__main__":
    # ticker_file = pd.read_csv(NASDAQ_TICKERS, sep='|')
    # tickers = ticker_file['Symbol'].values
    ticker_file = pd.read_csv(COMPANY_DATA)
    tickers = ticker_file['Symbol'].values
    tiingo = TiingoData()
    tiingo.get(tickers, "nasdaq", '2000-01-01', 'today')
