import json
import requests
import pandas as pd
from io import StringIO

import datetime as dt

from researchproject.src import TIINGO_KEY, TIINGO_DATA


class TiingoData:
    """Gets data on historic stock prices from Tiingo and exports them to csv."""
    def __init__(self):
        with open(TIINGO_KEY, 'r') as fn:
            api_key = json.load(fn)['API_KEY']
        self.headers = {'Content-Type': 'application/json', 'Authorization' : f'Token {api_key}'}

    def get(self, tickers, start_date, end_date):
        params = {
            'startDate': start_date,
            'endDate':   end_date,
            'format':    'csv'
        }

        for ticker in tickers:
            data_file = TIINGO_DATA / f"{ticker}.csv"
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

            if response.status_code != 200:
                #TODO: Maybe a more informational error message?
                print("ERROR MOTHERFUCKER!")
                continue

            df = pd.read_csv(StringIO(response.text))
            df['ticker'] = ticker
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            df = df[df['date'] > cutoff_date]

            with open(data_file, 'a') as f:
                df.to_csv(f, header=f.tell()==0)

if __name__ == "__main__":
    tiingo = TiingoData()
    tiingo.get(['AAPL'], '2000-01-01', '2020-04-30')