import requests
import datetime as dt
import pandas as pd
import time
import json
from tqdm import tqdm
from src import YAHOO_DATA, PG_CREDENTIALS
from src.data.utils import create_connection_psql


def get_market_cap(ticker, start, end):
    fmt = '%d/%m/%Y'
    start_ts = dt.datetime.strptime(start, fmt).timestamp()
    end_ts = dt.datetime.strptime(end, fmt).timestamp()

    url = f'https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{ticker}'
    params = {
        'lang': 'en-gb',
        'region': 'GB',
        'symbol': ticker,
        'padTimeSeries': 'true',
        'type': 'annualBasicAverageShares',
        'merge': 'false',
        'period1': int(start_ts),
        'period2': int(end_ts),
        'corsDomain': 'uk.finance.yahoo.com',
    }
    req = requests.get(url, params=params)
    time.sleep(1)
    assert req.status_code == 200
    with open((YAHOO_DATA / f"{ticker}.json"), 'w') as f:
        json.dump(req.json(), f)


def format_market_cap():
    data = []
    for file in YAHOO_DATA.iterdir():
        with open(file, 'r') as f:
            j = json.loads(f.read())
        ticker = file.stem
        try:
            avg_shares_container = j['timeseries']['result'][0]['annualBasicAverageShares']
        except KeyError:
            continue
        for row in avg_shares_container:
            date = row['asOfDate']
            shares = row['reportedValue']['raw']
            data.append([ticker, date, shares])
    df = pd.DataFrame(data, columns=['ticker', 'date', 'avgShares'])
    engine = create_connection_psql(PG_CREDENTIALS)
    df.to_sql('sharevol', engine, index=False, if_exists='append')


if __name__ == "__main__":
    start = '01/01/2010'
    end = '15/08/2020'
    engine = create_connection_psql(PG_CREDENTIALS)
    resultset = engine.execute("""SELECT DISTINCT ticker FROM tickerdata""")
    results = resultset.fetchall()
    for (ticker,) in tqdm(results):
        get_market_cap(ticker, start, end)
    format_market_cap()