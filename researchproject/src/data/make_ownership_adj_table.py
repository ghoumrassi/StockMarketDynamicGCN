import pandas as pd
import datetime as dt
from tqdm import tqdm

from src.data.utils import create_connection_psql
from src import PG_CREDENTIALS

# TODO: Numbers seem wrong; CHECK AGAIN
# TODO: Maybe metric should be average, not product?

def make_accurate_sharevol():
    engine = create_connection_psql(PG_CREDENTIALS)
    sharevol = pd.read_sql('sharevol', engine)
    tickerdata = pd.read_sql_query(
        '''select ticker, date, tickerdata."splitFactor" from tickerdata where tickerdata."splitFactor" != 1''',
        engine)
    sharevol['date'] = sharevol['date'].apply(get_timestamp)
    sharevol = sharevol.loc[sharevol.groupby("ticker")["date"].idxmin()]
    tqdm.pandas()
    tickerdata = tickerdata.groupby('ticker').progress_apply(lambda x: accurate_sharevol(x, sharevol))
    tickerdata.reset_index(inplace=True, drop=True)
    df = pd.merge(tickerdata, sharevol, on='ticker')
    df['adjShares'] = df['avgShares'] / df['splitFactor']
    df = df[['ticker', 'date_x', 'adjShares']]
    df.rename(columns={'date_x': 'date'}, inplace=True)
    df.to_sql('adj_sharevol', engine, index=False, if_exists='replace')


def make_adj_table():
    engine = create_connection_psql(PG_CREDENTIALS)
    ownership = pd.read_sql('sec_ownership', engine)
    sharevol = pd.read_sql('adj_sharevol', engine)
    sharevol['date'] = sharevol['date']
    ownership = ownership.assign(ticker=ownership['ticker'].str.split(',')).explode('ticker')
    ownership['ticker'] = ownership['ticker'].apply(strip_chars)
    ownership['date'] = ownership['date'].apply(get_timestamp)
    num_companies_per_id = ownership.groupby('owner_cik')['ticker'].nunique()
    ids_over_one_company = num_companies_per_id[num_companies_per_id >= 2].index.tolist()
    ownership = ownership[ownership['owner_cik'].isin(ids_over_one_company)]
    df = pd.merge(ownership, sharevol, on='ticker')
    df['datediff'] = df['date_x'] - df['date_y']
    tqdm.pandas()
    df = df.groupby(['ticker', 'owner_cik', 'date_x', 'num_shares']).progress_apply(closest_preferably_lower)
    df = df.to_frame().reset_index()
    df.rename(columns={0: 'adjShares', 'date_x': 'date'}, inplace=True)
    df['estimatedOwnership'] = df['num_shares'].astype(float) / df['adjShares'].astype(float)
    df.drop(['num_shares', 'adjShares'], 1, inplace=True)
    init_df = df[['ticker', 'owner_cik']].drop_duplicates()
    init_df['date'] = 0
    init_df['estimatedOwnership'] = 0
    df = pd.concat([init_df, df])
    df = df.groupby(['ticker', 'owner_cik']).progress_apply(get_end_date).reset_index()
    adj_df = pd.merge(df, df, on='owner_cik')
    adj_df = adj_df[adj_df['ticker_x'] < adj_df['ticker_y']]
    adj_df['relStart'] = adj_df[['date_x', 'date_y']].max(1)
    adj_df['relEnd'] = adj_df[['endDate_x', 'endDate_y']].min(1)
    adj_df['jointOwnership'] = adj_df['estimatedOwnership_x'] * adj_df['estimatedOwnership_y']
    adj_df = adj_df[['ticker_x', 'ticker_y', 'owner_cik', 'relStart', 'relEnd', 'jointOwnership']]
    adj_df.to_sql('sec_jointownership', engine, index=False, if_exists='replace')


def strip_chars(x):
    for char in " ()":
        x = x.replace(char, "")
    return x


def get_timestamp(x):
    try:
        return dt.datetime.strptime(x, '%Y-%m-%d').timestamp()
    except:
        pass


def closest_preferably_lower(x):
    if (x['datediff'] > 0).sum():
        return x[x['datediff'] == x[x['datediff'] > 0]['datediff'].min()]['adjShares'].iloc[0]
    else:
        return x[x['datediff'] == x['datediff'].min()]['adjShares'].iloc[0]

def get_end_date(x):
    x = x[['date', 'estimatedOwnership']].sort_values('date')
    x['endDate'] = x['date'].shift(-1)
    return x

def accurate_sharevol(x, sharevol):
    try:
        s = sharevol[sharevol['ticker'] == x['ticker'].iloc[0]]
        x_below = x[x['date'] < s['date'].iloc[0]].sort_values('date', ascending=False)
        x_above = x[x['date'] >= s['date'].iloc[0]].sort_values('date', ascending=True)
        x_above['splitFactor'] = 1 / x_above['splitFactor']
        x_below['splitFactor'] = x_below['splitFactor'].cumprod()
        x_above['splitFactor'] = x_above['splitFactor'].cumprod()
        df = pd.concat([x_below, x_above])
    except:
        return None
    return df

def rev_after(x, date):
    if x['date'] < date:
        return x['splitFactor']
    else:
        return 1 / x['splitFactor']


if __name__ == '__main__':
    make_accurate_sharevol()
    make_adj_table()
