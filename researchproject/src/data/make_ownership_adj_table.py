import pandas as pd
import datetime as dt
from tqdm import tqdm

from src.data.utils import create_connection_psql
from src import PG_CREDENTIALS


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
    df['adjShares'] = df['avgShares'] / df['cumulativeSplitFactor']
    df = df[['ticker', 'date_x', 'adjShares', 'splitFactor', 'cumulativeSplitFactor']]
    df.rename(columns={'date_x': 'date'}, inplace=True)
    df.to_sql('adj_sharevol', engine, index=False, if_exists='replace')


def make_adj_table():
    engine = create_connection_psql(PG_CREDENTIALS)
    ownership = pd.read_sql('sec_ownership', engine)
    sharevol = pd.read_sql('adj_sharevol', engine)

    ownership = ownership.assign(ticker=ownership['ticker'].str.split(',')).explode('ticker')
    ownership['ticker'] = ownership['ticker'].apply(strip_chars)
    num_companies_per_id = ownership.groupby('owner_cik')['ticker'].nunique()
    ids_over_one_company = num_companies_per_id[num_companies_per_id >= 2].index.tolist()
    ownership = ownership[ownership['owner_cik'].isin(ids_over_one_company)]
    ownership['date'] = ownership['date'].apply(get_timestamp)
    df = pd.merge(ownership, sharevol, on='ticker')
    df['datediff'] = df['date_x'] - df['date_y']
    tqdm.pandas()

    # Get all observations where a split has just occured (i.e. the share values are correct)
    df_high = df.groupby(['ticker', 'owner_cik', 'date_x', 'num_shares']).progress_apply(closest_higher).reset_index(drop=True)

    # Get all observations with split that is about to happen (share amounts need to be increased on split date)
    df_low = df.groupby(['ticker', 'owner_cik', 'date_x', 'num_shares']).progress_apply(closest_lower).reset_index(drop=True)
    # Compute new share amounts after split
    df_low = df_low.groupby(['ticker', 'owner_cik', 'date_y']).progress_apply(get_value_after_split).reset_index(drop=True)

    df = pd.concat([df_low, df_high])
    df['estimatedOwnership'] = df['num_shares'].astype(float) / df['adjShares'].astype(float)
    df.rename(columns={'date_x': 'date'}, inplace=True)
    df.drop(['num_shares', 'adjShares', 'date_y', 'splitFactor', 'cumulativeSplitFactor', 'datediff'], 1, inplace=True)
    init_df = df[['ticker', 'owner_cik']].drop_duplicates()
    init_df['date'] = 0
    init_df['estimatedOwnership'] = 0
    df = pd.concat([init_df, df])
    df = df.groupby(['ticker', 'owner_cik']).progress_apply(get_end_date).reset_index()
    adj_df = pd.merge(df, df, on='owner_cik')
    adj_df = adj_df[adj_df['ticker_x'] < adj_df['ticker_y']]
    adj_df['relStart'] = adj_df[['date_x', 'date_y']].max(1)
    adj_df['relEnd'] = adj_df[['endDate_x', 'endDate_y']].min(1)
    adj_df = adj_df[adj_df['relStart'] < adj_df['relEnd']]
    adj_df['jointOwnership'] = (adj_df['estimatedOwnership_x'] + adj_df['estimatedOwnership_y']) / 2
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


def closest_higher(x):
    return x[x['datediff'] == x[x['datediff'] > 0]['datediff'].min()].head(1)
def closest_lower(x):
    return x[x['datediff'] == x[x['datediff'] <= 0]['datediff'].min()].head(1)

def get_end_date(x):
    x = x[['date', 'estimatedOwnership']].sort_values('date')
    x['endDate'] = x['date'].shift(-1)
    return x

def accurate_sharevol(x, sharevol):
    try:
        s = sharevol[sharevol['ticker'] == x['ticker'].iloc[0]]
        x_below = x[x['date'] < s['date'].iloc[0]].sort_values('date', ascending=False)
        x_above = x[x['date'] >= s['date'].iloc[0]].sort_values('date', ascending=True)
        x_above['cumulativeSplitFactor'] = 1 / x_above['splitFactor']
        x_below['cumulativeSplitFactor'] = x_below['splitFactor']

        x_above['cumulativeSplitFactor'] = x_above['cumulativeSplitFactor'].cumprod()
        x_below['cumulativeSplitFactor'] = x_below['cumulativeSplitFactor'].cumprod()
        df = pd.concat([x_below, x_above])
    except:
        return None
    return df

def get_value_after_split(x):
    x = x.sort_values('date_x', ascending=False).head(1)
    x['date_x'] = x['date_y']
    x['num_shares'] = x['num_shares'].astype(float) * x['splitFactor']
    return x

def rev_after(x, date):
    if x['date'] < date:
        return x['splitFactor']
    else:
        return 1 / x['splitFactor']


if __name__ == '__main__':
    make_accurate_sharevol()
    make_adj_table()
