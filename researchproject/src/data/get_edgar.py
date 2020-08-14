import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from tqdm import tqdm
import time
import pickle

from src.data.utils import create_connection_psql
from src import PG_CREDENTIALS, QUERIES, EDG_SAVE


def make_ticker_cik_table(conn):
    url = 'https://www.sec.gov/files/company_tickers.json'
    while True:
        try:
            req = requests.get(url)
            time.sleep(2)
            break
        except requests.exceptions.ConnectionError:
            time.sleep(10)
    data = req.json()
    df = pd.DataFrame(data).T
    df.to_sql('cikmapper', conn, if_exists='replace', index=False)


def make_cik_list(conn):
    with open((QUERIES / 'psql' / 'make_cik_list.q'), 'r') as f:
        q = f.read()
    resultsset = conn.execute(q)
    return [str(item[0]) for item in resultsset.fetchall()]

def get_daily_indices(year, qtr):
    base_url = 'https://www.sec.gov/Archives/edgar/daily-index/'
    url = base_url + str(year) + '/QTR' + str(qtr) + '/'
    while True:
        try:
            index = requests.get(url)
            time.sleep(2)
            break
        except requests.exceptions.ConnectionError:
            time.sleep(10)
    soup = BeautifulSoup(index.content, 'lxml')
    main_container = soup.find('div', {'id': 'main-content'})
    links = main_container.find_all('a')
    form_links = [url + link.text for link in links if re.search('^form', link.text)]

    return form_links


def get_form_4(index, cik_list):
    base_url = 'https://www.sec.gov/Archives/'
    while True:
        try:
            req = requests.get(index)
            time.sleep(2)
            break
        except requests.exceptions.ConnectionError:
            time.sleep(10)

    docs = []
    for line in req.text.split('\n'):
        line = line.strip().split('  ')
        line = [col for col in line if col]
        if len(line) >= 3:
            if line[0].strip() == '4':
                if line[2].strip() in cik_list:
                    docs.append(base_url + line[-1].strip())
    return docs


def parse_form_4(form, conn):
    while True:
        try:
            req = requests.get(form)
            time.sleep(2)
            break
        except requests.exceptions.ConnectionError:
            time.sleep(10)

    soup = BeautifulSoup(req.content, 'lxml')
    ticker_container = soup.find('issuertradingsymbol')
    if not ticker_container:
        time.sleep(2)
        return None
    ticker = ticker_container.text.strip()

    owner_cik_container = soup.find('rptownercik')
    if not owner_cik_container:
        time.sleep(2)
        return None
    owner_cik = owner_cik_container.text.strip()

    nonderivative_transaction_container = soup.find_all('nonderivativetransaction')
    if not nonderivative_transaction_container:
        time.sleep(2)
        return None
    nonderivative_transaction = nonderivative_transaction_container[-1]

    transaction_date_container = nonderivative_transaction.find('transactiondate')
    if not transaction_date_container:
        time.sleep(2)
        return None
    transaction_date = transaction_date_container.text.strip()

    shares_after_transaction_container = nonderivative_transaction.find('sharesownedfollowingtransaction')
    if not shares_after_transaction_container:
        time.sleep(2)
        return None
    shares_after_transaction = shares_after_transaction_container.text.strip()

    d = {'ticker': [ticker], 'owner_cik': [owner_cik], 'date': [transaction_date],
         'num_shares': [shares_after_transaction]}
    df = pd.DataFrame(d)
    df.to_sql('sec_ownership', conn, if_exists='append', index=False)


if __name__ == "__main__":
    conn = create_connection_psql(PG_CREDENTIALS)
    make_ticker_cik_table(conn)
    cik_list = make_cik_list(conn)

    with open(EDG_SAVE, 'rb') as f:
        store_year = pickle.load(f)
        store_qtr = pickle.load(f)
        store_index = pickle.load(f)
        store_form = pickle.load(f)

    first_loop = True

    try:
        for year in tqdm(range(2010, 2020), desc='Year'):
            if first_loop:
                if year < store_year:
                    continue
            store_year = year
            for qtr in tqdm(range(1, 5), desc='Qtr'):
                if first_loop:
                    if qtr < store_qtr:
                        continue
                store_qtr = qtr
                indices = get_daily_indices(year, qtr)
                for i, index in enumerate(tqdm(indices, desc='Index')):
                    if first_loop:
                        if i < store_index:
                            continue
                    store_index = i
                    form_4_container = get_form_4(index, cik_list)
                    for j, form in enumerate(tqdm(form_4_container, desc='Form')):
                        if first_loop:
                            if j < store_form:
                                continue
                        store_form = j
                        parse_form_4(form, conn)

                        first_loop = False
    except Exception as e:
        with open(EDG_SAVE, 'wb') as f:
            pickle.dump(store_year, f)
            pickle.dump(store_qtr, f)
            pickle.dump(store_index, f)
            pickle.dump(store_form, f)
        raise Exception(e)
