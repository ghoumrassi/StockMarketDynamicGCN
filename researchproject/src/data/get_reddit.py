import praw
from sqlalchemy import text
import string
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import datetime as dt
import json
from itertools import combinations
from pathlib import Path
import pickle
from src.data.utils import create_connection_psql
from src import PG_CREDENTIALS, REDDIT_CREDENTIALS


def get_reddit_mentions(db):
    pickle_file = Path('../misc/reddit.p')
    finance_srs = [
        "investing", "stocks", "RobinHood", "wallstreetbets", "SecurityAnalysis", "InvestmentClub", "StockMarket",
        "Stock_Picks", "Forex", "options"
    ]

    news_srs = [
        "news", "WorldNews", "anythinggoesnews", "fullnews", "inthenews", "neutralnews", "nottheonion", "offbeat",
        "onthescene", "qualitynews", "thenews", "upliftingnews", "USNews"
    ]

    politics_srs = [
        "politics", "CredibleDefense", "GeoPolitics", "neutralpolitics", "politic", "politicaldiscussion",
        "worldpolitics"
    ]

    subreddits_string = "+".join(["+".join(sr) for sr in [finance_srs, news_srs, politics_srs]])
    with open(REDDIT_CREDENTIALS, 'r') as f:
        creds = json.load(f)
    reddit = praw.Reddit(**creds)
    company_terms = get_company_terms(db)

    len_iterator = sum(1 for _ in combinations(company_terms, 2))
    company_iterator = combinations(company_terms, 2)
    if pickle_file.exists():
        with open(pickle_file, 'rb') as f:
            start_idx, mention_counts = pickle.load(f)
    else:
        start_idx = None
        mention_counts = defaultdict(int)
    for i, (a, b) in enumerate(tqdm(company_iterator, total=len_iterator)):
        if start_idx and i < start_idx:
            continue
        results = reddit.subreddit(subreddits_string).search(f"{company_terms[a]} AND {company_terms[b]}")
        for result in results:
            date = dt.datetime.fromtimestamp(result.created_utc).date()
            date = dt.datetime(
                year=date.year,
                month=date.month,
                day=date.day,
                tzinfo=dt.timezone.utc
            ).timestamp()
            mention_counts[(a, b, int(date))] += 1
        with open(pickle_file, 'wb') as f:
            pickle.dump((i, mention_counts), f)

    df = pd.Series(mention_counts).reset_index()
    df.columns = ['a', 'b', 'date', 'count']
    df.to_sql('reddit_mentions', engine, if_exists='replace', index=False)


def get_company_terms(engine):
    rs = engine.execute("""SELECT DISTINCT "Symbol" FROM nasdaq100""")
    results = rs.fetchall()
    ticker_list = [result[0] for result in results]

    terms = {}
    for ticker in ticker_list:
        q = text("""SELECT DISTINCT "altLabel", "parentLabel" FROM subsidiaries 
        WHERE ticker = :ticker AND "subsidiaryLabel" = "parentLabel" """)
        rs = engine.execute(q, ticker=ticker)
        results = rs.fetchall()
        if results:
            results = list(set([result[0] for result in results] + [results[0][1]]))
            results = [
                f"({result.lower().translate(str.maketrans('', '', string.punctuation))})" for result in
                results + [ticker]
            ]
            term_str = "(" + " OR ".join(results) + ")"
            terms[ticker] = term_str
        else:
            print(ticker)
            terms[ticker] = ticker.lower()

    return terms


if __name__ == "__main__":
    engine = create_connection_psql(PG_CREDENTIALS)
    get_reddit_mentions(engine)