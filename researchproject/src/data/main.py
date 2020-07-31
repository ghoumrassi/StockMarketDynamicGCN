import pandas as pd
import pickle

from src import WD_OUTPUT, NYT_OUTPUT, FM_OUTPUT, SQLITE_DB
from src.data import format_nytimes, get_wikidata, get_tiingo, fuzzy_match


def main(download=False):
    # Creates counts file for all organisations in nytimes folder
    start_date = '01/01/2010'
    # format_nytimes.get_entity_list(start_date=start_date)

    if download:
        # Creates wikidata tables
        print("Downloading tables from wikidata...")
        # get_wikidata.run_wd_queries()

        # Make ticker lists
        print("Making ticker lists...")
        subsidiaries = pd.read_csv((WD_OUTPUT / 'subsidiaries.csv'))

        # Download new stock data
        print("Downloading stock data...")
        tiingo = get_tiingo.TiingoData()
        tiingo.get(subsidiaries['ticker'].unique(), 'nasdaq')
    else:
        # Make ticker lists
        print("Making ticker lists...")
        subsidiaries = pd.read_csv((WD_OUTPUT / 'subsidiaries.csv'))

    # Fuzzy match nytimes with wikidata
    print("Creating fuzzy matching files...")
    with open((NYT_OUTPUT / 'entities.p'), 'rb') as fn:
        nyt_entity_count = pickle.load(fn)

    nyt_names = list(nyt_entity_count.keys())
    company_aliases = subsidiaries['altLabel'].astype(str).unique()

    # Make nasdaq match output
    fuzzy_match.make_fm_mapping(nyt_names, company_aliases, 'nasdaq')

    # Make nasdaq nyt summaries
    with open((FM_OUTPUT / 'nasdaq_mapper.p'), 'rb') as fn:
        mapper = pickle.load(fn)
    format_nytimes.make_summaries(mapper, 'summaries.csv', start_date=start_date)

    # Make db
    connection = create_connection(SQLITE_DB)
    populate_summaries(conn)
    populate_industry(conn)
    populate_dates(conn)
    populate_mapper(conn)
    populate_subsidiaries(conn)
    populate_ticker(conn)  # This one takes a looong time

    print("DONE!")


if __name__ == "__main__":
    main(download=False)
