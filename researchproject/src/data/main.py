import pandas as pd
import pickle

from researchproject.src import WD_OUTPUT, NYT_ENTITY_COUNT_MAP, FM_OUTPUT
from researchproject.src.data import format_nytimes, get_wikidata, get_tiingo, fuzzy_match


def main(download=False):
    # Creates counts file for all organisations in nytimes folder
    # format_nytimes.get_entity_list(start_date='01/01/2010')

    if download:
        # Creates wikidata tables
        print("Downloading tables from wikidata...")
        # get_wikidata.run_wd_queries()

        # Make ticker lists
        print("Making ticker lists...")
        nasdaq_subsidiaries = pd.read_csv((WD_OUTPUT / 'subsidiaries.csv'))
        lse_subsidiaries = pd.read_csv((WD_OUTPUT / 'subsidiaries_lse.csv'))

        # Download new stock data
        print("Downloading stock data...")
        tiingo = get_tiingo.TiingoData()
        tiingo.get(nasdaq_subsidiaries['ticker'].unique(), 'nasdaq')
        tiingo.get(lse_subsidiaries['ticker'].unique(), 'lse')
    else:
        # Make ticker lists
        print("Making ticker lists...")
        nasdaq_subsidiaries = pd.read_csv((WD_OUTPUT / 'subsidiaries.csv'))
        lse_subsidiaries = pd.read_csv((WD_OUTPUT / 'subsidiaries_lse.csv'))

    # Fuzzy match nytimes with wikidata
    print("Creating fuzzy matching files...")
    with open(NYT_ENTITY_COUNT_MAP, 'rb') as fn:
        nyt_entity_count = pickle.load(fn)

    nyt_names = list(nyt_entity_count.keys())
    nasdaq_names = nasdaq_subsidiaries['altLabel'].astype(str).unique()
    lse_names = lse_subsidiaries['altLabel'].astype(str).unique()

    # Make nasdaq match output
    fuzzy_match.make_fm_mapping(nyt_names, nasdaq_names, 'nasdaq_mapper.p')

    # Make lse match output
    fuzzy_match.make_fm_mapping(nyt_names, lse_names, 'lse_mapper.p')

    print("DONE!")


if __name__ == "__main__":
    main(download=False)
