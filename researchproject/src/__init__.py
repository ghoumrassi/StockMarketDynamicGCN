from pkg_resources import resource_filename
from pathlib import Path
import pandas as pd

''' NY Times '''
# NYTimes raw files
NY_FOLDER = Path(resource_filename(__name__, '../data/external/nytimes'))

# NYT processed data output
NYT_OUTPUT = Path(resource_filename(__name__, '../data/interim/nytimes'))

''' Tiingo '''
# Folder containing historic price data taken from Tiingo
TIINGO_DATA = Path(resource_filename(__name__, '../data/external/tiingo'))
# API key for access to Tiingo API
TIINGO_KEY = Path(resource_filename(__name__, '../misc/key.json'))

''' Wikidata '''
# Folder containing SPARQL queries for Wikidata
WD_QUERIES = Path(resource_filename(__name__, '../queries/SPARQL'))
# Folder containing output Wikidata files
WD_OUTPUT = Path(resource_filename(__name__, '../data/external/wikidata'))

''' Fuzzy matching '''
# Output folder for fuzzy matching
FM_OUTPUT = Path(resource_filename(__name__, '../data/interim/fuzzy'))
# Text file containing common terms in entity list for replacement
FUZZY_STOP_FILE = Path(resource_filename(__name__, '../misc/company_stopwords.txt'))

''' SQLite Database '''
SQLITE_DB = Path(resource_filename(__name__, '../data/projectdata.db'))
SQL_QUERIES = Path(resource_filename(__name__, '../queries/SQLite'))

''' H5 '''
# H5 File
H5_FILE = Path(resource_filename(__name__, '../data/interim/h5/data.h5'))

''' Other '''
# S&P500 Company Information [NOT USED ANYMORE]
# (obtained from: https://datahub.io/core/s-and-p-500-companies#resource-s-and-p-500-companies_zip)
COMPANY_DATA = Path(resource_filename(__name__, '../data/external/datahub_io_s_and_p_companies.csv'))

# File of all NASDAQ listed tickers (obtained from: ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt)
NASDAQ_TICKERS = Path(resource_filename(__name__, '../data/external/nasdaq/nasdaqlisted.txt'))

# Mapping file made using "fuzzy_match.py" that maps NYT company names to their "true" variant.
COMPANY_NAME_MAPPING_FILE = Path(resource_filename(__name__, '../data/processed/nyt_to_nasdaq_mapping_current.csv'))
COMPANY_NAME_MAPPING = {x[1]['name']: x[1]['Security Name'] for x in pd.read_csv(COMPANY_NAME_MAPPING_FILE).iterrows()}

# Adjacency matrix file made using format_nytimes.py
NYT_ADJ_MATRIX = Path(resource_filename(__name__, '../data/processed/nyt_adj.npy'))


