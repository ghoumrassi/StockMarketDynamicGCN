from pkg_resources import resource_filename
from pathlib import Path
import pandas as pd

"""
FOR DATA
"""

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

''' Yahoo '''
# Folder containing historic market cap data taken from Yahoo
YAHOO_DATA = Path(resource_filename(__name__, '../data/external/yahoo'))

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

''' SEC EDGAR '''
# Pickle containing last run file info
EDG_SAVE = Path(resource_filename(__name__, '../misc/edgar.p'))

''' Queries '''
QUERIES = Path(resource_filename(__name__, '../queries'))

''' sqlite Database '''
SQLITE_DB = Path(resource_filename(__name__, '../data/projectdata.db'))

''' PostgreSQL Database '''
PG_CREDENTIALS = Path(resource_filename(__name__, '../misc/pg_creds.json'))

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
try:
    COMPANY_NAME_MAPPING = {#
        x[1]['name']: x[1]['Security Name'] for x in pd.read_csv(COMPANY_NAME_MAPPING_FILE).iterrows()
    }
except FileNotFoundError:
    print("Warning: COMPANY_NAME_MAPPING was not found. \
          The program will break if data manipulation is being performed.")
# Adjacency matrix file made using format_nytimes.py
NYT_ADJ_MATRIX = Path(resource_filename(__name__, '../data/processed/nyt_adj.npy'))


"""
FOR MODEL
"""

MODEL_SAVE_DIR = Path(resource_filename(__name__, '../misc/checkpoints'))
if not MODEL_SAVE_DIR.exists():
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ARGS = Path(resource_filename(__name__, '../yaml'))

"""
PYTORCH GEO
"""
GEO_DATA = Path(resource_filename(__name__, '../data/processed/geo'))
if not GEO_DATA.exists():
    GEO_DATA.mkdir(parents=True, exist_ok=True)

"""
MISC
"""
MISC = Path(resource_filename(__name__, '../misc'))
if not MISC.exists():
    MISC.mkdir(parents=True, exist_ok=True)

"""
DARA
"""
DATA = Path(resource_filename(__name__, '../data'))
if not DATA.exists():
    DATA.mkdir(parents=True, exist_ok=True)