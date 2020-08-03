"""
Contains functions pertaining to the handling and logging of errors.
"""

import pandas as pd
import time
from src import SQLITE_DB
from src.data.utils import create_connection


def error_handling(error):
    pass


def log_model_error(error, model_files, epoch, iteration):
    timestamp = time.time()
    model_files = "||".join([str(model_file) for model_file in model_files])
    df = pd.DataFrame({'timestamp': [timestamp], 'model_files': [model_files],
                       'epoch': [epoch], 'iteration': [iteration], 'error': str(error)})
    df.set_index('timestamp')
    df.to_sql("modelerror", create_connection(SQLITE_DB), if_exists='append')