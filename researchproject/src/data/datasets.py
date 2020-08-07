import torch
from torch.utils.data import Dataset
import datetime as dt
import numpy as np
from sqlite3.dbapi2 import OperationalError
from sqlalchemy import text

from src import *
from src.data.utils import create_connection, create_connection_psql


class CompanyStockGraphDataset(Dataset):
    """
    Creates dataset for use in D-GNN models.
    Each slice returns [X_t, A_t, y_t]
    """
    def __init__(self, features, device="cpu", window_size=90, predict_periods=3, persistence=None,
                 returns_threshold=0.03, start_date='01/01/2010', end_date=None, timeout=30, db='psql'):
        self.features = features
        self.device = device
        self.window_size = window_size
        self.predict_periods = predict_periods
        self.persistence = persistence
        self.returns_threshold = returns_threshold
        self.start_date = dt.datetime.strptime(start_date, '%d/%m/%Y').timestamp()
        if end_date:
            self.end_date = dt.datetime.strptime(end_date, '%d/%m/%Y').timestamp()
        else:
            self.end_date = dt.datetime.now().timestamp()
        self.timeout = timeout
        self.db = db

        if self.db == 'sqlite':
            self.conn = create_connection(str(SQLITE_DB), timeout=self.timeout)
            self.c = self.conn.cursor()
        elif self.db == 'psql':
            self.engine = create_connection_psql()
        else:
            raise NotImplementedError("Must use modes 'sqlite' or 'psql' for db.")

        with open((QUERIES / self.db / 'article_pair_counts.q'), 'r') as f:
            self.pair_count_query = f.read()

        with open((QUERIES / self.db / 'ticker_history.q'), 'r') as f:
            q = f.read()
            if self.features:
                if self.db == 'sqlite':
                    additional_col_str = ", " + ", ".join(features)
                elif self.db == 'psql':
                    additional_col_str = ', "' + '", "'.join(features) + '"'
            else:
                additional_col_str = ""
            self.ticker_hist_query = q.format(additional_columns=additional_col_str)

        with open((QUERIES / self.db / 'returns_future.q'), 'r') as f:
            self.ticker_future_query = f.read()

        with open((QUERIES / self.db / 'get_distinct_dates.q'), 'r') as f:
            self.distinct_dates_query = f.read()

        with open((QUERIES / self.db / 'get_distinct_tickers.q'), 'r') as f:
            self.distinct_tickers_query = f.read()

        if self.db == 'sqlite':
            self.c.execute(self.distinct_dates_query, (self.start_date, self.end_date))
            dates_results = self.c.fetchall()
            self.c.execute(self.distinct_tickers_query, (self.start_date, self.end_date))
            tickers_results = self.c.fetchall()
        elif self.db == 'psql':
            resultset = self.engine.execute(text(self.distinct_dates_query),
                                startdate=self.start_date, enddate=self.end_date)
            dates_results = resultset.fetchall()
            resultset = self.engine.execute(text(self.distinct_tickers_query),
                                            startdate=self.start_date, enddate=self.end_date)
            tickers_results = resultset.fetchall()
        else:
            raise NotImplementedError("Must use modes 'sqlite' or 'psql' for db.")

        self.idx_date_map = {i: str(int(date[0])) for i, date in enumerate(dates_results)}
        self.date_idx_map = {str(int(date)): i for i, date in self.idx_date_map.items()}
        self.date_array = np.array([int(date) for date in self.date_idx_map.keys()])

        self.idx_ticker_map = {i: ticker[0] for i, ticker in enumerate(tickers_results)}
        self.ticker_idx_map = {ticker: i for i, ticker in self.idx_ticker_map.items()}

    def __len__(self):
        return len(self.idx_date_map) - (self.window_size + self.predict_periods)

    def __getitem__(self, idx):
        start_date = self.idx_date_map[idx]
        current_date = self.idx_date_map[idx + self.window_size]
        end_date = self.idx_date_map[idx + (self.window_size + self.predict_periods)]

        A = self.get_A(idx, start_date, current_date)
        X = self.get_X(idx, start_date, current_date)
        k = self.get_k()
        y = self.get_y(current_date, end_date)

        return A, X, k, y

    def get_X(self, idx, start, current):
        X = torch.zeros(
            (self.window_size, len(self.ticker_idx_map), len(self.features)+1),
            device=self.device)
        if self.db == 'sqlite':
            self.c.execute(self.ticker_hist_query, (start, current))
            results = self.c.fetchall()
        elif self.db == 'psql':
            resultset = self.engine.execute(text(self.ticker_hist_query),
                                            startdate=start, enddate=current)
            results = resultset.fetchall()
        else:
            raise NotImplementedError()

        for date, ticker, returns, *args in results:
            if not returns:
                continue
            date_idx = self.date_idx_map[str(int(date))] - idx
            ticker_idx = self.ticker_idx_map[ticker]
            X[date_idx, ticker_idx, :] = torch.tensor([returns] + args)
        return X

    def get_y(self, current, end):
        y = torch.zeros((len(self.ticker_idx_map),), device=self.device)

        if self.db == 'sqlite':
            self.c.execute(self.ticker_future_query, (current, end))
            results = self.c.fetchall()
        elif self.db == 'psql':
            resultset = self.engine.execute(text(self.ticker_future_query),
                                            startdate=current, enddate=end)
            results = resultset.fetchall()
        else:
            raise NotImplementedError()

        for ticker, returns in results:
            if not returns:
                continue
            ticker_idx = self.ticker_idx_map[ticker]
            if returns < -self.returns_threshold:
                y[ticker_idx] = 0
            elif returns > self.returns_threshold:
                y[ticker_idx] = 2
            else:
                y[ticker_idx] = 1
        return y

    def get_A(self, idx, start, current):
        A = torch.zeros(
            (self.window_size, len(self.ticker_idx_map), len(self.ticker_idx_map)),
            device=self.device
        )

        if self.db == 'sqlite':
            self.c.execute(self.pair_count_query, (start, current))
            results = self.c.fetchall()
        elif self.db == 'psql':
            resultset = self.engine.execute(text(self.pair_count_query),
                                            startdate=start, enddate=current)
            results = resultset.fetchall()
        else:
            raise NotImplementedError()

        for date, a, b, count in results:
            # TODO: Extremely messy solution: please fix.
            try:
                a_j = self.ticker_idx_map[a]
                b_j = self.ticker_idx_map[b]
            except KeyError:
                # print("Ticker doesn't exist.")
                continue
            try:
                d_i = self.date_idx_map[str(int(date))] - idx
            except KeyError:
                new_date = self.date_array[self.date_array < date].max()
                d_i = self.date_idx_map[str(int(new_date))] - idx
            A[d_i, a_j, b_j] += count
            A[d_i, b_j, a_j] += count
        return A

    def get_k(self):
        k = torch.zeros((self.window_size,), device=self.device)
        k = k.fill_(len(self.ticker_idx_map))
        return k

    def close_connection(self):
        self.conn.close()

    def open_connection(self):
        self.conn = create_connection(str(SQLITE_DB), timeout=self.timeout)
        self.c = self.conn.cursor()


if __name__ == "__main__":
    ds = CompanyStockGraphDataset(features=['adjVolume'])
    for i in range(2500, 2510):
        A, X, k, y = ds[i]
        print("Classes: ")
        print(f"-ve: {(y == 0).sum()}")
        print(f"neu: {(y == 1).sum()}")
        print(f"+ve: {(y == 2).sum()}")
        if i == 2500:
            print(f"A.shape: {A.shape}\nX.shape: {X.shape}\ny.shape: {y.shape}\n")
        # print("A: ", A)
        # print("X: ", X)
        # print("y: ", y)

