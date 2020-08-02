import torch
from torch.utils.data import Dataset
import pickle
import datetime as dt
import numpy as np


from src import *
from src.data.utils import create_connection


class CompanyStockGraphDataset(Dataset):
    """
    Creates dataset for use in D-GNN models.
    Each slice returns [X_t, A_t, y_t]
    """

    def __init__(self, features, device="cpu", window_size=90, predict_periods=3, persistence=None,
                 returns_threshold=0.01, start_date='01/01/2010', end_date=None):
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

        self.conn = create_connection(str(SQLITE_DB))
        self.c = self.conn.cursor()

        with open((SQL_QUERIES / 'article_pair_counts.q'), 'r') as f:
            self.pair_count_query = f.read()

        with open((SQL_QUERIES / 'ticker_history.q'), 'r') as f:
            q = f.read()
            if self.features:
                additional_col_str = ", " + ", ".join(features)
            else:
                additional_col_str = ""
            self.ticker_hist_query = q.format(additional_columns=additional_col_str)
            self.ticker_future_query = q.format(additional_columns="")

        with open((SQL_QUERIES / 'get_distinct_dates.q'), 'r') as f:
            self.distinct_dates_query = f.read()
        self.c.execute(self.distinct_dates_query, (self.start_date, self.end_date))
        self.idx_date_map = {i: str(int(date[0])) for i, date in enumerate(self.c.fetchall())}
        self.date_idx_map = {str(int(date)): i for i, date in self.idx_date_map.items()}
        self.date_array = np.array([int(date) for date in self.date_idx_map.keys()])

        with open((SQL_QUERIES / 'get_distinct_tickers.q'), 'r') as f:
            self.distinct_tickers_query = f.read()
        self.c.execute(self.distinct_tickers_query, (self.start_date, self.end_date))
        self.idx_ticker_map = {i: ticker[0] for i, ticker in enumerate(self.c.fetchall())}
        self.ticker_idx_map = {ticker: i for i, ticker in self.idx_ticker_map.items()}

    def __len__(self):
        return len(self.idx_date_map) - (self.window_size + self.predict_periods)

    def __getitem__(self, idx):
        X = torch.zeros(
            (self.window_size, len(self.ticker_idx_map), len(self.features)+1),
            device=self.device)
        y = torch.zeros((len(self.ticker_idx_map),), device=self.device)
        A = torch.zeros(
            (self.window_size, len(self.ticker_idx_map), len(self.ticker_idx_map)),
            device=self.device
        )
        k = torch.zeros((self.window_size, ), device=self.device)

        start_date = self.idx_date_map[idx]
        current_date = self.idx_date_map[idx + self.window_size]
        end_date = self.idx_date_map[idx + (self.window_size + self.predict_periods)]

        # Get A
        self.c.execute(self.pair_count_query, (start_date, current_date))
        results = self.c.fetchall()
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

        # Get X
        self.c.execute(self.ticker_hist_query, (start_date, current_date))
        results = self.c.fetchall()
        for date, ticker, returns, *args in results:
            if not returns:
                continue
            date_idx = self.date_idx_map[str(int(date))] - idx
            ticker_idx = self.ticker_idx_map[ticker]
            X[date_idx, ticker_idx, :] = torch.tensor([returns] + args)

        # Get y
        self.c.execute(self.ticker_future_query, (end_date, end_date))
        results = self.c.fetchall()
        for _, ticker, returns in results:
            if not returns:
                continue
            ticker_idx = self.ticker_idx_map[ticker]
            # oh_vector = torch.zeros(3, device=self.device)
            if returns < -self.returns_threshold:
                # oh_vector[0] = 1
                y[ticker_idx] = 0
            elif returns > self.returns_threshold:
                # oh_vector[2] = 1
                y[ticker_idx] = 2
            else:
                # oh_vector[1] = 1
                y[ticker_idx] = 1
        # Get k
        k = k.fill_(len(self.ticker_idx_map))

        return A, X, k, y


if __name__ == "__main__":
    import unittest

    ds = CompanyStockGraphDataset(features=['adjVolume'])
