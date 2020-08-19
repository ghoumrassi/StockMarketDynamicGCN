import torch
from torch.utils.data import Dataset
import datetime as dt
import numpy as np
from sqlite3.dbapi2 import OperationalError
from sqlalchemy import text
from tqdm import tqdm
import networkx as nx
import itertools
from torch_geometric.data import Data

from src import *
from src.data.utils import create_connection, create_connection_psql


class CompanyStockGraphDataset(Dataset):
    """
    Creates dataset for use in D-GNN models.
    Each slice returns [X_t, A_t, y_t]
    """
    def __init__(self, features, device="cpu", window_size=90, predict_periods=3, persistence=None, adj=True,
                 adj2=False, k=True, returns_threshold=0.03, start_date='01/01/2010', end_date=None, timeout=30,
                 db='psql'):
        self.features = features

        self.device = device
        self.window_size = window_size
        self.predict_periods = predict_periods
        self.persistence = persistence
        self.adj_flag = adj
        self.adj_flag_2 = adj2
        self.k_flag = k
        self.format = format
        self.returns_threshold = returns_threshold
        self.start_date = dt.datetime.strptime(start_date, '%d/%m/%Y').timestamp()
        if end_date:
            self.end_date = dt.datetime.strptime(end_date, '%d/%m/%Y').timestamp()
        else:
            self.end_date = dt.datetime.now().timestamp()
        self.timeout = timeout
        self.db = db

        self.engine = create_connection_psql(PG_CREDENTIALS)

        # Make normalisers for feature columns
        self.normalisers = []
        for feature in ['returns'] + self.features:
            resultset = self.engine.execute(f"""SELECT MIN("{feature}"), MAX("{feature}") FROM tickerdata""")
            results = resultset.fetchall()
            self.normalisers.append(make_normaliser(*results[0]))

        with open((QUERIES / 'psql' / 'article_pair_counts.q'), 'r') as f:
            self.pair_count_query = f.read()

        with open((QUERIES / 'psql' / 'ticker_history.q'), 'r') as f:
            q = f.read()
            if self.features:
                additional_col_str = ', "' + '", "'.join(features) + '"'
            else:
                additional_col_str = ""
            self.ticker_hist_query = q.format(additional_columns=additional_col_str)

        with open((QUERIES / 'psql' / 'returns_future.q'), 'r') as f:
            self.ticker_future_query = f.read()

        with open((QUERIES / 'psql' / 'get_distinct_dates.q'), 'r') as f:
            self.distinct_dates_query = f.read()

        with open((QUERIES / 'psql' / 'get_distinct_tickers.q'), 'r') as f:
            self.distinct_tickers_query = f.read()

        with open((QUERIES / 'psql' / 'get_joint_ownership.q'), 'r') as f:
            self.sec_joint_query = f.read()

        resultset = self.engine.execute(text(self.distinct_dates_query),
                            startdate=self.start_date, enddate=self.end_date)
        dates_results = resultset.fetchall()
        resultset = self.engine.execute(text(self.distinct_tickers_query),
                                        startdate=self.start_date, enddate=self.end_date)
        tickers_results = resultset.fetchall()

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

        output = []

        X = self.get_X(idx, start_date, current_date)

        for i in range(1, X.shape[2]):
            X[:, :, i] = self.normalisers[i-1](X[:, :, i])

        y = self.get_y(current_date, end_date)

        if self.adj_flag:
            A = self.get_A(idx, start_date, current_date)
            output.append(A)

        if self.adj_flag_2:
            A_2 = self.get_A2(idx, start_date, current_date)
            output.append(A_2)

        output.append(X)

        if self.k_flag:
            k = self.get_k()
            output.append(k)

        output.append(y)

        return output


    def get_X(self, idx, start, current):
        X = torch.zeros(
            (self.window_size, len(self.ticker_idx_map), len(self.features)+1),
            device=self.device)

        resultset = self.engine.execute(text(self.ticker_hist_query),
                                        startdate=start, enddate=current)
        results = resultset.fetchall()

        for date, ticker, returns, *args in results:
            if not returns:
                continue
            date_idx = self.date_idx_map[str(int(date))] - idx
            ticker_idx = self.ticker_idx_map[ticker]
            X[date_idx, ticker_idx, :] = torch.tensor([returns] + args)
        return X

    def get_y(self, current, end):
        y = torch.zeros((len(self.ticker_idx_map),), device=self.device)
        resultset = self.engine.execute(text(self.ticker_future_query),
                                        startdate=current, enddate=end)
        results = resultset.fetchall()

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
        resultset = self.engine.execute(text(self.pair_count_query),
                                        startdate=start, enddate=current)
        results = resultset.fetchall()

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
        for i in range(A.shape[0]):
            A[i] = self.normalise_adj(A[i])
        return A

    def get_A2(self, idx, start, current):

        resultset = self.engine.execute(text(self.sec_joint_query),
                                        startdate=start, enddate=current)
        results = resultset.fetchall()
        A = torch.zeros(
            (self.window_size, len(self.ticker_idx_map), len(self.ticker_idx_map)),
            device=self.device
        )
        for date_start, date_end, a, b, weight in results:
            # TODO: Extremely messy solution: please fix.
            try:
                a_j = self.ticker_idx_map[a]
                b_j = self.ticker_idx_map[b]
            except KeyError:
                # print("Ticker doesn't exist.")
                continue
            new_start_array = self.date_array[self.date_array <= date_start]
            if new_start_array.size != 0:
                new_start_date = new_start_array.max()
                d_i_s = self.date_idx_map[str(int(new_start_date))] - idx
                if d_i_s < 0:
                    d_i_s = 0
            else:
                d_i_s = 0

            new_end_array = self.date_array[self.date_array <= date_end]
            if new_end_array.size != 0:
                new_end_date = new_end_array.max()
                d_i_e = self.date_idx_map[str(int(new_end_date))] - idx
            else:
                continue

            if (d_i_e >= d_i_s) and (d_i_e <= self.window_size):
                pass
            else:
                continue

            A[d_i_s: d_i_e, a_j, b_j] += weight
            A[d_i_s: d_i_e, b_j, a_j] += weight

        for i in range(A.shape[0]):
            A[i] = self.normalise_adj(A[i])
        return A

    def normalise_adj(self, A):
        A_tilda = A + torch.eye(A.shape[0], device=self.device)
        D_tilda = (A_tilda.sum(dim=0) ** (-1/2)).diag()
        A_norm = D_tilda * A_tilda * D_tilda
        return A_norm

    def get_k(self):
        k = torch.zeros((self.window_size,), device=self.device)
        k = k.fill_(len(self.ticker_idx_map))
        return k

    def close_connection(self):
        self.conn.close()


class CompanyStockGraphDatasetPTG(CompanyStockGraphDataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, idx):
        start_date = self.idx_date_map[idx]
        current_date = self.idx_date_map[idx + self.window_size]
        end_date = self.idx_date_map[idx + (self.window_size + self.predict_periods)]

        output = []

        X = self.get_X(idx, start_date, current_date)

        for i in range(1, X.shape[2]):
            X[:, :, i] = self.normalisers[i-1](X[:, :, i])

        y = self.get_y(current_date, end_date)

        if self.adj_flag:
            A = self.get_A(idx, start_date, current_date)
            output.append(A)

        if self.adj_flag_2:
            A_2 = self.get_A2(idx, start_date, current_date)
            output.append(A_2)

        output.append(X)

        if self.k_flag:
            k = self.get_k()
            output.append(k)

        output.append(y)

        return output

    def get_A(self, idx, start, current):
        edge_indices = [None] * self.window_size
        edge_weights = [None] * self.window_size
        resultset = self.engine.execute(text(self.pair_count_query),
                                        startdate=start, enddate=current)
        results = resultset.fetchall()

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
            new_indices = torch.tensor([[a_j, b_j], [b_j, a_j]], device=self.device)
            new_weights = torch.tensor([count, count], device=self.device)
            if edge_indices[d_i]:
                edge_indices[d_i] = torch.cat((edge_indices[d_i], new_indices), dim=1)
                edge_weights[d_i] = torch.cat((edge_weights[d_i], new_weights))
            else:
                edge_indices[d_i] = new_indices
                edge_weights[d_i] = new_weights

        return (edge_indices, edge_weights)

    def get_A2(self, idx, start, current):
        edge_indices = [None] * self.window_size
        edge_weights = [None] * self.window_size
        resultset = self.engine.execute(text(self.sec_joint_query),
                                        startdate=start, enddate=current)
        results = resultset.fetchall()

        for date_start, date_end, a, b, weight in results:
            # TODO: Extremely messy solution: please fix.
            try:
                a_j = self.ticker_idx_map[a]
                b_j = self.ticker_idx_map[b]
            except KeyError:
                # print("Ticker doesn't exist.")
                continue
            new_start_array = self.date_array[self.date_array <= date_start]
            if new_start_array.size != 0:
                new_start_date = new_start_array.max()
                d_i_s = self.date_idx_map[str(int(new_start_date))] - idx
                if d_i_s < 0:
                    d_i_s = 0
            else:
                d_i_s = 0

            new_end_array = self.date_array[self.date_array <= date_end]
            if new_end_array.size != 0:
                new_end_date = new_end_array.max()
                d_i_e = self.date_idx_map[str(int(new_end_date))] - idx
            else:
                continue

            if (d_i_e >= d_i_s) and (d_i_e <= self.window_size):
                pass
            else:
                continue

            new_indices = torch.tensor([[a_j, b_j], [b_j, a_j]], device=self.device)
            new_weights = torch.tensor([weight, weight], device=self.device)
            for i in range(d_i_s, d_i_e):
                if edge_indices[i]:
                    edge_indices[i] = torch.cat((edge_indices[i], new_indices), dim=1)
                    edge_weights[i] = torch.cat((edge_weights[i], new_weights))
                else:
                    edge_indices[i] = new_indices
                    edge_weights[i] = new_weights

        return (edge_indices, edge_weights)

def make_normaliser(min, max):
    return lambda x: (x - min) / (max - min)


if __name__ == "__main__":
    ds = CompanyStockGraphDataset(features=['adjVolume'], adj=False, adj2=True, format='edge_index')
    for i in tqdm(range(2500, 2510)):
        A, A_2, X, k, y = ds[i]
        print("Classes: ")
        print(f"-ve: {(y == 0).sum()}")
        print(f"neu: {(y == 1).sum()}")
        print(f"+ve: {(y == 2).sum()}")
        if i == 2500:
            print(f"A.shape: {A.shape}\nX.shape: {X.shape}\ny.shape: {y.shape}\n")
        # print("A: ", A)
        # print("X: ", X)
        # print("y: ", y)

