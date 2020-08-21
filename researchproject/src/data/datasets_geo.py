import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils.undirected import to_undirected, is_undirected
from sqlalchemy import text
import numpy as np
from pathlib import Path
from tqdm import tqdm
import datetime as dt
from itertools import product
import pickle

from src import QUERIES, PG_CREDENTIALS, GEO_DATA, MISC
from src.data.utils import create_connection_psql


class CompanyGraphDatasetGeo(Dataset):
    # start date 2010-01-01
    def __init__(self, root, features=None, device='cpu', start_date='01/01/2010', end_date='31/12/2100',
                 periods=1, sequence_length=30, rthreshold=0.01):
        if features is None:
            features = ('adjVolume', 'adjHigh', 'adjLow')

        geo_pkl = (MISC / 'dgeo.p')
        if geo_pkl.exists():
            with open(geo_pkl, 'rb') as f:
                geo_feat_ids = pickle.load(f)
        else:
            geo_feat_ids = {}
        if "".join(sorted(features)) not in geo_feat_ids:
            geo_feat_ids["".join(sorted(features))] = len(geo_feat_ids)
            with open(geo_pkl, 'wb') as f:
                pickle.dump(geo_feat_ids, f)
        self.feat_id = geo_feat_ids["".join(sorted(features))]
        self.device = device
        self.engine = create_connection_psql(PG_CREDENTIALS)
        self.features = features
        self.periods = periods
        self.seq_len = sequence_length
        self.rthreshold = rthreshold
        start_date = dt.datetime.strptime(start_date, '%d/%m/%Y').timestamp()
        end_date = dt.datetime.strptime(end_date, '%d/%m/%Y').timestamp()
        with open((QUERIES / 'psql' / 'get_distinct_dates.q'), 'r') as f:
            self.distinct_dates_query = f.read()
            resultset = self.engine.execute(text(self.distinct_dates_query),
                                            startdate=start_date, enddate=end_date)
            dates_results = resultset.fetchall()

        with open((QUERIES / 'psql' / 'get_distinct_tickers.q'), 'r') as f:
            self.distinct_tickers_query = f.read()
            resultset = self.engine.execute(text(self.distinct_tickers_query),
                                            startdate=start_date, enddate=end_date)
            tickers_results = resultset.fetchall()

        self.idx_date_map = {i: str(int(date[0])) for i, date in enumerate(dates_results)}
        self.date_idx_map = {str(int(date)): i for i, date in self.idx_date_map.items()}
        self.date_array = np.array([int(date) for date in self.date_idx_map.keys()])

        self.idx_ticker_map = {i: ticker[0] for i, ticker in enumerate(tickers_results)}
        self.ticker_idx_map = {ticker: i for i, ticker in self.idx_ticker_map.items()}
        self.query_dir = (Path(QUERIES) / 'psql' / 'geo')
        super().__init__(root=root)

    @property
    def processed_file_names(self):
        proc_files = []
        for date in self.date_array[self.seq_len - 1: -self.periods]:
            proc_files.append(f'data_{str(int(date))}_{str(self.seq_len)}_{str(self.periods)}_{self.feat_id}.pt')
        return proc_files

    def process(self):
        data_dir = Path(self.processed_dir)
        data_list = []
        for i, date in enumerate(tqdm(self.date_array, desc="Processing dataset...")):
            fn = (data_dir / f'data_{str(int(date))}_{str(self.seq_len)}_{str(self.periods)}_{self.feat_id}.pt')
            if fn.exists():
                continue
            if i == 0:
                prev_date = 0
            else:
                prev_date = self.idx_date_map[i - 1]
            if i >= len(self.date_array) - self.periods:
                continue
            X = self.get_X(date)
            y = self.get_y(date)
            E, w = self.get_edges(prev_date, date)
            try:
                data = Data(x=X, y=y, edge_index=E, edge_attr=w)
            except ValueError:
                data = Data(x=X, y=y, edge_index=E.long(), edge_attr=w)
            data_list.append(data)
            if len(data_list) < self.seq_len:
                continue
            elif len(data_list) > self.seq_len:
                data_list.pop(0)

            data, slices = self.collate(data_list)
            torch.save((data, slices), fn)
        print("Done.")

    def len(self):
        return len(self.processed_file_names)

    def get(self, i):
        date = self.idx_date_map[i + self.seq_len - 1] # maybeee?
        data_dir = Path(self.processed_dir)
        data = torch.load(
            (data_dir / f'data_{str(int(date))}_{str(self.seq_len)}_{str(self.periods)}_{self.feat_id}.pt')
        )
        y = data[0].y.clone()
        y[data[0].y < -self.rthreshold] = 0
        y[(data[0].y >= -self.rthreshold) & (data[0].y <= self.rthreshold)] = 1
        y[data[0].y > self.rthreshold] = 2
        data[0].y = y
        return data

    def get_X(self, date):
        X = torch.zeros((len(self.ticker_idx_map), len(self.features)+1), device=self.device)
        with open((self.query_dir / 'ticker_history.q'), 'r') as f:
            q = f.read().format(additional_columns="".join([f', "{feat}"' for feat in self.features]))
            rs = self.engine.execute(text(q), date=int(date))
        results = rs.fetchall()
        for ticker, returns, *args in results:
            if not returns:
                continue
            ticker_idx = self.ticker_idx_map[ticker]
            X[ticker_idx, :] = torch.tensor([returns] + args)
        return X

    def get_y(self, date):
        future_date_idx = np.where(self.date_array == date)[0] + self.periods
        future_date = self.date_array[future_date_idx]
        y = torch.zeros((len(self.ticker_idx_map),), device=self.device)
        with open((self.query_dir / 'future_returns.q'), 'r') as f:
            rs = self.engine.execute(text(f.read()), date=int(future_date))
        results = rs.fetchall()
        for ticker, returns in results:
            if not returns:
                continue
            ticker_idx = self.ticker_idx_map[ticker]
            y[ticker_idx] = returns
        return y

    def get_edges(self, prev_date, date):
        with open((self.query_dir / 'edges_nyt.q'), 'r') as f:
            rs = self.engine.execute(text(f.read()), prevdate=int(prev_date), date=int(date))
        results_1 = rs.fetchall()
        with open((self.query_dir / 'edges_sec.q'), 'r') as f:
            rs = self.engine.execute(text(f.read()), prevdate=int(prev_date), date=int(date))
        results_2 = rs.fetchall()
        with open((self.query_dir / 'edges_corr.q'), 'r') as f:
            rs = self.engine.execute(text(f.read()), date=int(date))
        results_3 = rs.fetchall()
        tot_len = len(results_1) + len(results_2) + len(results_3)
        if tot_len == 0:
            return torch.empty(2, device=self.device), torch.empty(2, device=self.device)
        E = torch.zeros((2, tot_len), dtype=torch.long, device=self.device)
        W = torch.zeros((tot_len, 3), device=self.device)
        i = 0
        prev_edges = {}
        for a, b, weight in results_1:
            try:
                ai = self.ticker_idx_map[a]
                bi = self.ticker_idx_map[b]
            except KeyError:
                continue
            new_edges = torch.tensor(
                [[ai, bi]],
                device=self.device)
            new_weights = torch.tensor(
                [weight, 0, 0],
                device=self.device)

            if str(ai) + " " + str(bi) in prev_edges:
                idx = prev_edges[str(ai) + " " + str(bi)]
                W[idx, :] += new_weights
            else:
                prev_edges[str(ai) + " " + str(bi)] = i
                E[:, i] = new_edges
                W[i, :] = new_weights
                i += 1
        for a, b, weight in results_2:
            try:
                ai = self.ticker_idx_map[a]
                bi = self.ticker_idx_map[b]
            except KeyError:
                continue
            new_edges = torch.tensor(
                [[ai, bi]],
                device=self.device)
            new_weights = torch.tensor(
                [0, weight, 0],
                device=self.device)

            if str(ai) + " " + str(bi) in prev_edges:
                idx = prev_edges[str(ai) + " " + str(bi)]
                W[idx, :] += new_weights
            else:
                prev_edges[str(ai) + " " + str(bi)] = i
                E[:, i] = new_edges
                W[i, :] = new_weights
                i += 1
        for a, b, weight in results_3:
            try:
                ai = self.ticker_idx_map[a]
                bi = self.ticker_idx_map[b]
            except KeyError:
                continue
            new_edges = torch.tensor(
                [[ai, bi]],
                device=self.device)
            new_weights = torch.tensor(
                [0, 0, weight],
                device=self.device)

            if str(ai) + " " + str(bi) in prev_edges:
                idx = prev_edges[str(ai) + " " + str(bi)]
                W[idx, :] += new_weights
            else:
                prev_edges[str(ai) + " " + str(bi)] = i
                E[:, i] = new_edges
                W[i, :] = new_weights
                i += 1

        E = E[:, : i]
        E = torch.cat((E, E.flip(dims=(0,))), dim=1)
        W = W[: i, :]
        W = torch.cat((W, W), dim=0)

        return E, W

    @staticmethod
    def collate(data_list):
        r"""Collates a python list of data objects to the internal storage
        format of :class:`torch_geometric.data.InMemoryDataset`."""
        keys = data_list[0].keys
        data = data_list[0].__class__()

        for key in keys:
            data[key] = []
        slices = {key: [0] for key in keys}

        for item, key in product(data_list, keys):
            data[key].append(item[key])
            if torch.is_tensor(item[key]):
                s = slices[key][-1] + item[key].size(
                    item.__cat_dim__(key, item[key]))
            else:
                s = slices[key][-1] + 1
            slices[key].append(s)

        if hasattr(data_list[0], '__num_nodes__'):
            data.__num_nodes__ = []
            for item in data_list:
                data.__num_nodes__.append(item.num_nodes)

        for key in keys:
            item = data_list[0][key]
            if torch.is_tensor(item):
                data[key] = torch.cat(data[key],
                                      dim=data.__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                data[key] = torch.tensor(data[key])

            slices[key] = torch.tensor(slices[key], dtype=torch.long)

        return data, slices


if __name__ == "__main__":
    ds = CompanyGraphDatasetGeo(root=GEO_DATA)
    for i in range(5):
        print(ds[i])