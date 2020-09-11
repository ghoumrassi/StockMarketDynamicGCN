import torch
from torch_geometric.data import Dataset, Data, Batch
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
    # data is loaded from 1/1/2009 to present
    def __init__(self, root, features=None, device='cpu', start_date='01/01/2010', end_date='31/12/2100',
                 periods=1, sequence_length=30, rthreshold=0.01, persistence=300, test=False, simplify=True,
                 edgetypes=(0,), conv_first=True):
        if features is None:
            features = ('adjVolume', '5-day', '10-day', '20-day', '30-day')

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
        self.feature_mins = []
        self.feature_maxs = []
        self.periods = periods
        self.seq_len = sequence_length
        self.rthreshold = rthreshold
        self.persistence = persistence
        self.test = test
        self.simplify = simplify
        self.edgetypes = edgetypes
        self.conv_first = conv_first
        start_date = dt.datetime.strptime(start_date, '%d/%m/%Y').timestamp()
        end_date = dt.datetime.strptime(end_date, '%d/%m/%Y').timestamp()
        with open((QUERIES / 'psql' / 'get_distinct_dates.q'), 'r') as f:
            self.distinct_dates_query = f.read()
            resultset = self.engine.execute(text(self.distinct_dates_query),
                                            startdate=1230768000, enddate=9e20)
            dates_results = resultset.fetchall()

        with open((QUERIES / 'psql' / 'get_distinct_tickers.q'), 'r') as f:
            self.distinct_tickers_query = f.read()
            resultset = self.engine.execute(text(self.distinct_tickers_query),
                                            startdate=1230768000, enddate=9e20)
            tickers_results = resultset.fetchall()

        with open((QUERIES / 'psql' / 'min_max_feats.q'), 'r') as f:
            self.min_max_query = f.read()

        for feature in self.features:
            resultset = self.engine.execute(text(self.min_max_query.format(feature=feature)),
                                            feature=feature, startdate=start_date, enddate=end_date)
            results = resultset.fetchone()
            self.feature_mins.append(results[0])
            self.feature_maxs.append(results[1])

        self.date_array = np.array([int(date[0]) for date in dates_results])
        self.date_array = self.date_array[(self.date_array > start_date) & (self.date_array <= end_date)]

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
        if self.test:
            return
        data_dir = Path(self.processed_dir)
        data_list = []
        data_range = self.date_array[: -self.periods - 1]
        file_names = []
        for date in data_range:
            fn = '_'.join(
                ['data', str(int(date)), str(self.seq_len), str(self.periods), str(self.persistence), str(self.feat_id)]
            ) + '.pt'
            file_names.append(fn)
        fn_exists = [(data_dir / fn).exists() for fn in file_names]
        fn_can_skip = [all(fn_exists[i: i + self.seq_len]) for i in range(len(fn_exists))]
        if all(fn_can_skip[self.seq_len:]):
            return
        for i, date in enumerate(tqdm(data_range, desc="Processing dataset...")):
            fn = (data_dir / file_names[i])

            if fn_can_skip[i]:
                data_list = []
                continue
            if i < self.seq_len:
                if all(fn_exists[self.seq_len: self.seq_len + i]):
                    continue

            if i == 0:
                prev_date = 0
            else:
                prev_date = data_range[i - 1]
            if i >= len(self.date_array) - self.periods:
                continue
            X = self.get_X(date)
            y = self.get_y(date)
            E, w = self.get_edges(prev_date, date)

            if self.conv_first:
                try:
                    data = Data(x=X, y=y, edge_index=E, edge_attr=w)
                except ValueError:
                    data = Data(x=X, y=y, edge_index=E.long(), edge_attr=w)
                data_list.append(data)
                if len(data_list) < self.seq_len:
                    continue
                elif len(data_list) > self.seq_len:
                    data_list.pop(0)
                batch = Batch.from_data_list(data_list)
                torch.save(batch, fn)
            else:
                data_list.append(X)
                if len(data_list) < self.seq_len:
                    continue
                elif len(data_list) > self.seq_len:
                    data_list.pop(0)
                Xt = torch.stack(data_list)
                try:
                    data = Data(x=Xt, y=y, edge_index=E, edge_attr=w)
                except ValueError:
                    data = Data(x=Xt, y=y, edge_index=E.long(), edge_attr=w)
                torch.save(data, fn)
        print("Done.")

    def len(self):
        return len(self.date_array)

    def get(self, i):
        date = self.date_array[self.seq_len - 1 + i]
        data_dir = Path(self.processed_dir)
        fn = '_'.join(
            ['data', str(int(date)), str(self.seq_len), str(self.periods), str(self.persistence), str(self.feat_id)]
        ) + '.pt'
        data = torch.load((data_dir / fn))
        y = data.y.clone()
        data.r = data.y
        y[data.y < -self.rthreshold] = 0
        y[(data.y >= -self.rthreshold) & (data.y <= self.rthreshold)] = 1
        y[data.y > self.rthreshold] = 2
        data.y = y
        data.edge_attr = data.edge_attr[:, list(self.edgetypes)]
        if self.conv_first:
            data.seq = data.batch
            del data.batch
        else:
            data.x = data.x.permute(1, 0, 2)
        return data

    def get_X(self, date):
        X = torch.zeros((len(self.ticker_idx_map), len(self.features) + 1), device=self.device)
        with open((self.query_dir / 'ticker_history.q'), 'r') as f:
            q = f.read().format(additional_columns="".join([f', "{feat}"' for feat in self.features]))
            rs = self.engine.execute(text(q), date=int(date))
        results = rs.fetchall()
        for ticker, returns, *args in results:
            if not returns:
                continue
            ticker_idx = self.ticker_idx_map[ticker]
            norm_args = []
            for i, arg in enumerate(args):
                norm_arg = (arg - self.feature_mins[i]) / (self.feature_maxs[i] - self.feature_mins[i])
            X[ticker_idx, :] = torch.tensor([returns] + norm_args)
        return X

    def get_y(self, date):
        future_date_idx = np.where(self.date_array == date)[0] + self.periods
        future_date = self.date_array[future_date_idx]
        y = torch.zeros((len(self.ticker_idx_map),), device=self.device)
        with open((self.query_dir / 'future_returns.q'), 'r') as f:
            rs = self.engine.execute(text(f.read()), date=int(date), futuredate=int(future_date))
        results = rs.fetchall()
        for ticker, returns in results:
            if not returns:
                continue
            ticker_idx = self.ticker_idx_map[ticker]
            y[ticker_idx] = returns
        return y

    def get_edges(self, prev_date, date):
        if self.persistence:
            epoch_day = 86400
            start_date = date - (self.persistence * epoch_day)
        else:
            start_date = prev_date
        # NYT
        with open((self.query_dir / 'edges_nyt.q'), 'r') as f:
            rs = self.engine.execute(text(f.read()), prevdate=int(start_date), date=int(date))
        results_1 = rs.fetchall()
        # SEC
        with open((self.query_dir / 'edges_sec.q'), 'r') as f:
            rs = self.engine.execute(text(f.read()), prevdate=int(prev_date), date=int(date))
        results_2 = rs.fetchall()

        if not self.simplify:
            # Correlations
            with open((self.query_dir / 'edges_corr.q'), 'r') as f:
                rs = self.engine.execute(text(f.read()), date=int(date))
            results_3 = rs.fetchall()
            # Reddit
            with open((self.query_dir / 'edges_reddit.q'), 'r') as f:
                rs = self.engine.execute(text(f.read()), prevdate=int(start_date), date=int(date))
            results_4 = rs.fetchall()

            # Wikidata not returning much data
            # #Wikidata
            # with open((self.query_dir / 'edges_wd.q'), 'r') as f:
            #     rs = self.engine.execute(text(f.read()), prevdate=int(start_date), date=int(date))
            # results_5 = rs.fetchall()
        if self.simplify:
            tot_len = len(results_1) + len(results_2)
        else:
            tot_len = len(results_1) + len(results_2) + len(results_3) + len(results_4)

        if tot_len == 0:
            return torch.empty(2, device=self.device), torch.empty(2, device=self.device)
        E = torch.zeros((2, tot_len), dtype=torch.long, device=self.device)
        W = torch.zeros((tot_len, 4), device=self.device)
        i = 0
        prev_edges = {}

        E, W, i, prev_edges = self.make_edges(E, W, i, prev_edges, results_1, position=0)
        E, W, i, prev_edges = self.make_edges(E, W, i, prev_edges, results_2, position=1)
        if not self.simplify:
            E, W, i, prev_edges = self.make_edges(E, W, i, prev_edges, results_3, position=2)
            E, W, i, prev_edges = self.make_edges(E, W, i, prev_edges, results_4, position=3)

        E = E[:, :i]
        E = torch.cat((E, E.flip(dims=(0,))), dim=1)
        W = W[:i, :]
        W = torch.cat((W, W), dim=0)

        return E, W

    def make_edges(self, E, W, i, prev_edges, results, position):
        for a, b, weight in results:
            try:
                ai = self.ticker_idx_map[a]
                bi = self.ticker_idx_map[b]
            except KeyError:
                continue
            new_edges = torch.tensor([[ai, bi]], device=self.device)
            new_weights = torch.tensor([0, 0, 0, 0], device=self.device)
            new_weights[position] = weight
            if (ai, bi) in prev_edges:
                idx = prev_edges[(ai, bi)]
                W[idx, :] += new_weights
            else:
                prev_edges[(ai, bi)] = i
                E[:, i] = new_edges
                W[i, :] = new_weights
                i += 1
        return E, W, i, prev_edges

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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ds = CompanyGraphDatasetGeo(root=GEO_DATA, periods=3, device=device, simplify=True, start_date='01/01/2010',
                                conv_first=False)
    for i in range(5):
        print(ds[i])
