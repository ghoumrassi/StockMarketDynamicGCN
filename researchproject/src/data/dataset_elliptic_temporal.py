import os
import torch
import tarfile
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import product
from torch_geometric.data import Dataset, Data

from src import MISC, DATA


class EllipticTemporalDataset(Dataset):
    def __init__(self, device='cpu'):
        self.device = device
        self.seq_len = 7
        super().__init__(root=(DATA / 'processed' / 'elliptic'))

    @property
    def processed_file_names(self):
        return [f'data_{str(int(time))}.pt' for time in range(self.seq_len-1, 49)]

    def len(self):
        return len(self.processed_file_names)

    def get(self, i):
        data_dir = Path(self.processed_dir)
        data = torch.load((data_dir / f'data_{str(int(i+self.seq_len-1))}.pt'))
        return data

    def process(self):
        data_dir = Path(self.processed_dir)
        tar_archive = tarfile.open((MISC / 'elliptic_bitcoin_dataset_cont.tar.gz'), 'r:gz')
        nodes_labels_times = self.load_node_labels(tar_archive)
        edges = self.load_transactions(tar_archive)
        _, nodes = self.load_node_feats(tar_archive)
        periods = nodes_labels_times[:, 2].max().item() + 1
        num_labels = nodes_labels_times[:, 1].max().item() + 1
        num_nodes = nodes.shape[0]
        nodes = nodes.to(self.device)
        data_list = []
        for time in tqdm(range(periods)):
            fn = (data_dir / f'data_{str(int(time))}.pt')
            nodes_labels_t = nodes_labels_times[nodes_labels_times[:, 2] == time]
            edge_check = edges['idx'][:, -1] == time
            edges_t = edges['idx'][edge_check][:, :-1].t().to(self.device)
            edge_attr_t = edges['vals'][edge_check].reshape(-1, 1).to(self.device)
            y = torch.zeros((num_nodes,), device=self.device)
            for label in range(num_labels):
                label_idx = nodes_labels_t[nodes_labels_t[:, 1] == label][:, 0]
                y[label_idx] = label
            data = Data(x=nodes, edge_index=edges_t, edge_attr=edge_attr_t, y=y)
            data_list.append(data)
            if len(data_list) < self.seq_len:
                continue
            else:
                data, slices = self.collate(data_list)
                torch.save((data, slices), fn)
                data_list.pop(0)

    def load_node_feats(self, tar_archive):
        data = load_data_from_tar('elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv',
                                  tar_archive, starting_line=0)
        nodes = data

        nodes_feats = nodes[:, 1:]

        self.num_nodes = len(nodes)
        self.feats_per_node = data.size(1) - 1

        return nodes, nodes_feats.float()

    def load_node_labels(self, tar_archive):
        labels = load_data_from_tar('elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv',
                                    tar_archive, replace_unknow=True).long()
        times = load_data_from_tar('elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv',
                                   tar_archive, replace_unknow=True).long()
        lcols = {'nid': 0, 'label': 1}
        tcols = {'nid':0, 'time':1}

        nodes_labels_times = []
        for i in range(len(labels)):
            label = labels[i, [lcols['label']]].long()
            if label >= 0:
                nid = labels[i, [lcols['nid']]].long()
                time = times[nid, [tcols['time']]].long()
                nodes_labels_times.append([nid, label, time])
        nodes_labels_times = torch.tensor(nodes_labels_times)

        return nodes_labels_times

    def load_transactions(self, tar_archive):
        data = load_data_from_tar('elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist_timed.csv',
                                  tar_archive, type_fn=float, tensor_const=torch.LongTensor)
        tcols = {'source': 0, 'target': 1, 'time': 2}

        data = torch.cat([data, data[:, [1, 0, 2]]])

        self.max_time = data[:,tcols['time']].max()
        self.min_time = data[:,tcols['time']].min()

        return {'idx': data, 'vals': torch.ones(data.size(0))}

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


def load_data_from_tar(file, tar_archive, replace_unknow=False, starting_line=1, sep=',', type_fn=float,
                       tensor_const=torch.DoubleTensor):
    f = tar_archive.extractfile(file)
    lines = f.read()
    lines = lines.decode('utf-8')
    if replace_unknow:
        lines = lines.replace('unknow', '-1')
        lines = lines.replace('-1n', '-1')

    lines = lines.splitlines()

    data = [[type_fn(r) for r in row.split(sep)] for row in lines[starting_line:]]
    data = tensor_const(data)

    return data


if __name__ == '__main__':
    ds = EllipticTemporalDataset()