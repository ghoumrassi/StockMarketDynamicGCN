import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, DenseGCNConv
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
import pandas as pd

from src.models.layers import *


class DGCN(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.conv1 = GCNConv(args.node_feat_dim, args.layer_1_dim)
        self.conv2 = GCNConv(args.layer_1_dim, args.layer_2_dim)
        self.temporal = TemporalLayer(args, device=device)
        self.clf = ClassifierLayer(args)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, data):
        x = data.x
        # x = normalize(data.x)
        edge_attr = data.edge_attr[:, self.args.edgetype].abs()

        batch_size = data.batch.max() + 1
        seq_len = data.seq.max() + 1
        print(x.shape)
        print(data.edge_index.shape)
        print(edge_attr.shape)
        out = self.conv1(x, data.edge_index, edge_weight=edge_attr)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out, data.edge_index, edge_weight=edge_attr)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.temporal(out, data)
        out = self.clf(out)
        return out


class DGCN2(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.temporal = TemporalLayer(args, device=device)
        self.conv1 = DenseGCNConv(args.temporal_out_dim, args.layer_1_dim)
        self.conv2 = DenseGCNConv(args.layer_1_dim, args.layer_2_dim)
        self.clf = ClassifierLayer(args)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, data):
        x = normalize(data.x)
        edge_attr = data.edge_attr[:, self.args.edgetype].abs()

        out = self.temporal(x, data)

        adj = to_dense_adj(data.edge_index, batch=data.batch, edge_attr=edge_attr)
        l = []
        for batch in adj:
            edge_index, edge_attr = dense_to_sparse(batch)
            l.append(to_dense_adj(edge_index, batch=data.seq, edge_attr=edge_attr)[-1])
        adj = torch.stack(l)

        out = self.conv1(out, adj=adj)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out, adj=adj)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.clf(out)

        return out


class DGCNAgg(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.num_edge_types = 3
        self.conv1 = []
        self.conv2 = []
        for i in range(self.num_edge_types):
            self.conv1.append(GCNConv(args.node_feat_dim, args.layer_1_dim).to(self.device))
            self.conv2.append(GCNConv(args.layer_1_dim, args.layer_2_dim).to(self.device))

        self.temporal = TemporalLayer(args, device=device)
        self.clf = ClassifierLayer(args)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, data):
        x = normalize(data.x)

        batch_size = data.batch.max() + 1
        seq_len = data.seq.max() + 1

        out_list = []
        for i in range(self.num_edge_types):
            edge_attr = data.edge_attr[:, i].abs()
            out = self.conv1[i](x, data.edge_index, edge_weight=edge_attr)
            out = F.relu(out)
            out = self.dropout(out)
            out = self.conv2[i](out, data.edge_index, edge_weight=edge_attr)
            out = F.relu(out)
            out = self.dropout(out)
            out, mask = to_dense_batch(out, data.batch.to(self.device))
            out = out.reshape(batch_size, seq_len, -1, self.args.temporal_in_dim)
            out = out.permute(0, 2, 1, 3)
            out = out.reshape(-1, seq_len, self.args.temporal_in_dim)
            out_list.append(out)
        out = torch.cat(out_list, dim=2)
        out = self.temporal(out, data)
        out = self.clf(out)
        return out

def normalize(x):
    means = x.mean(dim=0, keepdim=True)
    stds = x.std(dim=0, keepdim=True)
    return (x - means) / stds