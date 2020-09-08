import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch
from src.models import evolvegcn


class ClassifierLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(in_features=args.clf_in_dim, out_features=args.fc_1_dim)
        self.fc2 = nn.Linear(in_features=args.fc_1_dim, out_features=args.fc_2_dim)
        self.fc3 = nn.Linear(in_features=args.fc_2_dim, out_features=args.out_dim)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = torch.softmax(out, dim=1)
        return out


class TemporalLayer(nn.Module):
    def __init__(self, args, device, reshape=True):
        super().__init__()
        self.args = args
        self.device = device
        self.reshape = reshape
        if args.temporal_layer == 'lstm':
            self.temporal = nn.LSTM(args.temporal_in_dim, args.temporal_out_dim, num_layers=args.temporal_num_layers,
                                    batch_first=True)
        elif args.temporal_layer == 'gru':
            self.temporal = nn.GRU(args.temporal_in_dim, args.temporal_out_dim, num_layers=args.temporal_num_layers,
                                   batch_first=True)
        elif args.temporal_layer == 'cnn':
            self.temporal = nn.ModuleList(
                [
                    nn.Conv2d(in_channels=1, out_channels=args.num_filters, kernel_size=(fs, args.temporal_in_dim))
                    for fs in args.filter_sizes
                ]
            )
        else:
            raise NotImplementedError("Only lstm, gru or cnn.")

        self.dropout = nn.Dropout(args.dropout)

    def init_hidden_lstm(self, batch_size):
        return (
            Variable(
                torch.randn((self.args.temporal_num_layers, batch_size, self.args.temporal_out_dim))
            ).to(self.device),
            Variable(
                torch.randn((self.args.temporal_num_layers, batch_size, self.args.temporal_out_dim))
            ).to(self.device)
        )

    def forward(self, x, data):
        if self.reshape:
            batch_size = data.batch.max() + 1
            seq_len = data.seq.max() + 1
            x = x.reshape(batch_size, seq_len, -1, self.args.temporal_in_dim)
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(-1, seq_len, self.args.temporal_in_dim)

        if self.args.temporal_layer == 'lstm':
            self.hidden = self.init_hidden_lstm(x.shape[0])
            out, _ = self.temporal(x, self.hidden)
            out = out[:, -1, :]
        elif self.args.temporal_layer == 'gru':
            out, _ = self.temporal(x)
            out = out[:, -1, :]
        elif self.args.temporal_layer == 'cnn':
            out = x.unsqueeze(1)
            conved = [F.relu(conv(out)).squeeze(3) for conv in self.temporal]
            pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
            out = torch.cat(pooled, dim=1)
        else:
            raise NotImplementedError()

        out = self.dropout(out)
        return out


class GCNLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1 = GCNConv(args.gcn_in_dim, args.gcn_1_dim)
        self.conv2 = GCNConv(args.gcn_1_dim, args.gcn_out_dim)

    def forward(self, x, edge_idx, edge_attr):
        out = self.conv1(x, edge_idx, edge_weight=edge_attr)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out, edge_idx, edge_weight=edge_attr)
        out = F.relu(out)
        out = self.dropout(out)
