import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch

from src.models.models import NodePredictionModel

class DGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1 = GCNConv(args.node_feat_dim, args.layer_1_dim)
        self.conv2 = GCNConv(args.layer_1_dim, args.layer_2_dim)
        self.lstm = nn.LSTM(args.layer_2_dim, args.lstm_dim, args.num_layers, batch_first=True)
        self.fc1 = nn.Linear(in_features=args.lstm_dim, out_features=args.fc_1_dim)
        self.fc2 = nn.Linear(in_features=args.fc_1_dim, out_features=args.fc_2_dim)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, data):
        batch_size = data.batch.max() + 1
        seq_len = data.seq.max() + 1
        out = self.conv1(data.x, data.edge_index, edge_weight=data.edge_attr[:, self.args.edgetype])
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out, data.edge_index, edge_weight=data.edge_attr[:, self.args.edgetype])
        out = F.relu(out)
        out = self.dropout(out)
        out, mask = to_dense_batch(out, data.batch)
        out = out.reshape(batch_size, seq_len, -1, self.args.lstm_dim)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(-1, seq_len, self.args.lstm_dim)
        out, _ = self.lstm(out)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = torch.softmax(out, dim=1)
        return out.reshape(batch_size, -1, self.args.fc_2_dim)
