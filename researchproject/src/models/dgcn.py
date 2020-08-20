import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

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

    def forward(self, inputs):
        data, mask = inputs
        batch_size = len(mask['x'])
        out = self.conv1(data.x, data.edge_index, edge_weight=data.edge_attr[:, 0])
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out, data.edge_index, edge_weight=data.edge_attr[:, 0])
        out = F.relu(out)
        out = self.dropout(out)

        emb_dim = out.shape[-1]

        # Output transformed in form (batch, seq_len, features)
        # new_shape = [batch_size * tickers, seq_len, emb_dim]
        new_shape = (-1, self.args.seq_length, self.args.layer_2_dim)
        out, _ = self.lstm(out.view(*new_shape))  # Figure this piece of the puzzle out and problem solved!
        out = out[:, -1, :].view(batch_size, -1, self.args.lstm_dim)
        out = self.dropout(out)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = torch.softmax(out, dim=1)
        return out
