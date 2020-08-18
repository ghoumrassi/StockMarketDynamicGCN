import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

from src.models.models import NodePredictionModel

class DGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = GCNConv(args.node_feat_dim, args.layer_1_dim)
        self.conv2 = GCNConv(args.layer_1_dim, args.layer_2_dim)
        self.lstm = nn.LSTM(args.layer_2_dim, args.lstm_dim, args.num_layers)
        self.fc1 = nn.Linear(in_features=args.lstm_dim, out_features=args.fc_1_dim)
        self.fc2 = nn.Linear(in_features=args.fc_1_dim, out_features=args.fc_2_dim)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, A, X_list):
        cell_state = None
        edge_index_list = A[0]
        edge_weight_list = A[1]
        for i in range(len(X_list)):
            X = X_list[i]
            edge_index = edge_index_list[i]
            edge_weight = edge_weight_list[i]
            conv_1_out = self.conv1(X, edge_index, edge_weight=edge_weight)
            relu_1_out = F.relu(conv_1_out)
            drp_1_out = self.dropout(relu_1_out)
            conv_2_out = self.conv2(drp_1_out, edge_index, edge_weight=edge_weight)
            relu_2_out = F.relu(conv_2_out)
            drp_2_out = self.dropout(relu_2_out)
            if cell_state:
                lstm_out, cell_state = self.lstm(drp_2_out.view(drp_2_out.shape[0], 1, drp_2_out.shape[1]), cell_state)
            else:
                lstm_out, cell_state = self.lstm(drp_2_out.view(drp_2_out.shape[0], 1, drp_2_out.shape[1]))

        drp_3_out = self.dropout(lstm_out.view(lstm_out.shape[0], lstm_out.shape[2]))
        fc1_out = self.fc1(drp_3_out)
        relu_out = torch.relu(fc1_out)
        drp_4_out = self.dropout(relu_out)
        fc2_out = self.fc2(drp_4_out)
        out = torch.softmax(fc2_out, dim=1)
        return out
