import torch
from torch import nn
from src.models import evolvegcn


# class EvolveGCNDenseModel(nn.Module):
#     def __init__(self, args, activation, skipfeats=False, predict_periods=3):
#         super().__init__()
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         self.predict_periods = predict_periods
#
#         self.egcn = evolvegcn.EvolveGCN(args, activation, skipfeats=skipfeats)
#
#         self.lstm = nn.LSTM(input_size=args.layer_2_dim, hidden_size=args.layer_2_dim)
#         self.fc = nn.Linear(in_features=args.layer_2_dim, out_features=3)
#         self.dropout = nn.Dropout(args.dropout)
#         self.params = nn.ParameterList(list(self.egcn.parameters()) + list(self.lstm.parameters())
#                                        + list(self.fc.parameters())
#                                        )
#         # self._parameters = nn.ParameterList().cuda(device=self.device)
#         # self._parameters.extend(list(self.egcn.parameters()))
#         # self._parameters.extend(list(self.lstm.parameters()))
#
#     def forward(self, a, x, n_mask, n_idx):
#         egcn_out = self.egcn(a, x, n_mask)
#         dp_1_out = self.dropout(egcn_out)
#
#         bs = 100  # batch size
#         pred = []
#         for i in range(1 + (n_idx.size(1) // bs)):
#             clf_in = self.get_node_embs(egcn_out, n_idx[:, i*bs: (i+1)*bs])
#             pred.append(self.classifier(clf_in))
#
#         return act_out
class NodePredictionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(in_features=args.layer_2_dim, out_features=args.fc_1_dim)
        self.fc2 = nn.Linear(in_features=args.fc_1_dim, out_features=args.fc_2_dim)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        dr1_out = self.dropout(x)
        fc1_out = self.fc1(dr1_out)
        relu_out = torch.relu(fc1_out)
        dr1_out = self.dropout(relu_out)
        fc2_out = self.fc2(dr1_out)
        return fc2_out


if __name__ == "__main__":
    pass
