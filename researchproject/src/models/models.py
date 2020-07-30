import torch
from torch import nn
from src.models import evolvegcn


class ModelA_EvolveGCN_w_Dense(nn.Module):
    def __init__(self, args, activation, skipfeats=False):
        super().__init__()
        self.egcn = evolvegcn.EvolveGCN(args, activation, skipfeats=skipfeats)
        self.fc1 = nn.Linear(in_features=args.layer_2_dim, out_features=args.fc_1_dim)
        self.fc2 = nn.Linear(in_features=args.fc_1_dim, out_features=args.fc_2_dim)
        self.dropout = nn.Dropout(args.dropout)

        self._parameters = nn.ParameterList()
        self._parameters.extend(list(self.egcn.parameters()))
        self._parameters.extend(list(self.fc1.parameters()))
        self._parameters.extend(list(self.fc2.parameters()))

    def forward(self, a, x, n_mask):
        egcn_out = self.egcn(a, x, n_mask)
        dp_1_out = self.dropout(egcn_out)
        fc_1_out = self.fc1(dp_1_out)
        act_1_out = torch.sigmoid(fc_1_out)
        dp_2_out = self.dropout(act_1_out)
        fc_2_out = self.fc1(dp_2_out)
        act_2_out = torch.sigmoid(fc_2_out)
        return act_2_out

    def parameters(self):
        return self._parameters


if __name__ == "__main__":
    pass
