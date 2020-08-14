import torch
from torch import nn
from src.models import evolvegcn


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
        out = torch.softmax(fc2_out, dim=1)
        return out


if __name__ == "__main__":
    pass
