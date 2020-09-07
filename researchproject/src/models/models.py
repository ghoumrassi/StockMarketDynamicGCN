import torch
from torch import nn
from src.models import evolvegcn
from src.models.layers import TemporalLayer, ClassifierLayer, GCNLayer


class LSTMModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.temporal = TemporalLayer(args, device)
        self.clf = ClassifierLayer(args)

    def forward(self, data):
        x = data.x[:, 0]
        # x = normalize(x)
        out = self.temporal(x, data)
        out = self.clf(out)
        return out


class LSTMGCN(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.temporal = TemporalLayer(args, device)
        self.conv = GCNLayer(args)
        self.clf = ClassifierLayer(args)

    def forward(self, data):
        x = data.x[:, 0]
        edge_attr = data.edge_attr[:, 2].abs()
        out = self.temporal(x, data)
        # TODO: Need to figure out how to get last graph in sequence (for each batch)
        out = self.conv(out, data.edge_index, edge_attr)
        out = self.clf(out)

        return out


if __name__ == "__main__":
    pass
