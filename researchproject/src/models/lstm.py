from torch_geometric.utils import to_dense_batch
from src.models.layers import TemporalLayer, ClassifierLayer
from src.models.utils import normalize

from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.temporal = TemporalLayer(args, device)
        self.clf = ClassifierLayer(args)

    def forward(self, data):
        x = data.x[:, 0]
        x = normalize(x)
        out = self.temporal(x, data)
        out = self.clf(out)
        return out