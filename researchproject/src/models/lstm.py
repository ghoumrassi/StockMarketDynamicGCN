from src.models.models import NodePredictionModel

from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lstm = nn.LSTM(args.lstm_input_size, args.layer_2_dim, num_layers=args.num_layers)
        self.clf = NodePredictionModel(args)

    def forward(self, data):
        lstm_out, _ = self.lstm(data.x)
        clf_out = self.clf(lstm_out[-1])
        return clf_out