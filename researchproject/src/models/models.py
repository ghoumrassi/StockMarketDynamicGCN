import torch
from torch import nn
from src.models import evolvegcn


class EvolveGCNDenseModel(nn.Module):
    def __init__(self, args, activation, skipfeats=False, predict_periods=3):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.predict_periods = predict_periods

        self.egcn = evolvegcn.EvolveGCN(args, activation, skipfeats=skipfeats)

        self.lstm = nn.LSTM(input_size=args.layer_2_dim, hidden_size=args.layer_2_dim)
        self.fc = nn.Linear(in_features=args.layer_2_dim, out_features=3)
        self.dropout = nn.Dropout(args.dropout)

        self._parameters = nn.ParameterList()
        self._parameters.extend(list(self.egcn.parameters()))
        self._parameters.extend(list(self.lstm.parameters()))

    def forward(self, a, x, n_mask):
        egcn_out = self.egcn(a, x, n_mask)
        dp_1_out = self.dropout(egcn_out)

        # out_seq = torch.zeros((self.predict_periods, 3), device=self.device)
        lstm_in = dp_1_out.view(1, *dp_1_out.shape)
        lstm_out_list = []
        for i in range(self.predict_periods):
            if i == 0:
                lstm_in, (h, c) = self.lstm(lstm_in)
            else:
                lstm_in, (h, c) = self.lstm(lstm_in, (h, c))
            lstm_out_list.append(lstm_in)
        lstm_out = torch.cat(lstm_out_list)
        fc_out = self.fc(lstm_out)
        act_out = torch.relu(fc_out)

        return act_out

    def parameters(self):
        return self._parameters


if __name__ == "__main__":
    pass
