import torch
from torch_geometric.nn import GCNConv


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = []
        for i in range(3):
            self.conv.append(GCNConv(4, 16))

    def forward(self, data):
        out_list = []
        for i in range(self.num_edge_types):
            out = self.conv[i](data.x, data.edge_index, edge_weight=data.edge_attr[:, 0])
            out_list.append(out)
        return torch.cat(out_list, dim=1)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = Model().to(device)
assert model.conv[0].weight.is_cuda()