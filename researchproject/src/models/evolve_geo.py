import torch
from torch.nn import GRU
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse

class Evolve(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pooling_layer = TopKPooling(args.node_feat_dim, args.ratio)

        self.gru1 = GRU(args.node_feat_dim, args.node_feat_dim, num_layers=1, batch_first=True)
        self.conv1 = GCNConv(args.node_feat_dim, args.node_feat_dim)

        self.gru2 = GRU(args.layer_1_dim, args.layer_1_dim, num_layers=1, batch_first=True)
        self.conv2 = GCNConv(args.layer_1_dim, args.layer_2_dim)


    def forward(self, data):
        x, _ = to_dense_batch(data.x, data.batch)
        adj = to_dense_adj(data.edge_index, batch=data.batch, edge_attr=data.edge_attr)

        data_batches = []
        for i in range(x.shape[0]):
            edge_index, edge_attr = dense_to_sparse(adj[i, :, :, self.args.edgetype])
            X_tilde = self.pooling_layer(x[i], edge_index, edge_attr=edge_attr)
            X_tilde = X_tilde[0][None, :, :]
            W = self.conv1.weight[None, :, :]
            X_tilde, W = self.gru1(X_tilde, W)
            self.conv_layer.weight = torch.nn.Parameter(W.squeeze())
            out = self.conv1(data.x, data.edge_index, data.edge_weight)

            W = self.conv2.weight[None, :, :]
            out, W = self.gru2(out, W)
            self.conv2.weight = torch.nn.Parameter(W.squeeze())
            out = self.conv2(out, data.edge_index, data.edge_weight)
            data_batches.append(out)

        return torch.stack(data_batches)