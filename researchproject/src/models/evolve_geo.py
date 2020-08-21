import torch
from torch.nn import GRU
from torch_geometric.nn import GCNConv
from torch_geometric.nn import TopKPooling


class EvolveGCN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_of_nodes = num_of_nodes

        self.ratio = self.in_channels / self.num_of_nodes

        self.pooling_layer = TopKPooling(self.input_dim, self.ratio)

        self.gru = GRU(input_size = args.input_dim,
                       hidden_size = args.input_dim,
                       num_layers = 1)

        self.conv = GCNConv(args.node_feat_dim, args.layer_1_dim)
        self.conv_layer = GCNConv(in_channels = args.input_dim,
                                  out_channels = args.input_dim,
                                  bias = False)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor=None) -> torch.FloatTensor:
        """
        Making a forward pass.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node embedding.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.

        Return types:
            * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
        """
        X_tilde = self.pooling_layer(X, edge_index)
        X_tilde = X_tilde[0][None, :, :]
        W = self.conv_layer.weight[None, :, :]
        X_tilde, W = self.gru(X_tilde, W)
        self.conv_layer.weight = torch.nn.Parameter(W.squeeze())
        X = self.conv_layer(X, edge_index, edge_weight)
        return X