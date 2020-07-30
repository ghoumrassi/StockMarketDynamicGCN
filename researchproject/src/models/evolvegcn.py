"""
Copyright 2019 IBM
"""

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math

from src.models.utils import Namespace
class EvolveGCN(nn.Module):
    """
    Implementation of the EvolveGCN model for learning on dynamically evolving graphs.

    Original paper: https://arxiv.org/abs/1902.10191
    Original implementation: https://github.com/IBM/EvolveGCN
    """
    def __init__(self, args, activation, skipfeats=False):
        super().__init__()
        GRCU_args = Namespace({})

        features = [
            args.node_feat_dim,
            args.layer_1_dim,
            args.layer_2_dim
        ]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Adds a skip connection that concatenates together the node features and layer output as input features for the
        # next layer.
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()

        # Makes GRCU_layers: a list of the 3 GRCU layers for each GCN cell
        for i in range(1, len(features)):
            GRCU_args = Namespace({
                'in_feats': features[i - 1],
                'out_feats': features[i],
                'activation': activation
            })

            grcu_i = GRCU(GRCU_args).to(self.device)
            self.GRCU_layers.append(grcu_i)
            self._parameters.extend(list(grcu_i.parameters()))

    def forward(self, A_list, X_list, node_mask_list):
        """
        Inputs -
            A_list:             list of Adj matrices at each timestep t
            X_list:             list of feature matrices at each timestep t
            node_mask_list:     ???
        """
        node_features = X_list[-1]

        for unit in self.GRCU_layers:
            X_list = unit(A_list, X_list, node_mask_list)

        out = X_list[-1]
        if self.skipfeats:
            out = torch.cat((out, node_features), dim=1)
        return out

    def parameters(self):
        return self._parameters


class GRCU(nn.Module):
    """ Underlying Graph Recurrent Convolution Units for the EvolveGCN layer. """
    def __init__(self, args):
        super().__init__()
        self.args = args
        cell_args = Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = Mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(
            self.args.in_feats,
            self.args.out_feats
        ))
        self.reset_params(self.GCN_init_weights)

    def reset_params(self, p):
        # Standardise initial parameters based on initial feature size
        std = 1. / math.sqrt(p.size(1))
        p.data.uniform_(-std, std)

    def forward(self, A_list, emb_list, node_mask_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, Ahat in enumerate(A_list):
            node_embedding = emb_list[t]
            GCN_weights = self.evolve_weights(GCN_weights, node_embedding, node_mask_list[t])

            # H_t_(l+1) = act( Ahat_t_l * H_t_l * W_t_l )
            node_embedding = self.activation(
                Ahat.matmul(
                    node_embedding.matmul(
                        GCN_weights
                    )
                )
            )

            out_seq.append(node_embedding)

        return out_seq


class Mat_GRU_cell(nn.Module):
    """
    GRU cell, with the following adjustments:
        1. Input is a matrix, not a vector
        2. Hidden state H is adjusted such that it's feature dimensionality is the same as the input
           [i.e. X.size(1) = H.size(1)]
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update = Mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.sigmoid)

        self.reset = Mat_GRU_gate(args.rows,
                                  args.cols,
                                  torch.sigmoid)

        self.h_tilda = Mat_GRU_gate(args.rows,
                                    args.cols,
                                    nn.Tanh())

        self.choose_topk = TopK(features=args.rows, k=args.cols)


    def forward(self, X, prev_H, mask):
        """
        Takes the input feature matrix and the hidden state of the layer at the last timestep
        and returns the hidden state at this timestep.
        """
        H_topk = self.choose_topk(prev_H, mask)

        update = self.update(H_topk, X)
        reset = self.reset(H_topk, X)

        H_cap = reset * X
        H_cap = self.h_tilda(H_topk, H_cap)

        H = (1 - update) * X + update * H_cap

        return H


class Mat_GRU_gate(nn.Module):
    """ Gate for GRU cell. """
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation

        self.W = Parameter(torch.Tensor(rows, rows))
        self.reset_params(self.W)

        self.U = Parameter(torch.Tensor(rows, rows))
        self.reset_params(self.U)

        self.b = Parameter(torch.zeros(rows, cols))

    def reset_params(self, p):
        std = 1. / math.sqrt(p.size(1))
        p.data.uniform_(-std, std)

    def forward(self, x, h):
        out = self.activation(
            self.W.matmul(x) + \
            self.U.matmul(h) + \
            self.b
        )
        return out


class TopK(nn.Module):
    """
    Creates a lower dimensional representation of the input feature matrix, with only k-features.
    This is necessary to preserve the input and output features are equal in length for the EGRC unit.
    """
    def __init__(self, features, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(features, 1))
        self.reset_params(self.scorer)

        self.k = k

    def reset_params(self, p):
        std = 1. / math.sqrt(p.size(0))
        p.data.uniform_(-std, std)

    def forward(self, embeddings, mask):
        # y_t = (X_t * p) / ||p||
        scores = embeddings.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        values, topk_indices = scores.view(-1).topk(self.k)

        # Handles case where many values are nan
        topk_indices = topk_indices[values > -float("Inf")]
        if topk_indices.size(0) < self.k:
            topk_indices = pad_with_lst_val(topk_indices, self.k)

        # Handles case where input tensor is sparse
        if isinstance(embeddings, torch.sparse.FloatTensor) or \
                isinstance(embeddings, torch.cuda.sparse.FloatTensor):
            embeddings = embeddings.to_dense()

        # out = [X_t * tanh(y_t)]_i_t
        out = embeddings[topk_indices] * nn.Tanh(scores[topk_indices].view(-1, 1))

        return out.t()
