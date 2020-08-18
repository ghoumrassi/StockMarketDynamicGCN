import dgl
import math
import torch
from torch import nn

from src.models.models import NodePredictionModel


class EvolveGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.grcu_1 = GRCU(args).to(self.device)
        self.grcu_2 = GRCU(args).to(self.device)
        self.clf = NodePredictionModel(args).to(self.device)

    def forward(self, A_list, X_list, node_mask_list):
        # X last observed
        node_features = X_list[-1]
        grcu_1_out = self.grcu_1(A_list, X_list, node_mask_list)
        grcu_2_out = self.grcu_2(A_list, grcu_1_out, node_mask_list)
        out = self.clf(grcu_2_out)
        return out


class GRCU(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.evolve_weights = MatGRUCell(args)
        self.init_weights = nn.Parameter(args.in, args.out)
        self.reset_params(self.init_weights)

    def forward(self, A_list, emb_list, node_mask_list):
        W = self.init_weights
        out = []
        for t in range(A_list):
            A = A_list[t]
            node_embedding = emb_list[t]

            # Update weights
            W = self.evolve_weights(A, node_embedding, node_mask_list[t])

            node_embedding = torch.relu(A @ node_embedding @ W)

            out.append(node_embedding)

class MatGRUCell(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.update = MatGRUGate(args.rows, args.cols, torch.sigmoid)
        self.reset = MatGRUGate(args.rows, args.cols, torch.sigmoid)
        self.h_tilda = MatGRUGate(args.rows, args.cols, torch.tanh)
        self.choose_topk = TopK(features=args.rows, k=args.cols)

    def forward(self, X, prev_H, mask):
        H_topk = self.choose_topk(prev_H, mask)

        update = self.update(H_topk, X)
        reset = self.reset(H_topk, X)

        H_cap = reset * X
        H_cap = self.h_tilda(H_topk, H_cap)
        H = (1 - update) * X + update * H_cap

        return H

class MatGRUGate(nn.Module):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation

        self.W = nn.Parameter(torch.Tensor(rows, rows))
        self.reset_params(self.W)
        self.U = nn.Parameter(torch.Tensor(rows, rows))
        self.reset_params(self.U)
        self.b = nn.Parameter(torch.zeros(rows, cols))

    def reset_params(self, p, dim):
        std = 1. / math.sqrt(p.size(dim))
        p.data.uniform_(-std, std)

    def forward(self, X, H):
        out = self.activation(self.W @ X + self.U @ H + self.b)
        return out

class TopK(nn.Module):
    def __init__(self, features, k):
        super().__init__()
        self.scorer = nn.Parameter(torch.Tensor(features, 1))
        reset_params(self.scorer, 0)
        self.k = k

    def forward(self, embeddings, mask):
        scores = embeddings.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        values, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[values > -float("Inf")]
        if topk_indices.size(0) < self.k:
            topk_indices = pad_with_last_val(topk_indices, self.k)

        if isinstance(embeddings, torch.sparse.FloatTensor) or \
                isinstance(embeddings, torch.cuda.sparse.FloatTensor):
            embeddings = embeddings.to_dense()

        tanh = nn.Tanh()
        out = embeddings[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        return out.t()


def reset_params(p, dim):
    # Standardise initial parameters based on initial feature size
    std = 1. / math.sqrt(p.size(dim))
    p.data.uniform_(-std, std)