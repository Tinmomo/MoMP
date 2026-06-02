import torch
import torch.nn as nn
import torch.nn.functional as F
from moe import MoE


class MoELayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, num_experts=8, topk=2, batch=1):
        super(MoELayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.moe = MoE(in_features, out_features, num_experts, topk)
        self.linear = nn.Linear(in_features, out_features, bias=False)

        # 分 batch 计算减少显存占用
        self.batch = batch

    def reset_parameters(self):
        self.moe.reset_parameters()
        self.linear.reset_parameters()

    def batch_compute(self, h, edge_index, return_logits=False):
        origin_logits = []
        indices = []
        delta_feature = torch.zeros_like(h)
        if return_logits:
            aggreate_feature, origin_logits_batch, indices_batch = self.moe(h, edge_index, return_logits)
            origin_logits.append(origin_logits_batch)
            indices.append(indices_batch)
        else:
            aggreate_feature = self.moe(h, edge_index)
        edge_index, edge_weight = edge_index
        edge_start, edge_end = edge_index
        delta_feature = torch.index_add(delta_feature, dim=0, index=edge_end, source=aggreate_feature)
        if return_logits:
            return delta_feature, origin_logits, indices
        else:
            return delta_feature

    def forward(self, h, edge_index, return_logits=False):
        # (e, d) 使用 MoE 计算每一条边传递的信息
        origin_logits = []
        indices = []

        if return_logits:
            delta_feature, origin_logits, indices = self.batch_compute(h, edge_index)
        else:
            delta_feature = self.batch_compute(h, edge_index)

        delta_feature = self.linear(delta_feature)
        h = torch.add(h, delta_feature)

        if return_logits:
            return h, origin_logits, indices
        else:
            return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, activate="leaky_relu", dropout=0.6, alpha=0.2):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(nn.Linear(in_dim, out_dim))
            if activate == "leaky_relu":
                self.layers.append(nn.LeakyReLU(alpha))
            else:
                self.layers.append(nn.ReLU())
            self.layers.append(nn.LayerNorm(out_dim))

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x):
        for layer in self.layers:
            x = F.dropout(x, self.dropout)
            x = layer(x)
        return x
