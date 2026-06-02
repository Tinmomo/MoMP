import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MoELayer
from torch_geometric.nn import MLP


class GraphMoE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, num_layers=2, num_experts=8, topk=1, batch=1, bn=None, use_classifier=False):
        super(GraphMoE, self).__init__()
        self.dropout = dropout
        self.num_experts = num_experts

        self.linear = nn.Linear(nfeat, nhid, bias=False)
        self.moe_layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.use_classifier = use_classifier

        for i in range(num_layers):
            in_channels = nhid
            if use_classifier:
                out_channels = nhid
            else:
                out_channels = nclass if i == num_layers - 1 else nhid
            self.moe_layers.append(MoELayer(in_channels, out_channels, dropout, alpha, num_experts, topk, batch))
            if bn == 'LayerNorm':
                self.bns.append(nn.LayerNorm(normalized_shape=out_channels))
            else:
                self.bns.append(nn.Identity())

        self.mlp = MLP([nhid, 2 * nhid, nclass], act='elu', dropout=self.dropout, norm="layer_norm")
        self.leaky_relu = nn.LeakyReLU(alpha)

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.linear.reset_parameters()
        for layer in self.moe_layers:
            layer.reset_parameters()
        for layer in self.bns:
            if not isinstance(layer, nn.Identity):
                layer.reset_parameters()

    def forward(self, x, edge_index, return_logits=False):
        total_logits = []
        total_indices = []

        x = self.leaky_relu(self.linear(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.normalize(x, p=2, dim=1)

        for moe_layer, bn in zip(self.moe_layers[:-1], self.bns[:-1]):
            if return_logits:
                x, origin_logits, indices = moe_layer(x, edge_index)
                total_logits.append(origin_logits)
                total_indices.append(indices)
            else:
                x = moe_layer(x, edge_index)
            x = bn(x)
            x = self.leaky_relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        if return_logits:
            x, origin_logits, indices = self.moe_layers[-1](x, edge_index)
            total_logits.append(origin_logits)
            total_indices.append(indices)
        else:
            x = self.moe_layers[-1](x, edge_index)

        if self.use_classifier:
            x = self.bns[-1](x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.mlp(x)

        return x, total_logits, total_indices

    def get_experts_similarity(self):
        similarity = []
        for layer in self.moe_layers:
            experts = [layer.moe.shared_expert.W]
            for i, expert in enumerate(layer.moe.experts):
                if i < layer.moe.freeze_experts:
                    continue
                experts.append(expert.W)
            experts = torch.cat(experts, dim=0)
            experts_norm = experts / torch.norm(experts, dim=1, keepdim=True)
            similarity.append(torch.mm(experts_norm, experts_norm.T))
        return similarity

    def print_experts(self):
        for layer in self.moe_layers:
            print("Shared Experts:")
            print(layer.moe.shared_expert.W)
            for index, expert in enumerate(layer.moe.experts):
                print(index, expert.W)
        similarity = self.get_experts_similarity()
        print(similarity)

    def print_similarity(self):
        print(self.get_experts_similarity())

    def mask_experts(self, threshold):
        with torch.no_grad():
            for layer in self.moe_layers:
                layer.moe.shared_expert.W.data[layer.moe.shared_expert.W.data < threshold] = 0
                for expert in layer.moe.experts:
                    expert.W.data[expert.W.data < threshold] = 0
