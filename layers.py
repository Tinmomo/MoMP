import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from moe import MoE, MoEFFNLayer
from torch_geometric.nn import MLP
import torch.utils.checkpoint as cp


class MoELayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, num_experts = 8, topk = 2, batch = 1):
        super(MoELayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.moe = MoE(in_features, out_features, num_experts, topk)
        self.moe_2hop = MoE(in_features, out_features, num_experts, topk)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.MoEFFNLayer = MoEFFNLayer(in_features, out_features)
        self.outMoEFFNLayer = MoEFFNLayer(out_features, out_features)
        self.mlp = MLP(in_features, out_features, 2*in_features)
        self.mix_hop_rate = nn.Parameter(torch.tensor(0.5))
        self.mix_hop_trans = MLP(in_features, 1)

        # 分batch计算减少显存占用
        self.batch = batch
    
    def reset_parameters(self):
        self.moe.reset_parameters()
        self.moe_2hop.reset_parameters()
        self.linear.reset_parameters()
        self.MoEFFNLayer.reset_parameters()
        self.outMoEFFNLayer.reset_parameters()
        self.mlp.reset_parameters()
        self.mix_hop_trans.reset_parameters()

    def batch_compute(self, h, edge_index, is_2hop=False):
        edge_index, edge_weight = edge_index
        edge_start, edge_end = edge_index
        origin_logits = []
        indices = []
        delta_feature = torch.zeros_like(h)
        for edge_batch, edge_end_batch in zip( zip( torch.chunk(edge_index, self.batch, dim=1), torch.chunk(edge_weight, self.batch) ), torch.chunk(edge_end, self.batch)):
            # aggreate_feature_batch, origin_logits_batch, indices_batch = self.moe(h, edge_batch)
            if is_2hop:
                aggreate_feature_batch, origin_logits_batch, indices_batch = cp.checkpoint(self.moe_2hop, h, edge_batch)
            else:
                aggreate_feature_batch, origin_logits_batch, indices_batch = cp.checkpoint(self.moe, h, edge_batch)
            # aggreate_feature.append(aggreate_feature_batch)
            origin_logits.append(origin_logits_batch)
            indices.append(indices_batch)
            delta_feature = torch.index_add(delta_feature, dim=0, index=edge_end_batch, source=aggreate_feature_batch)
        return delta_feature, origin_logits, indices

    def forward(self, h, edge_index, two_hop_edge_index = None):
        # (e, d)
        # 使用MOE计算出每一条边传递的信息
        aggreate_feature = []
        origin_logits = []
        indices = []
        
        # edge_index, edge_weight = edge_index
        # edge_start, edge_end = edge_index

        # h = self.linear(h)
        # h = self.MoEFFNLayer(h)
        # h = nn.functional.normalize(h, p=2, dim=1)

        # delta_feature = torch.zeros_like(h)
        # for edge_batch, edge_end_batch in zip( zip( torch.chunk(edge_index, self.batch, dim=1), torch.chunk(edge_weight, self.batch) ), torch.chunk(edge_end, self.batch)):
        #     # aggreate_feature_batch, origin_logits_batch, indices_batch = self.moe(h, edge_batch)
        #     aggreate_feature_batch, origin_logits_batch, indices_batch = cp.checkpoint(self.moe, h, edge_batch)
        #     # aggreate_feature.append(aggreate_feature_batch)
        #     origin_logits.append(origin_logits_batch)
        #     indices.append(indices_batch)
        #     delta_feature = torch.index_add(delta_feature, dim=0, index=edge_end_batch, source=aggreate_feature_batch)
        #     # del aggreate_feature_batch
        delta_feature, origin_logits, indices = self.batch_compute(h, edge_index)
        
        if two_hop_edge_index is not None:
            # two_hop_edge_index, two_hop_edge_weight = two_hop_edge_index
            # two_hop_edge_start, two_hop_edge_end = edge_index
            # delta_feature_two_hop = torch.zeros_like(h)
            # for edge_batch, edge_end_batch in zip( zip( torch.chunk(two_hop_edge_index, self.batch, dim=1), torch.chunk(two_hop_edge_weight, self.batch) ), torch.chunk(two_hop_edge_end, self.batch)):
            #     # aggreate_feature_batch, origin_logits_batch, indices_batch = self.moe(h, edge_batch)
            #     aggreate_feature_batch, origin_logits_batch, indices_batch = cp.checkpoint(self.moe, h, edge_batch)
            #     # aggreate_feature.append(aggreate_feature_batch)
            #     origin_logits.append(origin_logits_batch)
            #     indices.append(indices_batch)
            #     delta_feature_two_hop = torch.index_add(delta_feature_two_hop, dim=0, index=edge_end_batch, source=aggreate_feature_batch)
            #     # del aggreate_feature_batch

            delta_feature_two_hop, origin_logits_two_hop, indices_two_hop = self.batch_compute(h, two_hop_edge_index)
            mix_hop_rate = F.sigmoid(self.mix_hop_trans(h))
            delta_feature = (1-mix_hop_rate) * delta_feature + mix_hop_rate * delta_feature_two_hop
            # delta_feature = (1-self.mix_hop_rate) * delta_feature + self.mix_hop_rate * delta_feature_two_hop
            
        h = torch.add(h, delta_feature)
        # h = delta_feature
        # h = self.leakyrelu(self.linear(h))
        h = self.leakyrelu(self.linear(h))

        # h = torch.cat([h, delta_feature], dim=1)
        # h = self.leakyrelu(self.linear(h))

        # h = self.leakyrelu(self.MoEFFNLayer(h))
        # h = self.mlp(h)+h
        # h = F.elu(self.linear(h))
        
        # h = self.leakyrelu(h)
        # h = nn.functional.normalize(h, p=2, dim=1)
        return h, origin_logits, indices

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 128, num_layers = 3, activate = "leaky_relu", dropout = 0.6, alpha = 0.2):
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

# if __name__ == "__main__":
#     N = 1000
#     E = 20000
#     input_dim = 100
#     output_dim = 8
#     num_experts = 8
#     topk = 2
#     drop_out = 0.2
#     alpha = 0.1
#     xixi = torch.rand(N, input_dim)
#     edge_index = torch.randint(N, (2, E))
#     model = MoELayer(input_dim, output_dim, drop_out, alpha)
#     haha = model(xixi, edge_index)
#     # print(haha.shape)
#     pass