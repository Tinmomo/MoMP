import torch
import torch.nn as nn
import torch.nn.functional as F
# from moe import MoELayer
from layers import MoELayer, MoEFFNLayer
from torch_geometric.nn import MLP, GCNConv, SAGEConv
class GraphMoE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, num_layers = 2, num_experts = 8, topk = 1, batch = 1):
        super(GraphMoE, self).__init__()
        self.dropout = dropout
        self.num_experts = num_experts

        self.linear = nn.Linear(nfeat, nhid, bias=False)
        self.moe_layers = nn.ModuleList()

        for i in range(num_layers):
            # in_channels = nfeat if i == 0 else nhid
            in_channels = nhid
            out_channels = nclass if i == num_layers-1 else nhid
            # out_channels = nhid
            self.moe_layers.append(MoELayer(in_channels, out_channels, dropout, alpha, num_experts, topk, batch))
        
        self.mlp = MLP([nhid, 2*nhid, nclass], dropout=self.dropout, norm="layer_norm")
        self.moe_ffn = MoEFFNLayer(nfeat, nhid)
        self.classifier = nn.Linear(nhid, nclass)
        self.res_rate = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.gcn_conv = SAGEConv(nfeat, nhid)

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.classifier.reset_parameters()
        self.moe_ffn.reset_parameters()
        self.linear.reset_parameters()
        self.gcn_conv.reset_parameters()
        for layer in self.moe_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, two_hop_edge_index = None):
        total_logits = []
        total_indices = []
        gating_logits_list = []
        
        # if self.training:
        #     edge_index, edge_weight = edge_index
        #     random_index = torch.randint(0, edge_index.shape[1], (int(0.5*edge_index.shape[1]), ))
        #     edge_index = [edge_index[:, random_index], edge_weight[random_index]]

        # x = self.leaky_relu(self.linear(x))
    
        x = self.gcn_conv(x, edge_index[0])
        x = F.normalize(x, p=2, dim=1)

        for i, layer in enumerate(self.moe_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x, origin_logits, indices = layer(x, edge_index, two_hop_edge_index)
            # x = F.normalize(x, p=2, dim=1)
            # res_x = torch.concatenate([res_x, x], dim=1)
            total_logits.append(origin_logits)
            total_indices.append(indices)
        # x = self.moe_ffn(x)

        # res_x = torch.concatenate([res_x, x], dim=1)
        # res_x = F.dropout(res_x, self.dropout, training=self.training)
        # x = F.elu(self.classifier(res_x))


        # x = self.res_rate * res_x + (1-self.res_rate) * x
        # x = F.normalize(x, p=2, dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.classifier(x))
        
        # x = self.mlp(x)

        
        # x = F.elu(x)
        return F.log_softmax(x, dim=1), total_logits, total_indices
        # return x, total_logits, total_indices
    
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

    def print_mix_hop_rate(self):
        for layer in self.moe_layers:
            print(layer.mix_hop_rate.item())
    
    def print_similarity(self):
        print(self.get_experts_similarity())
    
    def mask_experts(self, threshold):
        with torch.no_grad():
            for layer in self.moe_layers:
                layer.moe.shared_expert.W.data[layer.moe.shared_expert.W.data < threshold] = 0
                for expert in layer.moe.experts:
                    # mask = expert.W < threshold
                    # expert.W = torch.where(mask, torch.tensor(0.0), expert.W)
                    expert.W.data[expert.W.data < threshold] = 0