import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch_scatter import scatter_softmax
from torch_geometric.nn import GATConv
import numpy as np

xixi = GATConv(12, 21)

class DotExpert(nn.Module): 
    def __init__(self, input_dim, output_dim, p = 0.6): 
        super(DotExpert, self).__init__() 
        self.W = nn.Parameter(torch.empty((1, input_dim)))
        nn.init.uniform_(self.W)
        # nn.init.ones_(self.W)
        self.p = p

    def forward(self, x, edge_weight): 
        # x = F.dropout(x, self.p, training=self.training)
        return torch.mul(edge_weight, self.W * x + 1e-5)
        # return F.leaky_relu(self.W * x)
        # return torch.mul(edge_weight, x)
    
    def reset_parameters(self):
        nn.init.uniform_(self.W)
        # nn.init.ones_(self.W)

class LinearExpert(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        super(LinearExpert, self).__init__() 
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x, edge_weight): 
        return self.linear(x)
    
    def reset_parameters(self):
        self.linear.reset_parameters()
        # nn.init.ones_(self.W)

# Define the gating model 
class Gating(nn.Module): 
    def __init__(self, input_dim: int, num_experts: int, topk=2, u = 0.1): 
        super(Gating, self).__init__() 
        # Layers 
        self.down_proj_start = nn.Linear(input_dim, input_dim)
        self.down_proj_end = nn.Linear(input_dim, input_dim)
        self.gating_linear = nn.Linear(input_dim, num_experts, bias=False)
        self.topk = topk
        self.num_experts = num_experts
        self.bias = torch.zeros(num_experts).cuda(1)
        self.u = u
    
    def reset_parameters(self):
        # nn.init.xavier_normal_(self.down_proj.weight)
        # nn.init.zeros_(self.down_proj.bias)
        self.down_proj_end.reset_parameters()
        self.down_proj_start.reset_parameters()
        nn.init.xavier_normal_(self.gating_linear.weight)
        nn.init.zeros_(self.bias)


    def forward(self, x, edge_index):
        edge_start, edge_end = edge_index

        # 优化后的门控计算
        gating_value_start = self.gating_linear(self.down_proj_start(x))
        gating_value_end = self.gating_linear(self.down_proj_end(x))
        origin_logits = gating_value_start[edge_start] + gating_value_end[edge_end]

        # 原始门控计算
        # gating_value = torch.cat((x[edge_start], x[edge_end]), dim=1)
        # gating_value = self.down_proj(gating_value)
        # origin_logits = self.gating_linear(gating_value)

        # x_norm = torch.norm(x, dim=1)+1e-6
        # expert_norm = torch.norm(self.gating_linear.weight, dim=1)+1e-6
        # origin_logits = torch.div(origin_logits, x_norm.unsqueeze(1))
        # origin_logits = torch.div(origin_logits, expert_norm.unsqueeze(0))
        # 根据门控分数选择topk个专家
        if self.training:
            values, indices = (origin_logits + self.bias.to(x.device)).topk(self.topk, dim=1, largest=True, sorted=True)
        else:
            values, indices = origin_logits.topk(self.topk, dim=1, largest=True, sorted=True)
        # values, indices = (origin_logits + self.bias.to(x.device)).topk(self.topk, dim=1, largest=True, sorted=True) 
        gating_logits = torch.zeros_like(origin_logits)

        # 使用不带偏置的门控分数聚合 
        values = torch.gather(origin_logits, dim=1, index=indices)
        values = torch.softmax(values, dim=1)
        gating_logits = torch.scatter(gating_logits, dim=1, index=indices, src=values)

        index = indices.reshape(-1)
        load = torch.zeros(self.num_experts).to(x.device)
        load = torch.index_add(load, dim=0, index=index, source=torch.ones_like(index, dtype=torch.float32).to(x.device))
        load_mu = torch.mean(load)
        sign = torch.sgn(load_mu - load)
        self.bias += sign * self.u
        return gating_logits, origin_logits, indices, load

class MoEFFNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts = 4):
        super(MoEFFNLayer, self).__init__()
        # self.GLUList = nn.ModuleList(
        #     [GLULayer(input_dim, 'SwishGLU'),
        #      GLULayer(input_dim, 'GEGLU'),
        #      GLULayer(input_dim, 'ReGLU')])
        self.FFNList = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_experts)
        ])
        # FFNGate
        self.FFNGate = nn.Linear(input_dim, num_experts)

    def reset_parameters(self):
        self.FFNGate.reset_parameters()
        for layer in self.FFNList:
            layer.reset_parameters()

    def forward(self, x):
        gate = F.softmax(self.FFNGate(x))
        x = torch.stack([layer(x) for layer in self.FFNList], dim=1)
        x = torch.sum(x * gate.unsqueeze(-1), dim=1)
        return x

class MoE(nn.Module): 
    def __init__(self, input_dim, output_dim, num_experts, topk, alpha = 0): 
        super(MoE, self).__init__() 
        # expert : (1, d)
        self.experts = nn.ModuleList(DotExpert(input_dim, output_dim) for i in range(num_experts))
        # self.experts = nn.ModuleList(LinearExpert(input_dim, output_dim) for i in range(num_experts))
        self.freeze_experts = 0
        self.freeze_experts_list = ['GCN', 'GAT', 'Average', 'Drop']
        for i in range(self.freeze_experts):
            expert_name = self.freeze_experts_list[i]
            if expert_name in ['GCN', 'GAT', 'Average']:
                nn.init.ones_(self.experts[i].W)
            elif expert_name in ['Drop']:
                nn.init.zeros_(self.experts[i].W)
            self.experts[i].W.requires_grad_ = False
        self.shared_expert = DotExpert(input_dim, output_dim)
        # nn.init.ones_(self.shared_expert.W)
        # self.shared_expert.W.requires_grad = False
        # self.shared_expert = LinearExpert(input_dim, output_dim)
        # self.experts = nn.ModuleList(LinearExpert(input_dim, output_dim) for i in range(num_experts)) 
        self.num_experts = num_experts
        self.topk = topk
        self.input_dim = input_dim
        # 用两个输入的特征进行路由
        self.gating = Gating(input_dim, num_experts, topk)
        self.alpha = alpha
        self.attention_start = nn.Linear(input_dim, 1)
        self.attention_end = nn.Linear(input_dim, 1)
        # self.gating_value = torch.empty(2*input_dim)

    def reset_parameters(self):
        for i in range(self.freeze_experts, len(self.experts)):
            self.experts[i].reset_parameters()
        self.shared_expert.reset_parameters()
        self.gating.reset_parameters()
        self.attention_end.reset_parameters()
        self.attention_start.reset_parameters()

    # edge_index: (2, e)
    # x: (N, d)
    def forward(self, x, edge_index): 
        edge_index, edge_weight = edge_index
        edge_start, edge_end = edge_index
        # gating_value = torch.cat((x[edge_start], x[edge_end]), dim=1)

        # Get the weights from the gating network
        # logits: (e, num_experts)
        gating_logits, origin_logits, indices, load = self.gating(x, edge_index)
        # gating_logits, origin_logits, indices, load = cp.checkpoint(self.gating, gating_value)

        edge_input = x[edge_start]
        # final_output = torch.zeros(edge_index.shape[1], x.shape[1]).to(x.device)
        final_output = self.shared_expert(edge_input, edge_weight.unsqueeze(1))

        atten_start = self.attention_start(x)
        atten_end = self.attention_end(x)
        atten = atten_start[edge_start] + atten_end[edge_end]
        atten = scatter_softmax(atten.squeeze(-1), edge_end)

        # nodes, count = torch.unique(edge_end, return_counts=True)
        # degree = torch.zeros(x.shape[0], dtype=torch.float32).to(x.device)
        # degree[nodes] = count.float()
        # non_zeros = torch.where(degree != 0)
        # degree[non_zeros] = 1/degree[non_zeros]
        # average_edge_input = degree[edge_end]

        for i, expert in enumerate(self.experts):
            input_mask = (indices == i).any(dim=-1)
            input_mask = input_mask.view(-1)
            if input_mask.any():
                expert_input = edge_input[input_mask]
                # if i == 1:
                #     edge_weight_input = atten[input_mask]
                # elif i == 2:
                #     edge_weight_input = average_edge_input[input_mask]
                # else:
                #     edge_weight_input = edge_weight[input_mask]
                edge_weight_input = atten[input_mask]
                # edge_weight_input = edge_weight[input_mask]
                expert_output = expert(expert_input, edge_weight_input.unsqueeze(1).expand_as(expert_input))

                gating_score = gating_logits[input_mask, i]
                expert_output = expert_output * gating_score.unsqueeze(1)
                if i < self.freeze_experts:
                    # 预设的消息传递方式不需要加上共享专家
                    final_output[input_mask] = expert_output.squeeze(1)
                else:
                    final_output[input_mask] += expert_output.squeeze(1)
        return final_output, gating_logits, load
    

# if __name__ == "__main__":
#     N = 10
#     E = 20
#     input_dim = 10
#     num_experts = 4
#     topk = 2
#     xixi = torch.rand(N, input_dim)
#     edge_index = torch.randint(10, (2, 20))
#     moe = MoE(input_dim, num_experts, topk)
#     haha = moe(xixi, edge_index)
#     print(haha.shape)
#     pass