import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax


class DotExpert(nn.Module):
    def __init__(self, input_dim, output_dim, p=0.6):
        super(DotExpert, self).__init__()
        self.W = nn.Parameter(torch.empty((1, input_dim)))
        nn.init.uniform_(self.W)
        self.p = p

    def forward(self, x, edge_weight):
        return torch.mul(edge_weight, self.W * x + 1e-5)

    def reset_parameters(self):
        nn.init.uniform_(self.W)


class LinearExpert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearExpert, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x, edge_weight):
        return self.linear(x)

    def reset_parameters(self):
        self.linear.reset_parameters()


# Define the gating model
class Gating(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, topk=2, u=0.05):
        super(Gating, self).__init__()
        self.down_proj_start = nn.Linear(input_dim, input_dim)
        self.down_proj_end = nn.Linear(input_dim, input_dim)
        self.gating_linear = nn.Linear(input_dim, num_experts, bias=False)
        self.topk = topk
        self.num_experts = num_experts
        self.bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)
        self.u = u
        self.history_load = torch.zeros(num_experts)

    def reset_parameters(self):
        self.down_proj_end.reset_parameters()
        self.down_proj_start.reset_parameters()
        nn.init.xavier_normal_(self.gating_linear.weight)
        nn.init.zeros_(self.bias)

    def load_balance_update(self, load):
        load_mu = torch.mean(load)
        sign = torch.sgn(load_mu - load)
        self.bias = self.bias
        self.bias += sign * self.u

    def forward(self, x, edge_index):
        edge_start, edge_end = edge_index

        # 优化后的门控计算：分别对起点和终点做投影后再累加
        x = F.normalize(x, p=2, dim=1)
        gating_value_start = self.gating_linear(self.down_proj_start(x))
        gating_value_end = self.gating_linear(self.down_proj_end(x))
        origin_logits = gating_value_start[edge_start] + gating_value_end[edge_end]

        # 根据带偏置的门控分数选择 topk 个专家
        values, indices = (origin_logits + self.bias.to(x.device)).topk(self.topk, dim=1, largest=True, sorted=True)
        gating_logits = torch.zeros_like(origin_logits)

        # 使用不带偏置的门控分数聚合
        values = torch.gather(origin_logits, dim=1, index=indices)
        values = torch.softmax(values, dim=1)
        gating_logits = torch.scatter(gating_logits, dim=1, index=indices, src=values)

        index = indices.reshape(-1)
        load = torch.zeros(self.num_experts).to(x.device)
        load = torch.index_add(load, dim=0, index=index, source=torch.ones_like(index, dtype=torch.float32).to(x.device))
        if self.training:
            self.load_balance_update(load)
        return gating_logits, indices, load


class MoEFFNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4):
        super(MoEFFNLayer, self).__init__()
        self.FFNList = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_experts)
        ])
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
    def __init__(self, input_dim, output_dim, num_experts, topk, alpha=0):
        super(MoE, self).__init__()
        # 每个 expert 形状: (1, d)
        self.experts = nn.ModuleList(DotExpert(input_dim, output_dim) for i in range(num_experts))
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
        self.num_experts = num_experts
        self.topk = topk
        self.input_dim = input_dim
        # 用两个端点的特征进行路由
        self.gating = Gating(input_dim, num_experts, topk)
        self.alpha = alpha
        self.attention_start = nn.Linear(input_dim, 1)
        self.attention_end = nn.Linear(input_dim, 1)

    def reset_parameters(self):
        for i in range(self.freeze_experts, len(self.experts)):
            self.experts[i].reset_parameters()
        self.shared_expert.reset_parameters()
        self.gating.reset_parameters()
        self.attention_end.reset_parameters()
        self.attention_start.reset_parameters()

    # edge_index: (2, e)
    # x: (N, d)
    def forward(self, x, edge_index, return_logits=False):
        edge_index, edge_weight = edge_index
        edge_start, edge_end = edge_index  # [E], [E]

        gating_logits, indices, load = self.gating(x, edge_index)  # [E,M], [E,k], [M]
        # 假设 topk=1
        top_expert = indices[:, 0]  # [E]
        top_gate = gating_logits[torch.arange(gating_logits.size(0), device=x.device), top_expert]  # [E]

        edge_input = x[edge_start]  # [E,D]

        # shared expert
        if edge_weight.dim() == 1:
            ew = edge_weight.unsqueeze(1)
        else:
            ew = edge_weight
        final_output = self.shared_expert(edge_input, ew)  # [E,D]

        # edge attention
        atten_start = self.attention_start(x)  # [N,1]
        atten_end = self.attention_end(x)      # [N,1]
        atten = atten_start[edge_start] + atten_end[edge_end]  # [E,1]
        atten = scatter_softmax(atten.squeeze(-1), edge_end)   # [E]

        # 堆叠 expert 参数: [M,D]
        W_all = torch.cat([expert.W for expert in self.experts], dim=0)

        # node-level 预计算: [N,M,D]
        node_expert_feat = x.unsqueeze(1) * W_all.unsqueeze(0) + 1e-5

        # 按边稀疏挑选 (只取被选中的那个 expert): [E,D]
        selected_feat = node_expert_feat[edge_start, top_expert]

        # 应用 attention + topk gate
        selected_feat = selected_feat * atten.unsqueeze(1) * top_gate.unsqueeze(1)

        final_output = final_output + selected_feat

        if return_logits:
            return final_output, gating_logits, load
        else:
            return final_output
