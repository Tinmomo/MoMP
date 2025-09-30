import torch
import torch.nn.functional as F
from utils import get_dataset
import numpy as np


dataset='Facebook'
# data = get_dataset('../dataset/', dataset)
# print("node_num:", data.x.shape[0])
# print("edge_num:", data.edge_index.shape[1])
# print("features:", data.x.shape[1])
# print("Classes:", data.y.max()+1)
# print("density:", 100*data.edge_index.shape[1] /data.x.shape[0] ** 2)



gating_logits = torch.load('./gating_logits({}).tr'.format(dataset))
data = get_dataset('../dataset/', dataset)
edge_index = data.edge_index
edge_index[0] = data.y[edge_index[0]]
edge_index[1] = data.y[edge_index[1]]
# gating_logits = gating_logits[3][0].cpu()
# gating_experts = torch.argmax(gating_logits, dim=1)
# node, count = torch.unique(gating_experts, return_counts=True)
# label_experts = torch.zeros((data.y.max(), data.y.max(), 4))
# for label1, label2, expert in zip(edge_index[0]-1, edge_index[1]-1, gating_experts-1):  
#     label_experts[label1, label2, expert] += 1
# label_experts = label_experts.numpy()

# np.save('./label_expert({}).npy'.format(dataset), label_experts)

label_experts = np.load('./label_expert({}).npy'.format(dataset))
sum_label_experts = np.sum(label_experts, axis=2).reshape(label_experts.shape[0], label_experts.shape[1], 1)
sum_label_experts = np.repeat(sum_label_experts, 4, axis=2)
label_experts  = label_experts / sum_label_experts

# label_experts = np.argmax(label_experts, axis=2)

print(label_experts)
print(sum_label_experts)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 2. 绘制热力图
plt.figure(figsize=(8, 7)) # 设置图表大小，使之更清晰

# 使用 seaborn.heatmap 绘制热力图
# cmap 参数是关键：
# - 'Blues': 蓝色系，数值越大颜色越深
# - 'Greens': 绿色系，数值越大颜色越深
# - 'Reds': 红色系，数值越大颜色越深
# - 'viridis', 'plasma', 'magma', 'inferno': 都是从浅到深变化的颜色映射
# - 'YlGnBu' (Yellow-Green-Blue): 也是一个常用的从浅到深的渐变色
sns.heatmap(label_experts.max(axis=2),
            annot=True,      # 在热力图上显示数值
            fmt=".2f",         # 格式化注解的字符串，"d" 表示整数
            cmap="Blues",    # 选择颜色映射，例如'Blues', 'Greens', 'Reds', 'YlGnBu'
            linewidths=.5,   # 网格线宽度，增加区分度
            linecolor='black', # 网格线颜色
            vmin=0,
            vmax=1,
            cbar=True)       # 显示颜色条

# 设置标题
plt.title("CV of expert selections for different edge categories.", fontsize=16)
plt.xlabel("Source Node Label", fontsize=12)
plt.ylabel("Target Node Label", fontsize=12)

# 确保布局紧凑，防止标签重叠
plt.tight_layout()

# 显示图表
plt.savefig('./label_expert_stdheat({}).pdf'.format(dataset))