import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit, Actor, DBLP, IMDB, facebook
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
import pandas as pd


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    Args:
        alpha (float or Tensor): Weighting factor for each class. Can be a single float
                                 (applied to all classes) or a Tensor of size (num_classes,).
                                 Default: 0.25
        gamma (float): Focusing parameter. Default: 2.0
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                         'mean': the sum of the output will be divided by the number of
                                 elements in the output, 'sum': the output will be summed.
                                 Default: 'mean'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Model's raw output (logits) of shape (N, C)
            targets (Tensor): Ground truth labels of shape (N,)
        """
        # 1. 计算 softmax 和 log_softmax
        # 使用 log_softmax 在数值上比 log(softmax(x)) 更稳定
        log_p = F.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)

        # 2. 获取每个样本的标准交叉熵损失 (NLL Loss)
        # 这里我们使用 nll_loss，它需要 log-probabilities 作为输入
        # 我们设置 reduction='none' 来获取每个样本的损失
        ce_loss = F.nll_loss(log_p, targets, reduction='none')

        # 3. 提取每个样本真实类别的预测概率 p_t
        # p.gather(1, targets.view(-1, 1)) 是一个高效的实现方式
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 4. 计算 Focal Loss 的调制因子
        modulating_factor = (1 - p_t)**self.gamma

        # 5. 处理 alpha 权重
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_factor = torch.full_like(targets, self.alpha, dtype=torch.float32)
            elif isinstance(self.alpha, torch.Tensor):
                # 根据 targets 的值来选取对应的 alpha
                alpha_factor = self.alpha.gather(0, targets)
            else:
                 raise TypeError("alpha must be a float, an int, or a torch.Tensor")
        else:
            # 如果不提供 alpha，则权重为 1
            alpha_factor = torch.ones_like(targets, dtype=torch.float32)

        # 6. 计算最终的 Focal Loss
        # loss = alpha * (1-p_t)^gamma * CE
        focal_loss = alpha_factor * modulating_factor * ce_loss

        # 7. 应用 reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.1, label_weight = None):
        super().__init__()
        self.smoothing = smoothing
        self.criterion = nn.CrossEntropyLoss(label_weight)
    
    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        # 创建平滑标签
        smoothed_targets = torch.full_like(logits, self.smoothing/(num_classes-1))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1-self.smoothing)
        
        # 计算交叉熵
        # log_probs = F.log_softmax(logits, dim=-1)
        
        # loss = -torch.sum(log_probs * smoothed_targets, dim=-1)
        loss = self.criterion(logits, smoothed_targets)
        return loss.mean()


transform = T.Compose([
    T.ToUndirected(),
])

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def get_dataset(root: str, name: str, transform=transform) -> Data:
    if name in {'arxiv', 'products', 'mag', 'proteins'}:
        from ogb.nodeproppred import PygNodePropPredDataset
        print('loading ogb dataset...')
        dataset = PygNodePropPredDataset(root=root, name=f'ogbn-{name}')
        if name in ['mag']:
            rel_data = dataset[0]
            # We are only interested in paper <-> paper relations.
            data = Data(
                    x=rel_data.x_dict['paper'],
                    edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
                    y=rel_data.y_dict['paper'])
            data = transform(data)
            split_idx = dataset.get_idx_split()
            data.train_mask = index_to_mask(split_idx['train']['paper'], data.num_nodes)
            data.val_mask = index_to_mask(split_idx['valid']['paper'], data.num_nodes)
            data.test_mask = index_to_mask(split_idx['test']['paper'], data.num_nodes)
        else:
            data = transform(dataset[0])
            split_idx = dataset.get_idx_split()
            data.train_mask = index_to_mask(split_idx['train'], data.num_nodes)
            data.val_mask = index_to_mask(split_idx['valid'], data.num_nodes)
            data.test_mask = index_to_mask(split_idx['test'], data.num_nodes)
            # data = T.RandomNodeSplit(num_val=0.32, num_test=0.2)(data)

    elif name in {'Cora', 'Citeseer', 'Pubmed'}:
        dataset = Planetoid(root, name)
        xixi = dataset[0]
        data = transform(dataset[0])

    elif name == 'Reddit':
        dataset = Reddit(os.join(root, name))
        data = transform(dataset[0])
    elif name in {'Photo', 'Computers'}:
        dataset = Amazon(root, name)
        data = transform(dataset[0])
        # data = T.RandomNodeSplit(num_val=0.32, num_test=0.2)(data)
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    elif name in {'CS', 'Physics'}:
        dataset = Coauthor(root, name)
        data = transform(dataset[0])
        # data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
        data = T.RandomNodeSplit(num_val=0.32, num_test=0.2)(data)
    elif name in {'Actor'}:
        dataset = Actor(root+'film')
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.32, num_test=0.2)(data)
    elif name in {'DBLP'}:
        dataset = DBLP(root+name)
        data = transform(dataset[0])
    elif name in {'IMDB'}:
        dataset = IMDB(root+name)
        data = transform(dataset[0])
    elif name in {'Chameleon'}:
        dataset = np.load(os.path.join(root+name,'chameleon_filtered.npz') )
        node_features = torch.tensor(dataset['node_features'])
        labels = torch.tensor(dataset['node_labels'])
        edges = torch.tensor(dataset['edges'])
        data = Data(x = node_features, edge_index=edges.T, y=labels)
        data = transform(data)
        data = T.RandomNodeSplit(num_val=0.32, num_test=0.2)(data)
    elif name in {'Facebook'}:
        node_features = pd.read_csv(os.path.join(root+name, 'features.csv')).to_numpy(dtype='int')
        edges = pd.read_csv(os.path.join(root+name, 'edges.csv')).to_numpy(dtype='int')
        labels = pd.read_csv(os.path.join(root+name, 'target.csv')).to_numpy(dtype='int')
        edges = torch.tensor(edges)
        labels = torch.tensor(labels[:,1])
        node_features = torch.tensor(node_features)
        features = torch.zeros(labels.shape[0], node_features[:,1].max()+1)
        features[node_features[:, 0], node_features[:, 1]] = torch.tensor(node_features[:, 2], dtype=torch.float32)

        data = Data(x = features, edge_index=edges.T, y=labels)
        data = transform(data)
        data = T.RandomNodeSplit(num_val=0.32, num_test=0.2)(data)
    else:
        raise ValueError(name)
    return data

def HeteroDataProcess(data):
    x = None
    for type in data.node_types:
        if x is None:
            x = data[type].x
        else:
            x = torch.cat([x, data[type].x])
    edge_indices = data.edge_stores
    edges = None
    for edge_index, edge_type in zip(edge_indices, data.edge_types):
        start_offset = data.node_offsets[edge_type[0]]
        end_offset  = data.node_offsets[edge_type[2]]
        edge_index['edge_index'][0] += start_offset
        edge_index['edge_index'][1] += end_offset
        if edges is None:
            edges = edge_index['edge_index']
        else:
            edges = torch.cat([edges, edge_index['edge_index']], dim=1)
        pass
    labels = data[data.node_types[0]].y
    return x, labels, edges.T

def load_data(path='../dataset/', dataset='Cora', two_hop_sample_number = 5):
    data = get_dataset(path, dataset)
    if dataset in {'DBLP', 'IMDB'}:
        features, labels, edges = HeteroDataProcess(data)
        access_data = data[data.node_types[0]]
        idx_train = torch.where(access_data.train_mask)[0]
        idx_val = torch.where(access_data.val_mask)[0]
        idx_test = torch.where(access_data.test_mask)[0]
    else:
        features = data.x
        labels = data.y
        edges = (data.edge_index).T
        idx_train = torch.where(data.train_mask)[0]
        idx_val = torch.where(data.val_mask)[0]
        idx_test = torch.where(data.test_mask)[0]
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(data.num_nodes, data.num_nodes), dtype=np.float32)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # if dataset not in ['arxiv']:
    #     two_hop_adj = adj.dot(adj)
    #     two_hop_adj = two_hop_adj.multiply(adj == 0)
    #     two_hop_adj = normalize_adj(two_hop_adj).tocoo()
    #     two_hop_edge_index = torch.tensor(np.vstack((two_hop_adj.row, two_hop_adj.col)).T, dtype=torch.long)
    #     two_hop_edge_weight = torch.tensor(two_hop_adj.data, dtype=torch.float32)
    # else:
    two_hop_edge_index = None
    two_hop_edge_weight = None

    # if labels.shape[1] == 1:
    #     class_num = labels.max()+1
    #     new_labels = torch.zeros(labels.shape[0], class_num, dtype=torch.int32)
    #     new_labels[labels] = 1
    #     labels = new_labels
    # adj = torch.FloatTensor(np.array(adj.todense()))
    
    if features is None:
        features = torch.load(path+f'ogbn_{dataset}/embedding.pt')
    features = normalize_features(features)
    features = torch.FloatTensor(features)
      
    return adj, edges, two_hop_edge_index, two_hop_edge_weight, features, labels, idx_train, idx_val, idx_test, data

def sample_2hop_neighbor(two_hop_edge_index, two_hop_edge_weight, two_hop_neighbor_count, sample_number):
    rand_index = torch.cat( [torch.randint(pos.item(), (sample_number,)) for pos in two_hop_neighbor_count] )
    base_index = torch.cat([torch.tensor([0]), two_hop_neighbor_count[: len(two_hop_neighbor_count)-1]])
    base_index = torch.cumsum(base_index, dim=0)
    base_index = torch.repeat_interleave(base_index, sample_number)
    index = base_index + rand_index
    two_hop_edge_index = two_hop_edge_index[index]
    two_hop_edge_weight = two_hop_edge_weight[index]
    softmaxed = nn.functional.softmax(two_hop_edge_weight.view(-1,sample_number), dim=1)
    two_hop_edge_weight = softmaxed.view(-1, )
    return two_hop_edge_index, two_hop_edge_weight

def visualize(x):
    pca = PCA(n_components=2)
    embedded_data = pca.fit_transform(x)
    plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
    plt.title('t-SNE Visualization')
    plt.savefig("./expert visualize.png")
    plt.show()

def decomposition(x, feature_size):
    svd = TruncatedSVD(n_components=feature_size)
    x = svd.fit_transform(x)
    return x

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

