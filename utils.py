import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from torch_geometric.datasets import (
    Amazon, Coauthor, Planetoid, Reddit, Actor, DBLP, IMDB,
    HeterophilousGraphDataset,
)
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask
import torch_geometric.transforms as T
import pandas as pd


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Args:
        alpha (float or Tensor): Weighting factor for each class. Can be a single float
                                 (applied to all classes) or a Tensor of size (num_classes,).
                                 Default: 0.25
        gamma (float): Focusing parameter. Default: 2.0
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. log_softmax 数值上更稳定
        log_p = F.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)

        # 2. 每个样本的 NLL Loss(reduction='none' 取 per-sample)
        ce_loss = F.nll_loss(log_p, targets, reduction='none')

        # 3. 真实类别的预测概率 p_t
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 4. focal 调制因子
        modulating_factor = (1 - p_t) ** self.gamma

        # 5. alpha 权重
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_factor = torch.full_like(targets, self.alpha, dtype=torch.float32)
            elif isinstance(self.alpha, torch.Tensor):
                alpha_factor = self.alpha.gather(0, targets)
            else:
                raise TypeError("alpha must be a float, an int, or a torch.Tensor")
        else:
            alpha_factor = torch.ones_like(targets, dtype=torch.float32)

        # 6. focal_loss = alpha * (1-p_t)^gamma * CE
        focal_loss = alpha_factor * modulating_factor * ce_loss

        # 7. reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.1, label_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.criterion = nn.CrossEntropyLoss(label_weight)

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        # 创建平滑标签
        smoothed_targets = torch.full_like(logits, self.smoothing / (num_classes - 1))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        loss = self.criterion(logits, smoothed_targets)
        return loss.mean()


transform = T.Compose([
    T.ToUndirected(),
])


def encode_onehot(labels):
    # 编码前先排序，确保第一个类别永远映射到 index 0
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
            # 只保留 paper <-> paper 关系
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

    elif name in {'Cora', 'Citeseer', 'Pubmed'}:
        dataset = Planetoid(root, name)
        data = transform(dataset[0])

    elif name == 'Reddit':
        dataset = Reddit(os.join(root, name))
        data = transform(dataset[0])
    elif name in {'Photo', 'Computers'}:
        dataset = Amazon(root, name)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    elif name in {'CS', 'Physics'}:
        dataset = Coauthor(root, name)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.32, num_test=0.2)(data)
    elif name in {'Actor'}:
        dataset = Actor(root + 'film')
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.32, num_test=0.2)(data)
    elif name in {'DBLP'}:
        dataset = DBLP(root + name)
        data = transform(dataset[0])
    elif name in {'IMDB'}:
        dataset = IMDB(root + name)
        data = transform(dataset[0])
    elif name in {'Chameleon', "Squirrel"}:
        dataset = np.load(os.path.join(root + name, f'{name}_filtered.npz'))
        node_features = torch.tensor(dataset['node_features'])
        labels = torch.tensor(dataset['node_labels'])
        edges = torch.tensor(dataset['edges'])
        data = Data(x=node_features, edge_index=edges.T, y=labels)
        data = transform(data)
        data = T.RandomNodeSplit(num_val=0.32, num_test=0.2)(data)
    elif name in {'Roman-empire'}:
        dataset = HeterophilousGraphDataset(root=root, name=name)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.32, num_test=0.2)(data)
    elif name in {'Facebook'}:
        node_features = pd.read_csv(os.path.join(root + name, 'features.csv')).to_numpy(dtype='int')
        edges = pd.read_csv(os.path.join(root + name, 'edges.csv')).to_numpy(dtype='int')
        labels = pd.read_csv(os.path.join(root + name, 'target.csv')).to_numpy(dtype='int')
        edges = torch.tensor(edges)
        labels = torch.tensor(labels[:, 1])
        node_features = torch.tensor(node_features)
        features = torch.zeros(labels.shape[0], node_features[:, 1].max() + 1)
        features[node_features[:, 0], node_features[:, 1]] = torch.tensor(node_features[:, 2], dtype=torch.float32)

        data = Data(x=features, edge_index=edges.T, y=labels)
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
        end_offset = data.node_offsets[edge_type[2]]
        edge_index['edge_index'][0] += start_offset
        edge_index['edge_index'][1] += end_offset
        if edges is None:
            edges = edge_index['edge_index']
        else:
            edges = torch.cat([edges, edge_index['edge_index']], dim=1)
    labels = data[data.node_types[0]].y
    return x, labels, edges.T


def load_data(path='../dataset/', dataset='Cora'):
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

    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(data.num_nodes, data.num_nodes), dtype=np.float32,
    )
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    if features is None:
        features = torch.load(path + f'ogbn_{dataset}/embedding.pt')
    features = normalize_features(features)
    features = torch.FloatTensor(features)

    return adj, edges, features, labels, idx_train, idx_val, idx_test, data


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
