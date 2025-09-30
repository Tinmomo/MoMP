from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchmetrics
from ogb.nodeproppred import Evaluator
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from utils import load_data, accuracy, visualize, LabelSmoothLoss, sample_2hop_neighbor, FocalLoss
from models import GraphMoE

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.', required=False)
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.', required=False)
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.', required=False)
parser.add_argument('--seed', type=int, default=42, help='Random seed.', required=False)
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.', required=False)
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.', required=False)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).', required=False)
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.', required=False)
parser.add_argument('--dropout', type=float, default=0.9, help='Dropout rate (1 - keep probability).', required=False)
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.', required=False)
parser.add_argument('--patience', type=int, default=200, help='Patience', required=False)
parser.add_argument('--num_experts', type=int, default=4, help='Number of experts', required=False)
parser.add_argument('--topk', type=int, default=1, help='Number of activated experts', required=False)
parser.add_argument('--alpha_lb', type=float, default=0, help='Alpha for load balancing', required=False)
parser.add_argument('--num_layers', type=int, default=4, help='Number of MOE layers', required=False)
parser.add_argument('--batch', type=int, default=1, help='Number of batch to compute', required=False)
parser.add_argument('--similarity_loss', type=bool, default=False, help='Enable Similarity loss', required=False)
parser.add_argument('--two_hop_sample_number', type=int, default=0, help='The number of two hop neighbor', required=False)
parser.add_argument('--batch_size', type=int, default=4096, help='Nodes batch size', required=False)
parser.add_argument('--dataset', type=str, default='Facebook', help='dataset', required=False)

args = parser.parse_args(args=[])
args.cuda = not args.no_cuda and torch.cuda.is_available()

def seed_everything(seed):
    # seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

seed_everything(args.seed)
# Load data
# adj, edge_index, features, labels, idx_train, idx_val, idx_test = load_data(dataset='arxiv')
# adj, edge_index, features, labels, idx_train, idx_val, idx_test = load_data(dataset='Photo')
# adj, edge_index, features, labels, idx_train, idx_val, idx_test = load_data(dataset='Citeseer')
# adj, edge_index, features, labels, idx_train, idx_val, idx_test = load_data(dataset='Pubmed')
adj, edge_index, two_hop_edge_index, two_hop_edge_weight, features, labels, idx_train, idx_val, idx_test, data = load_data(dataset=args.dataset)

# Model and optimizer

model = GraphMoE(nfeat=features.shape[1], 
            nhid=args.hidden, 
            nclass=int(labels.max()) + 1, 
            dropout=args.dropout, 
            alpha=args.alpha,
            num_experts= args.num_experts,
            num_layers= args.num_layers,
            topk= args.topk,
            batch = args.batch
            )

optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

_, label_count = torch.unique(labels, return_counts = True)
label_weight = torch.tensor([(len(labels) / count) for count in label_count])
label_weight = torch.clamp(label_weight, max=10.0)
# label_weight = F.softmax(label_weight, dim=0)



# model.load_state_dict(torch.load("./2996.pkl"))
device_num = 1
if args.cuda:
    model = model.cuda(device_num)
    features = features.cuda(device_num)
    labels = labels.squeeze().cuda(device_num)
    idx_train = idx_train.cuda(device_num)
    idx_val = idx_val.cuda(device_num)
    idx_test = idx_test.cuda(device_num)
    label_weight = label_weight.cuda(device_num)
    edge = edge_index.T.cpu().numpy()
    edge_weight = adj[edge[0], edge[1]]
    edge_index = edge_index.T.cuda(device_num)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32).squeeze(0).cuda(device_num)
    edge_index = [edge_index, edge_weight]
    
    if args.two_hop_sample_number != 0:
        nodes, two_hop_neighbor_count = torch.unique(two_hop_edge_index[:,0], return_counts=True)




# features, adj, labels = Variable(features), Variable(adj), Variable(labels)
# del adj



features, labels = Variable(features),  Variable(labels)
# criterion = nn.CrossEntropyLoss()
criterion = LabelSmoothLoss(smoothing=0.1, label_weight=None)
# criterion = FocalLoss(alpha=label_weight, gamma=2)
# criterion = F.nll_loss()
if args.dataset == "proteins":
    evaluator = Evaluator(name='ogbn-proteins')
    criterion = evaluator.eval

def calLoadBalancingLoss(logits, indices, alpha):
    loss = torch.zeros(1).cuda()
    for (logit, load) in zip(logits, indices):
        logit = logit[0]
        load = load[0]
        edge_num = logit.shape[0]
        experts_num = logit.shape[1]

        # print(load)
        load = torch.divide(load, edge_num)

        logit = torch.sum(logit, dim=0)
        logit = torch.divide(logit, edge_num)
        loss += torch.sum(logit * load)
        # loss += torch.std(load)
    return (alpha * experts_num * loss)

def calLoadBalancingLossCV(logits, indices, alpha):
    loss = torch.zeros(1).cuda()
    for (logit, index) in zip(logits, indices):
        edge_num = logit.shape[0]
        experts_num = logit.shape[1]

        index = index.reshape(-1)
        load = torch.zeros(model.num_experts, dtype=torch.long).cuda()
        load = torch.index_add(load, dim=0, index=index, source=torch.ones_like(index).cuda())
        print(load)
        load = torch.divide(load, edge_num)

        logit = torch.sum(logit, dim=0)
        sigma, mu = torch.std_mean(logit)
        loss += (sigma / mu**2)
    return (alpha * loss)

def calSimilarityLoss(layer_similarity, temperature=0.1):
    return temperature * torch.logsumexp(layer_similarity / temperature, dim=1).mean()

def train_step(epoch):
    alpha = 0.1
    t = time.time()
    model.train()
    optimizer.zero_grad()
    """
    origin train framework
    """
    if args.two_hop_sample_number == 0:
        two_hop_edge_index_step = None
    else:
        two_hop_edge_index_step, two_hop_edge_weight_step = sample_2hop_neighbor(
            two_hop_edge_index, two_hop_edge_weight, two_hop_neighbor_count, args.two_hop_sample_number)
        two_hop_edge_index_step = [two_hop_edge_index_step.T.cuda(device_num), two_hop_edge_weight_step.cuda(device_num)]
    output, logits, load = model(features, edge_index, two_hop_edge_index_step)
    # print(load)
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    loss_train = criterion(output[idx_train], labels[idx_train])
    
    if args.similarity_loss:
        experts_similarity = model.get_experts_similarity()
        for layer_similarity in experts_similarity:
            loss_train += alpha * calSimilarityLoss(layer_similarity - torch.eye(layer_similarity.shape[0]).to(loss_train.device) )
            # loss_train += alpha * torch.max(layer_similarity - torch.eye(layer_similarity.shape[0]).to(loss_train.device) )
            pass
    # 负载均衡损失
    loss_lb = calLoadBalancingLoss(logits, load, args.alpha_lb)
    loss_train = torch.add(loss_train, loss_lb)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()



    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        if args.two_hop_sample_number == 0:
            two_hop_edge_index_step = None
        else:
            two_hop_edge_index_step, two_hop_edge_weight_step = sample_2hop_neighbor(
                two_hop_edge_index, two_hop_edge_weight, two_hop_neighbor_count, args.two_hop_sample_number)
            two_hop_edge_index_step = [two_hop_edge_index_step.T.cuda(device_num), two_hop_edge_weight_step.cuda(device_num)]
        output, _, _= model(features, edge_index, two_hop_edge_index_step)

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    loss_val = criterion(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    # micro_f1_score = torchmetrics.F1Score(task="multiclass", num_classes=int(labels.max()) + 1, average='micro').to(output.device)
    # macro_f1_score = torchmetrics.F1Score(task="multiclass", num_classes=int(labels.max()) + 1, average='macro').to(output.device)
    # micro_f1_train = micro_f1_score(output[idx_train], labels[idx_train])
    # macro_f1_train = macro_f1_score(output[idx_train], labels[idx_train])
    # micro_f1_val = micro_f1_score(output[idx_val], labels[idx_val])
    # macro_f1_val = macro_f1_score(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
        #   'micro_f1_train: {:.4f}'.format(micro_f1_train),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
        #   'micro_f1_val:{:.4f}'.format(micro_f1_val),
            'loss_test: {:.4f}'.format(loss_test.data.item()),
          'acc_test: {:.4f}'.format(acc_test.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    # print(np.sum(in))
    return loss_val.data.item()


def compute_test(download=False):
    model.eval()
    if args.two_hop_sample_number == 0:
        two_hop_edge_index_step = None
    else:
        two_hop_edge_index_step, two_hop_edge_weight_step = sample_2hop_neighbor(
            two_hop_edge_index, two_hop_edge_weight, two_hop_neighbor_count, args.two_hop_sample_number)
        two_hop_edge_index_step = [two_hop_edge_index_step.T.cuda(device_num), two_hop_edge_weight_step.cuda(device_num)]
    output, gating_logits, _ = model(features, edge_index, two_hop_edge_index_step)
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    if download:
        torch.save(gating_logits, './gating_logits({}).tr'.format(args.dataset))
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    micro_f1_score = torchmetrics.F1Score(task="multiclass", num_classes=int(labels.max()) + 1, average='micro').to(output.device)
    macro_f1_score = torchmetrics.F1Score(task="multiclass", num_classes=int(labels.max()) + 1, average='macro').to(output.device)
    micro_f1 = micro_f1_score(output[idx_test], labels[idx_test])
    macro_f1 = macro_f1_score(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()),
          "micro_f1= {:.4f}".format(micro_f1),
        "macro_f1= {:.4f}".format(macro_f1) )
    return acc_test.data.item()

def train_model():
    # Train model
    t_total = time.time()
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        loss_values = train_step(epoch)

        
        if loss_values < best:
            best = loss_values
            best_epoch = epoch
            bad_counter = 0
            torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    # best_epoch = 589
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
    model.print_similarity()
    # model.print_mix_hop_rate()
    

    # Testing
    acc_test = compute_test()
    xixi = model.moe_layers[0].moe.gating.gating_linear.weight.detach().cpu().numpy()
    # visualize(xixi)
    return acc_test

def mask_experts_test(model):
    model.eval()
    mask_threshold = [i * 0.1 for i in range(0, 10)]
    acc_test_list = []
    for threshold in mask_threshold:
        model.mask_experts(threshold)
        model.print_experts()
        acc_test = compute_test(True)
        acc_test_list.append(acc_test)
    plt.title(args.dataset)
    plt.plot(mask_threshold, acc_test_list, marker='s')
    plt.xlabel('mask_threshold')
    plt.ylabel('test set accuracy')
    plt.grid(visible=True)
    plt.savefig('./threshold({}).png'.format(args.dataset))
    print(acc_test_list)
    
    
import torch_geometric.transforms as T

def random_split(data, dataset):
    if dataset in ['Photo', 'Computers']:
        data = T.RandomNodeSplit(num_val=0.10, num_test=0.8)(data)
    elif dataset in ['arxiv']:
        pass
    else:
        data = T.RandomNodeSplit(num_val=0.32, num_test=0.2)(data)
    return data

if __name__ == "__main__":
    test_acc_list = []
    runs = 1
    for i in range(runs):
        
        # data = random_split(data, args.dataset)
        # idx_train = torch.where(data.train_mask)[0].cuda(device_num)
        # idx_val = torch.where(data.val_mask)[0].cuda(device_num)
        # idx_test = torch.where(data.test_mask)[0].cuda(device_num)
        model.reset_parameters()
        acc_test = train_model()
        # model.print_experts()
        mask_experts_test(model)
        test_acc_list.append(acc_test)
    
    print(f"Average test accuracy of {runs} runs: {np.mean(test_acc_list):.2%} ± {np.std(test_acc_list):.2%}")
    print("seed:", args.seed)
    