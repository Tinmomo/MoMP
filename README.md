# MoMP

Official PyTorch implementation of **MoMP: Mixture of Message Passing for Graph Neural Networks**.

MoMP replaces the scalar aggregation coefficients of classic message passing with **learnable vector-valued aggregations**, and combines several specialized experts (learnable / shared / sparsely-routed) through a gating network. This lifts the expressive ceiling of message passing while keeping the per-layer cost comparable to a standard GNN layer.

---



## 🛠️ Requirements

### Hardware

| Item          | Specification                          |
| ------------- | -------------------------------------- |
| OS            | Linux (Ubuntu 20.04 / CentOS 7 tested) |
| GPU           | NVIDIA GeForce RTX 4090 (24 GB) × 4    |
| NVIDIA Driver | 550.163.01                             |
| CUDA Runtime  | 12.4                                   |
| RAM           | ≥ 32 GB                                |

### Software

Core dependencies (tested versions):

- Python 3.11.5
- PyTorch **2.4.0** (built with CUDA 12.4)
- PyTorch Geometric **2.7.0**
- torch-scatter **2.1.2** (`+pt24cu124` build)
- OGB **1.3.6**

Full dependency list: see [`requirements.txt`](./requirements.txt).

### Installation

```bash
# 1. Create conda environment
conda create -n momp python=3.11 -y
conda activate momp

# 2. Install PyTorch 2.4.0 with CUDA 12.4
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 3. Install PyG and torch-scatter
#    torch-scatter wheel must match torch + cuda version
pip install torch_geometric==2.7.0
pip install torch_scatter==2.1.2 \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# 4. Install other dependencies
pip install -r requirements.txt
```

> **Note**: `torch-scatter` is compiled against a specific `torch + CUDA` combination.
> If you use a different PyTorch / CUDA version, please find the matching wheel
> at https://data.pyg.org/whl/.

---



## 📁 Project Structure

```
MoMP/
├── train.py        # Entry point: training & evaluation loop
├── models.py       # GraphMoE backbone
├── layers.py       # MoE-based message passing layer
├── moe.py          # MoE router, gating, experts
├── utils.py        # Data loading, normalization, losses
├── requirements.txt
└── README.md
```



## 🚀Quick Start

Datasets supported by `utils.load_data` are downloaded automatically into
`../dataset/` on first use (override with the `path=` argument inside
`load_data`).

```bash
# Train on Photo with the default arguments in train.py
python train.py --dataset Photo
```

Override any hyperparameter from the command line, for example:

```bash
python train.py --dataset CS \
                --lr 1e-3 --weight_decay 1e-3 \
                --hidden 512 --dropout 0.3 \
                --num_experts 8 --num_layers 2
```

The most relevant CLI flags (see `train.py` for the full list):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset` | `Facebook` | Dataset name (see table below) |
| `--lr` | `5e-3` | Adam learning rate |
| `--weight_decay` | `5e-4` | L2 weight decay |
| `--hidden` | `128` | Hidden dim of MoMP layers |
| `--dropout` | `0.9` | Dropout rate |
| `--num_experts` | `4` | Number of experts per MoMP layer |
| `--topk` | `1` | Active experts per token (sparse routing) |
| `--num_layers` | `4` | Number of stacked MoMP layers |
| `--epochs` / `--patience` | `5000` / `200` | Training budget and early-stop patience |
| `--seed` | `42` | Random seed |



## Reproducing the Paper

The settings below are the configurations used to produce the numbers reported
in the paper. All omitted flags keep the defaults shown in the *Quick Start*
table (in particular `seed=42`, `epochs=5000`, `patience=200`, `topk=1`).

### 

### Ready-to-run commands

```bash
python train.py --dataset Photo \
                --lr 1e-3 --weight_decay 1e-3 \
                --hidden 128 --dropout 0.4 \
                --num_experts 4 --num_layers 3 \
                --use_classifier True --bn LayerNorm --label_smoothing 0.1

python train.py --dataset Computers \
                --lr 1e-3 --weight_decay 5e-4 \
                --hidden 128 --dropout 0.5 \
                --num_experts 4 --num_layers 3 \
                --alpha_lb 0.001 \
                --use_classifier True --bn LayerNorm --label_smoothing 0.1

python train.py --dataset Physics \
                --lr 5e-4 --weight_decay 5e-4 \
                --hidden 512 --dropout 0.3 \
                --num_experts 8 --num_layers 2 \
                --use_classifier True --bn LayerNorm --label_smoothing 0.1

python train.py --dataset CS \
                --lr 1e-3 --weight_decay 1e-3 \
                --hidden 512 --dropout 0.3 \
                --num_experts 8 --num_layers 2 \
                --use_classifier True --bn LayerNorm --label_smoothing 0.1

python train.py --dataset Facebook \
                --lr 5e-3 --weight_decay 1e-3 \
                --hidden 256 --dropout 0.5 \
                --num_experts 4 --num_layers 4 \
                --use_classifier True --bn LayerNorm --label_smoothing 0

python train.py --dataset arxiv \
                --lr 1e-3 --weight_decay 0 \
                --hidden 128 --dropout 0.3 \
                --num_experts 1 --num_layers 3 \
                --alpha_lb 0.001 \
                --use_classifier True --bn LayerNorm --label_smoothing 0

```



## 📜Citation

If you find MoMP useful in your research, please cite:

```bibtex
@inproceedings{momp2026,
  title     = {MoMP: Mixture of Message Passing for Graph Neural Networks},
  author    = { Zhaojun Luo and
                Jintang Li and
                Yuchang Zhu and
                Yun Fu and
                Liang Chen and
                Zibin Zheng},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining (KDD)},
  year      = {2026}
}
