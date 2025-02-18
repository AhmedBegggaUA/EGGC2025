import torch
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from models.gcn_mincut import GCNMincut
from models.gcn_diffpool import GCNDiffPool
from models.gcn_dmon import GCNDMoN
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix
import os
import warnings
warnings.filterwarnings("ignore")
from umap import UMAP

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--renyi', type=float, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--architecture', type=str, default='512')
    return parser.parse_args()

def get_embeddings(model, data, device):
    model.eval()
    with torch.no_grad():
        adj_tensor = torch.sparse_coo_tensor(
            data.edge_index, 
            torch.ones(data.edge_index.size(1)),
            (data.num_nodes, data.num_nodes)
        ).to(device)
        _,embeddings = model((data.x.to(device), adj_tensor))
    return embeddings.cpu().numpy()

def plot_embeddings(embeddings, labels, title, save_path):
    plt.figure(figsize=(8, 8))
    
    plt.scatter(embeddings[:, 0], embeddings[:, 1], 
               c=labels, cmap='inferno',
               alpha=0.75, s=60)
    
    plt.xticks([])
    plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=700, bbox_inches='tight')
    plt.close()

def plot_single_method(data, base_model, renyi_model, method, renyi_value, device, save_path_base):
    # Base model
    base_embeddings = get_embeddings(base_model, data, device)
    umap = UMAP(
        n_components=2,
        n_neighbors=10,  # Reducido de 15 a 10
        min_dist=0.8,    # Aumentado de 0.5 a 0.8
        spread=2.0,      # Añadido spread para mayor separación
        random_state=115  # Para consistencia
    )
    base_embeddings = umap.fit_transform(base_embeddings)
    plot_embeddings(
        base_embeddings, 
        data.y.numpy(), 
        f"{method} - Base Model", 
        f"{save_path_base}_base.pdf"
    )
    
    # Renyi model
    renyi_embeddings = get_embeddings(renyi_model, data, device)
    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.5)
    renyi_embeddings = umap.fit_transform(renyi_embeddings)
    plot_embeddings(
        renyi_embeddings, 
        data.y.numpy(), 
        f"{method} - Renyi Model (α={renyi_value})", 
        f"{save_path_base}_renyi.pdf"
    )

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = Planetoid(root=f'./data/{args.dataset}', name=args.dataset)
    data = dataset[0]
    
    for method in ['diffpool', 'mincut', 'dmonpool']:
        base_model = torch.load(f'models/{args.dataset}_{method}_R0.0_seed{args.seed}.pt')
        renyi_model = torch.load(f'models/{args.dataset}_{method}_R{args.renyi}_seed{args.seed}.pt')
        
        plot_single_method(
            data, 
            base_model,
            renyi_model,
            method,
            args.renyi,
            device,
            f'plots/{args.dataset}_{method}_R{args.renyi}'
        )

if __name__ == '__main__':
    main()