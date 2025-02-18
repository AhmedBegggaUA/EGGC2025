"""Main script for running GCN Mincut experiments with PyTorch Geometric datasets."""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import normalized_mutual_info_score
from models.gcn_mincut import GCNMincut
from models.gcn_diffpool import GCNDiffPool
from models.gcn_dmon import GCNDMoN
from metrics import conductance, modularity, pairwise_precision
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import os
def parse_args():
    parser = argparse.ArgumentParser(description='GCN Mincut experiments')
    parser.add_argument('--dataset', type=str, default='PubMed',
                      choices=['Cora', 'CiteSeer', 'PubMed', 'DBLP'],
                      help='Dataset name')
    parser.add_argument('--architecture', type=str, default='512',
                      help='Network architecture (comma-separated)')
    parser.add_argument('--method', type=str, default='mincut',
                        choices=['mincut', 'diffpool', 'dmonpool'],
                        help='Pooling method')
    parser.add_argument('--dropout', type=float, default=0.0,
                      help='Dropout rate')
    parser.add_argument('--runs', type=int, default=10,
                      help='Number of runs')
    parser.add_argument('--epochs', type=int, default=1000,
                      help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--renyi_loss', type=float, default=0.0,
                        help='Renyi loss hyperparameter')
    parser.add_argument('--log_file', default='log.txt',type=str,
                        help='Path to log file')
    return parser.parse_args()

def log_print(message, file):
    """Imprime y guarda en archivo el mensaje."""
    print(message)
    with open(file, 'a') as f:
        f.write(message + '\n')

def run_experiment(args, data, log_file,seed):
    """Ejecuta un experimento individual."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    adj_matrix = to_scipy_sparse_matrix(data.edge_index).tocsr()
    adj_tensor = torch.sparse_coo_tensor(
        data.edge_index, 
        torch.ones(data.edge_index.size(1)),
        (data.num_nodes, data.num_nodes)
    ).to(device)
    
    architecture = [int(x) for x in args.architecture.split(',')]
    n_clusters = len(np.unique(data.y.numpy()))
    
    log_print(f'Architecture: {architecture}, Clusters: {n_clusters}', log_file)
    log_print(f"Channels: {architecture + [n_clusters]}", log_file)
    
    if args.method == 'mincut':
        model = GCNMincut(
                            input_dim=data.num_features,
                            channel_sizes=architecture + [n_clusters],
                            dropout_rate=args.dropout,
                            renyi_loss=args.renyi_loss
                          ).to(device)
    elif args.method == 'diffpool':
        model = GCNDiffPool(
                            input_dim=data.num_features,
                            channel_sizes=architecture + [n_clusters],
                            dropout_rate=args.dropout,
                            renyi_lambda=args.renyi_loss
                          ).to(device)
    elif args.method == 'dmonpool':
        model = GCNDMoN(
                            input_dim=data.num_features,
                            channel_sizes=architecture + [n_clusters],
                            dropout_rate=args.dropout,
                            renyi_loss=args.renyi_loss
                          ).to(device)

    
    log_print(f"Número de parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad)}", log_file)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    features = data.x.to(device)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        _, assignments = model((features, adj_tensor))
        losses = model.get_pooling_losses()
        loss = sum(losses)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            log_print(f'Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}', log_file)
    # Después del for loop de seeds
    # Guardar todo el modelo, no solo state_dict
    model_path = f'models/{args.dataset}_{args.method}_R{args.renyi_loss}_seed{seed}.pt'
    os.makedirs('models', exist_ok=True)
    torch.save(model, model_path)
    model.eval()
    with torch.no_grad():
        _, assignments = model((features, adj_tensor))
        assignments = assignments.cpu().numpy()
        clusters = assignments.argmax(axis=1)
        
    true_labels = data.y.cpu().numpy()
    C = conductance(adj_matrix, clusters)
    Q = modularity(adj_matrix, clusters)
    NMI = normalized_mutual_info_score(true_labels, clusters, average_method='arithmetic')
    F1 = pairwise_precision(true_labels, clusters)
    
    return C, Q, NMI, F1

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    
    # Crear archivo de log
    with open(args.log_file, 'w') as f:
        f.write("")  # Limpiar archivo
    
    log_print(f'Running experiments for {args.dataset} dataset', args.log_file)
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(
            root='./data/' + args.dataset,
            name=args.dataset,
            #transform=NormalizeFeatures()
        )
    else:
        dataset = CitationFull(
            root='./data/' + args.dataset,
            name=args.dataset
        )
    data = dataset[0]
    
    results = {
        'C': [], 'Q': [], 'NMI': [], 'F1': []
    }
    
    seeds = range(10)
    for seed in seeds:
        log_print(f'\nRun {seed + 1}/{len(seeds)} - Seed: {seed}', args.log_file)
        
        fix_seed(seed)
        
        C, Q, NMI, F1 = run_experiment(args, data, args.log_file,seed)
        
        results['C'].append(C)
        results['Q'].append(Q)
        results['NMI'].append(NMI)
        results['F1'].append(F1)
        
        log_print(f'Run {seed + 1} results:', args.log_file)
        log_print(f'Conductance (C): {C:.4f}', args.log_file)
        log_print(f'Modularity (Q): {Q:.4f}', args.log_file)
        log_print(f'NMI: {NMI:.4f}', args.log_file)
        log_print(f'F1: {F1:.4f}', args.log_file)
    
    log_print('\nFinal Results:', args.log_file)
    final_results = {}
    for metric in ['C', 'Q', 'NMI', 'F1']:
        values = results[metric]
        mean = np.mean(values) * 100  # Multiplicar por 100
        std = np.std(values) * 100    # Multiplicar por 100
        log_print(f'{metric}: {mean:.4f} ± {std:.4f}', args.log_file)
        final_results[f'{metric}_mean'] = mean
        final_results[f'{metric}_std'] = std
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame({
        'Dataset': [args.dataset],
        'Method': [args.method],
        'Architecture': [args.architecture],
        'Dropout': [args.dropout],
        'Epochs': [args.epochs],
        'Renyi_Loss': [args.renyi_loss],
        'C_mean': [final_results['C_mean']],
        'C_std': [final_results['C_std']],
        'Q_mean': [final_results['Q_mean']],
        'Q_std': [final_results['Q_std']],
        'NMI_mean': [final_results['NMI_mean']],
        'NMI_std': [final_results['NMI_std']],
        'F1_mean': [final_results['F1_mean']],
        'F1_std': [final_results['F1_std']]
    })
    
    # Guardar resultados en CSV
    csv_file = 'results.csv'
    try:
        # Intentar leer el CSV existente
        existing_df = pd.read_csv(csv_file)
        # Concatenar con los nuevos resultados
        updated_df = pd.concat([existing_df, results_df], ignore_index=True)
        # Guardar todo
        updated_df.to_csv(csv_file, index=False)
        log_print(f"\nResults appended to existing {csv_file}", args.log_file)
    except FileNotFoundError:
        # Si el archivo no existe, crear uno nuevo
        results_df.to_csv(csv_file, index=False)
        log_print(f"\nNew results file created: {csv_file}", args.log_file)

if __name__ == '__main__':
    main()