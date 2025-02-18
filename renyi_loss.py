import torch
from torch.nn import functional as F

def renyi_loss(hidden,adj,epsilon=0.1, lambda_reg=0.1):
    hidden_norm = F.normalize(hidden, p=2, dim=1)
    sim_matrix = torch.mm(hidden_norm, hidden_norm.t())
    kernel_matrix = torch.exp(-torch.pow(sim_matrix, 2) / (2 * epsilon ** 2))
    kernel_matrix = kernel_matrix*adj
    n_samples = hidden.size(0)
    V = torch.sum(sim_matrix) / (n_samples ** 2)
    entropy = -torch.log(V)
    return entropy * lambda_reg