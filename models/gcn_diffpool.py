from typing import List, Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv as GCN
from layers.diffpool import DiffPooling

class GCNDiffPool(nn.Module):
    def __init__(self, input_dim: int, channel_sizes: List[int],
                 dropout_rate: float = 0.0, renyi_lambda: float = 0.0):
        super(GCNDiffPool, self).__init__()
        self.channel_sizes = channel_sizes
        
        # Construir capas GCN
        self.gcn_layers = nn.ModuleList()
        current_dim = input_dim
        for n_channels in channel_sizes[:-1]:
            self.gcn_layers.append(GCN(current_dim, n_channels))
            current_dim = n_channels
        
        # Capa de pooling
        self.pool_layer = DiffPooling(k=channel_sizes[-1], do_unpool=False,
                                        dropout_rate=dropout_rate, renyi_lambda=renyi_lambda)#.to('cuda:0')
        
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        features, graph = inputs
        output = features
        
        # Aplicar capas GCN
        for gcn_layer in self.gcn_layers:
            output = gcn_layer(output, graph)
            
        # Aplicar pooling
        pool, pool_assignment = self.pool_layer([output, graph])
        
        return pool, pool_assignment
    def get_pooling_losses(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Obtiene las pérdidas de la capa de pooling.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Mincut, orthogonality, and cluster size losses
        """
        return self.pool_layer.get_losses()
def gcn_diffpool(input_dim: int, channel_sizes: List[int]) -> nn.Module:
    """Función constructora para el modelo GCNDiffPool.
    
    Args:
        input_dim: Dimensión de las características de entrada
        channel_sizes: Lista de tamaños de canales para las capas GCN y pooling
        
    Returns:
        Modelo GCNDiffPool inicializado
    """
    return GCNDiffPool(input_dim, channel_sizes)