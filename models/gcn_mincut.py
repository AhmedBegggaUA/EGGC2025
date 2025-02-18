from typing import List, Tuple, Optional
import torch
import torch.nn as nn
#from layers.gcn import GCN
from torch_geometric.nn import GCNConv as GCN
from layers.mincut import MincutPooling
from renyi_loss import renyi_loss
class GCNMincut(nn.Module):
    """GCN model with Mincut pooling layer.
    
    Args:
        input_dim (int): Dimensión de las características de entrada
        channel_sizes (List[int]): Lista de tamaños de canales para las capas GCN y pooling
        orthogonality_regularization (float): Peso para la regularización de ortogonalidad
        cluster_size_regularization (float): Peso para la regularización del tamaño de clusters
        dropout_rate (float): Tasa de dropout
        pooling_mlp_sizes (List[int]): Tamaños de las capas ocultas del MLP en la capa de pooling
    """
    
    def __init__(self,
                 input_dim: int,
                 channel_sizes: List[int] = [64],
                 orthogonality_regularization: float = 1.0,
                 cluster_size_regularization: float = 0.0,
                 dropout_rate: float = 0.0,
                 pooling_mlp_sizes: Optional[List[int]] = None,
                 renyi_loss: float = 0.0):
        super(GCNMincut, self).__init__()
        
        self.channel_sizes = channel_sizes
        pooling_mlp_sizes = [] if pooling_mlp_sizes is None else pooling_mlp_sizes
        
        # Construir capas GCN
        self.gcn_layers = nn.ModuleList()
        current_dim = input_dim
        for n_channels in channel_sizes[:-1]:
            self.gcn_layers.append(GCN(current_dim, n_channels))
            current_dim = n_channels
            
        # Capa de pooling
        self.pool_layer = MincutPooling(
            k=channel_sizes[-1],
            do_unpool=False,
            orthogonality_regularization=orthogonality_regularization,
            cluster_size_regularization=cluster_size_regularization,
            dropout_rate=dropout_rate,
            mlp_sizes=pooling_mlp_sizes,
            renyi_lambda=renyi_loss
        )#.to('cuda:0')
        
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass del modelo.
        
        Args:
            inputs (Tuple[torch.Tensor, torch.Tensor]): Tuple conteniendo:
                - features: Características de los nodos [num_nodes, input_dim]
                - graph: Matriz de adyacencia dispersa
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple conteniendo:
                - pool: Características agrupadas
                - pool_assignment: Matriz de asignación de clusters
        """
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

def gcn_mincut(input_dim: int,
               channel_sizes: List[int],
               orthogonality_regularization: float = 1.0,
               cluster_size_regularization: float = 0.0,
               dropout_rate: float = 0.0,
               pooling_mlp_sizes: Optional[List[int]] = None) -> nn.Module:
    """Función constructora para el modelo GCNMincut.
    
    Args:
        input_dim: Dimensión de las características de entrada
        channel_sizes: Lista de tamaños de canales para las capas GCN y pooling
        orthogonality_regularization: Peso para la regularización de ortogonalidad
        cluster_size_regularization: Peso para la regularización del tamaño de clusters
        dropout_rate: Tasa de dropout
        pooling_mlp_sizes: Tamaños de las capas ocultas del MLP en la capa de pooling
        
    Returns:
        Modelo GCNMincut inicializado
    """
    return GCNMincut(
        input_dim,
        channel_sizes,
        orthogonality_regularization,
        cluster_size_regularization,
        dropout_rate,
        pooling_mlp_sizes
    )