from typing import Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.sparse import mm as sparse_dense_mm

class GCN(nn.Module):
    """Graph Convolutional Network layer.
    
    Args:
        n_channels (int): Número de canales de salida
        activation (Union[str, Callable, None]): Función de activación
        skip_connection (bool): Si se usa conexión residual
        no_features (bool): Si se usa un kernel sin características de entrada
    """
    
    def __init__(self,
                 n_channels: int,
                 activation: Union[str, Callable, None] = 'selu',
                 skip_connection: bool = True,
                 no_features: bool = False):
        super(GCN, self).__init__()
        
        self.n_channels = n_channels
        self.skip_connection = skip_connection
        self.no_features = no_features
        
        # Inicialización diferida de parámetros
        self.n_features = None
        self.kernel = None
        self.bias = None
        self.skip_weight = None
        
        # Configurar activación
        if isinstance(activation, str):
            if activation == 'selu':
                self.activation = F.selu
            else:
                raise ValueError(f'Activación desconocida: {activation}')
        elif callable(activation):
            self.activation = activation
        elif activation is None:
            self.activation = lambda x: x
        else:
            raise ValueError('Activación de tipo desconocido')
            
    def _build(self, input_shape: torch.Size):
        """Inicializa los parámetros de la capa.
        
        Args:
            input_shape: Forma del tensor de entrada
        """
        if self.no_features:
            self.n_features = input_shape[0]
        else:
            self.n_features = input_shape[-1]
            
        # Kernel
        self.kernel = nn.Parameter(
            torch.empty(self.n_features, self.n_channels))
        
        # Bias
        self.bias = nn.Parameter(torch.empty(self.n_channels))
        
        # Skip connection weight
        if self.skip_connection:
            self.skip_weight = nn.Parameter(torch.empty(self.n_channels))
        else:
            self.register_parameter('skip_weight', None)
            
        # Inicialización de parámetros
        nn.init.xavier_uniform_(self.kernel)
        nn.init.zeros_(self.bias)
        if self.skip_connection:
            nn.init.ones_(self.skip_weight)
    
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Forward pass de la capa GCN.
        
        Args:
            inputs (Tuple[torch.Tensor, torch.Tensor]): Tuple conteniendo:
                - features: Características de los nodos [batch_size, num_nodes, n_features]
                           o [num_nodes, n_features]
                - graph: Matriz de adyacencia dispersa
                
        Returns:
            torch.Tensor: Características transformadas
        """
        features, graph = inputs
        
        # Inicialización diferida
        if self.kernel is None:
            self._build(features.shape)
            
        # Computar salida base
        if self.no_features:
            output = self.kernel.expand(features.shape[0], -1, -1)
        else:
            output = torch.matmul(features, self.kernel)
            
        # Aplicar skip connection y multiplicación por grafo
        if self.skip_connection:
            output = output * self.skip_weight
        output = output + sparse_dense_mm(graph, output)
        
        # Aplicar bias y activación
        output = output + self.bias
        return self.activation(output)

    def extra_repr(self) -> str:
        """Información adicional para la representación de la capa."""
        return (f'n_channels={self.n_channels}, '
                f'skip_connection={self.skip_connection}, '
                f'no_features={self.no_features}')