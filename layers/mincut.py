from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.sparse import mm as sparse_dense_mm
from renyi_loss import renyi_loss
class MincutPooling(nn.Module):
    """Mincut pooling layer implementation in PyTorch.
    
    This layer implements a pooling operation that minimizes the normalized cut
    of the graph while learning a clustering of the nodes.
    
    Args:
        k (int): Number of clusters to pool into
        orthogonality_regularization (float): Weight for orthogonality loss
        cluster_size_regularization (float): Weight for cluster size loss
        dropout_rate (float): Dropout probability
        mlp_sizes (List[int], optional): Hidden layer sizes for the assignment MLP
        do_unpool (bool): Whether to perform unpooling operation
    """
    
    def __init__(self,
                 k: int,
                 orthogonality_regularization: float = 1.0,
                 cluster_size_regularization: float = 1.0,
                 dropout_rate: float = 0,
                 mlp_sizes: Optional[List[int]] = None,
                 do_unpool: bool = True,
                 renyi_lambda: float = 0.0):
        super(MincutPooling, self).__init__()
        self.k = k
        self.orthogonality_regularization = orthogonality_regularization
        self.cluster_size_regularization = cluster_size_regularization
        self.dropout_rate = dropout_rate
        self.mlp_sizes = [] if mlp_sizes is None else mlp_sizes
        self.do_unpool = do_unpool
        
        # El MLP se inicializará en el primer forward pass
        self.mlp = None
        self.renyi_loss = None
        self.renyi_lambda = renyi_lambda
    def _build_mlp(self, input_dim: int) -> nn.Sequential:
        """Construye el MLP para las asignaciones de clusters.
        
        Args:
            input_dim (int): Dimensión de entrada de las características
            
        Returns:
            nn.Sequential: MLP construido
        """
        layers = []
        current_dim = input_dim
        
        # Capas ocultas
        for size in self.mlp_sizes:
            layers.append(nn.Linear(current_dim, size))
            layers.append(nn.SELU())
            current_dim = size
        
        # Capa final
        final_layer = nn.Linear(current_dim, self.k)
        nn.init.orthogonal_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        
        # Dropout
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))
            
        return nn.Sequential(*layers)
    
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the MincutPooling layer.
        
        Args:
            inputs (Tuple[torch.Tensor, torch.Tensor]): Tuple containing:
                - features: Node features tensor of shape [num_nodes, num_features]
                - graph: Sparse adjacency matrix
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - features_pooled: Pooled node features
                - assignments: Cluster assignment matrix
        """
        features, graph = inputs
        
        # Inicializar MLP si es necesario
        if self.mlp is None:
            self.mlp = self._build_mlp(features.size(-1))
        self.mlp = self.mlp.to(features.device)
        # Generar matriz de asignación
        assignments = F.softmax(self.mlp(features), dim=1)
        assignments_pool = assignments / torch.sum(assignments, dim=0)
        
        # Calcular grafo agrupado
        # Nota: Asumimos que graph es una matriz dispersa de PyTorch
        graph_pooled = sparse_dense_mm(graph, assignments).t()
        graph_pooled = torch.matmul(graph_pooled, assignments)
        
        # Pérdida espectral (normalized cut)
        numerator = torch.trace(graph_pooled)
        denominator = torch.matmul(
            assignments.t() * torch.sparse.sum(graph, dim=-1).to_dense(),
            assignments
        )
        denominator = torch.trace(denominator)
        spectral_loss = -(numerator / denominator)
        self.spectral_loss = spectral_loss
        
        # Pérdida de ortogonalidad
        pairwise = torch.matmul(assignments.t(), assignments)
        identity = torch.eye(self.k, device=features.device)
        orthogonality_loss = torch.norm(
            pairwise / torch.norm(pairwise) - 
            identity / torch.sqrt(torch.tensor(self.k, dtype=torch.float))
        )
        self.orthogonality_loss = self.orthogonality_regularization * orthogonality_loss
        
        # Pérdida de tamaño de cluster
        cluster_loss = torch.norm(torch.sum(pairwise, dim=1)) / graph.size(0) * \
                      torch.sqrt(torch.tensor(self.k, dtype=torch.float)) - 1
        self.cluster_loss = self.cluster_size_regularization * cluster_loss
        
        # Pooling de características
        features_pooled = torch.matmul(assignments_pool.t(), features)
        features_pooled = F.selu(features_pooled)
        
        # Unpooling opcional
        if self.do_unpool:
            features_pooled = torch.matmul(assignments_pool, features_pooled)
        # Ahora aplicamos la función de pérdida de Renyi
        self.renyi_loss = renyi_loss(assignments,graph,epsilon=0.1, lambda_reg=self.renyi_lambda)
        return features_pooled, assignments
    
    def get_losses(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna las pérdidas calculadas durante el forward pass.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Spectral, orthogonality, and cluster size losses
        """
        return self.spectral_loss, self.orthogonality_loss, self.cluster_loss,self.renyi_loss