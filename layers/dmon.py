from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.sparse import mm as sparse_dense_mm
from renyi_loss import renyi_loss

class DMoN(nn.Module):
    """Implementation of Deep Modularity Network (DMoN) layer.
    
    DMoN optimizes modularity clustering objective in a fully unsupervised mode,
    however, this implementation can also be used as a regularizer in a supervised
    graph neural network. Optionally, it does graph unpooling.
    
    Args:
        k (int): Number of clusters in the model
        collapse_regularization (float): Weight for collapse regularization
        dropout_rate (float): Dropout probability
        mlp_sizes (List[int], optional): Hidden layer sizes for the assignment MLP
        do_unpool (bool): Whether to perform unpooling operation
        renyi_lambda (float): Weight for Renyi loss
    """
    
    def __init__(self,
                 k: int,
                 collapse_regularization: float = 0.1,
                 dropout_rate: float = 0.0,
                 mlp_sizes: Optional[List[int]] = None,
                 do_unpool: bool = False,
                 renyi_lambda: float = 0.0):
        super(DMoN, self).__init__()
        self.k = k
        self.collapse_regularization = collapse_regularization
        self.dropout_rate = dropout_rate
        self.mlp_sizes = [] if mlp_sizes is None else mlp_sizes
        self.do_unpool = do_unpool
        self.renyi_lambda = renyi_lambda
        
        # MLP will be initialized in first forward pass
        self.mlp = None
        self.spectral_loss = None
        self.collapse_loss = None
        self.renyi_loss = None

    def _build_mlp(self, input_dim: int) -> nn.Sequential:
        """Builds the MLP for cluster assignments.
        
        Args:
            input_dim (int): Input feature dimension
            
        Returns:
            nn.Sequential: Built MLP
        """
        layers = []
        current_dim = input_dim
        
        # Hidden layers
        for size in self.mlp_sizes:
            layers.append(nn.Linear(current_dim, size))
            layers.append(nn.SELU())
            current_dim = size
        
        # Final layer
        final_layer = nn.Linear(current_dim, self.k)
        nn.init.orthogonal_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        
        # Dropout
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))
            
        return nn.Sequential(*layers)
    
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the DMoN layer.
        
        Args:
            inputs (Tuple[torch.Tensor, torch.Tensor]): Tuple containing:
                - features: Node features tensor of shape [num_nodes, num_features]
                - adjacency: Sparse adjacency matrix
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - features_pooled: Pooled node features
                - assignments: Cluster assignment matrix
        """
        features, adjacency = inputs
        
        # Initialize MLP if needed
        if self.mlp is None:
            self.mlp = self._build_mlp(features.size(-1))
        self.mlp = self.mlp.to(features.device)
        # Validations
        assert isinstance(features, torch.Tensor)
        assert features.dim() == 2
        assert adjacency.dim() == 2
        assert features.size(0) == adjacency.size(0)
        
        # Generate assignment matrix
        assignments = F.softmax(self.mlp(features), dim=1)
        cluster_sizes = torch.sum(assignments, dim=0)
        assignments_pooling = assignments / cluster_sizes
        
        # Calculate degrees
        degrees = torch.sparse.sum(adjacency, dim=0).to_dense()
        degrees = degrees.reshape(-1, 1)
        
        number_of_nodes = adjacency.size(1)
        number_of_edges = torch.sum(degrees)
        
        # Calculate pooled graph [k, k] as S^T*A*S
        graph_pooled = sparse_dense_mm(adjacency, assignments).t()
        graph_pooled = torch.matmul(graph_pooled, assignments)
        
        # Calculate rank-1 normalizer
        normalizer_left = torch.matmul(assignments.t(), degrees)
        normalizer_right = torch.matmul(degrees.t(), assignments)
        
        normalizer = torch.matmul(normalizer_left, normalizer_right) / (2 * number_of_edges)
        self.spectral_loss = -torch.trace(graph_pooled - normalizer) / (2 * number_of_edges)
        
        # Collapse loss
        self.collapse_loss = self.collapse_regularization * (
            torch.norm(cluster_sizes) / number_of_nodes * 
            torch.sqrt(torch.tensor(self.k, dtype=torch.float, device=features.device)) - 1
        )
        
        # Pool features
        features_pooled = torch.matmul(assignments_pooling.t(), features)
        features_pooled = F.selu(features_pooled)
        
        if self.do_unpool:
            features_pooled = torch.matmul(assignments_pooling, features_pooled)
        
        # Apply Renyi loss
        self.renyi_loss = renyi_loss(assignments, adjacency, epsilon=0.1, lambda_reg=self.renyi_lambda)
        
        return features_pooled, assignments
    
    def get_losses(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the losses calculated during forward pass.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Spectral, collapse, and Renyi losses
        """
        return self.spectral_loss, 0*self.collapse_loss, self.renyi_loss