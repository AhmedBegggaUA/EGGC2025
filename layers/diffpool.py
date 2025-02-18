from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv as GCN
from renyi_loss import renyi_loss

class DiffPooling(nn.Module):
    """Differentiable pooling layer implementation in PyTorch.
    
    This layer implements the differentiable pooling operation that learns a
    soft clustering of the input nodes, allowing for hierarchical representation
    learning on graphs.
    
    Args:
        k (int): Number of clusters to pool into
        dropout_rate (float): Dropout probability
        mlp_sizes (List[int], optional): Hidden layer sizes for the assignment MLP
        do_unpool (bool): Whether to perform unpooling operation
        renyi_lambda (float): Weight for Renyi loss
    """
    
    def __init__(self,
                 k: int,
                 dropout_rate: float = 0,
                 mlp_sizes: Optional[List[int]] = None,
                 do_unpool: bool = True,
                 renyi_lambda: float = 0.0):
        super(DiffPooling, self).__init__()
        self.k = k
        self.dropout_rate = dropout_rate
        self.mlp_sizes = [] if mlp_sizes is None else mlp_sizes
        self.do_unpool = do_unpool
        self.renyi_lambda = renyi_lambda
        
        # MLP will be initialized in first forward pass
        self.mlp = None
        self.renyi_loss = None
        self.linkprediction_loss = None
        self.entropy_loss = None

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
        """Forward pass of the DiffPooling layer.
        
        Args:
            inputs (Tuple[torch.Tensor, torch.Tensor]): Tuple containing:
                - features: Node features tensor of shape [num_nodes, num_features]
                - graph: Adjacency matrix tensor
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - features_pooled: Pooled node features
                - assignments: Cluster assignment matrix
        """
        features, graph = inputs
        
        # Initialize MLP if needed
        if self.mlp is None:
            self.mlp = self._build_mlp(features.size(-1))
        self.mlp = self.mlp.to(features.device)
        # Generate assignment matrix
        assignments = F.softmax(self.mlp(features), dim=1)
        assignments_pool = assignments / torch.sum(assignments, dim=0)
        
        # Calculate graph reconstruction
        graph_reconstruction = torch.matmul(assignments, assignments.transpose(-2, -1))
        
        # Link prediction loss - convert sparse graph to dense for the operation
        dense_graph = graph.to_dense()
        self.linkprediction_loss = torch.norm(dense_graph - graph_reconstruction)
        
        # Entropy loss
        entropy = -torch.sum(torch.mul(assignments, torch.log(assignments + 1e-8)), dim=-1)
        self.entropy_loss = torch.mean(entropy)
        
        # Pool features
        features_pooled = torch.matmul(assignments_pool.t(), features)
        features_pooled = F.selu(features_pooled)
        
        # Optional unpooling
        if self.do_unpool:
            features_pooled = torch.matmul(assignments_pool, features_pooled)
        
        # Apply Renyi loss
        self.renyi_loss = renyi_loss(assignments, graph, epsilon=0.1, lambda_reg=self.renyi_lambda)
        
        return features_pooled, assignments
    
    def get_losses(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the losses calculated during forward pass.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Link prediction, entropy, and Renyi losses
        """
        return self.linkprediction_loss, 0*self.entropy_loss, self.renyi_loss