"""
Graph Attention Network (GAT) Model for Cryptocurrency Fraud Detection
========================================================================

This module implements a Graph Attention Network for binary classification
of cryptocurrency transactions as licit or illicit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data


class GATModel(nn.Module):
    """
    Graph Attention Network for fraud detection.
    
    Uses multi-head attention to learn different importance weights for neighbors
    when aggregating information in the graph.
    """
    
    def __init__(
        self,
        num_features=166,
        hidden_dim=64,
        num_classes=2,
        num_layers=2,
        num_heads=8,
        dropout=0.5,
        use_pooling=False
    ):
        """
        Initialize the GAT model.
        
        Args:
            num_features: Number of input features per node (default: 166)
            hidden_dim: Dimension of hidden layers (default: 64)
            num_classes: Number of output classes (default: 2 for binary classification)
            num_layers: Number of GAT layers (default: 2)
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout probability (default: 0.5)
            use_pooling: If True, use graph-level pooling (for graph classification)
                        If False, use node-level classification (default: False)
        """
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.use_pooling = use_pooling
        
        # First layer: input features -> hidden dimension
        # Multi-head attention: num_heads attention heads
        self.conv1 = GATConv(
            num_features,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True  # Concatenate attention heads
        )
        
        # Hidden layers
        self.convs = nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads if i == 0 else hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
        
        # Last layer: hidden dimension -> output classes
        # Single head for final classification
        if num_layers > 1:
            self.conv_final = GATConv(
                hidden_dim * num_heads if num_layers > 2 else hidden_dim * num_heads,
                hidden_dim,
                heads=1,
                dropout=dropout,
                concat=False  # Average attention heads for final layer
            )
        else:
            # If only one layer, directly output from first layer
            self.conv_final = GATConv(
                hidden_dim * num_heads,
                hidden_dim,
                heads=1,
                dropout=dropout,
                concat=False
            )
        
        # For graph-level pooling (if needed)
        if use_pooling:
            # Combine mean and max pooling
            self.pool_dim = hidden_dim * 2
        else:
            self.pool_dim = hidden_dim
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, data):
        """
        Forward pass through the network.
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node feature matrix [num_nodes, num_features]
                - edge_index: Graph connectivity [2, num_edges]
                - batch: Batch vector for batching multiple graphs (optional)
        
        Returns:
            Output logits [num_nodes, num_classes] or [batch_size, num_classes]
        """
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        
        # First GAT layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        
        # Hidden GAT layers
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(conv(x, edge_index))
        
        # Final GAT layer
        if self.num_layers > 1:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv_final(x, edge_index)
        
        # Graph-level pooling (if specified) or node-level classification
        if self.use_pooling and batch is not None:
            # Graph-level classification
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        # Otherwise, x is already node-level features
        
        # Final classification
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        
        return x


class GATModelSimple(nn.Module):
    """
    Simplified GAT model with fewer parameters.
    Good for faster training and smaller datasets.
    """
    
    def __init__(
        self,
        num_features=166,
        hidden_dim=64,
        num_classes=2,
        num_heads=4,
        dropout=0.5
    ):
        """
        Initialize simplified GAT model.
        
        Args:
            num_features: Number of input features per node
            hidden_dim: Dimension of hidden layers
            num_classes: Number of output classes
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(GATModelSimple, self).__init__()
        
        # Two-layer GAT
        self.conv1 = GATConv(
            num_features,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        self.conv2 = GATConv(
            hidden_dim * num_heads,
            num_classes,
            heads=1,
            dropout=dropout,
            concat=False
        )
        
        self.dropout = dropout
    
    def forward(self, data):
        """Forward pass."""
        x, edge_index = data.x, data.edge_index
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return x

