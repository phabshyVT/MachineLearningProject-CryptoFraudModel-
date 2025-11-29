"""
Data Loader for GNN Models
===========================

This module handles loading and preprocessing the Elliptic Bitcoin dataset
for Graph Neural Network training.

The dataset should be converted from CSV format to PyTorch Geometric Data format.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


def load_elliptic_data(features_path, classes_path, edgelist_path):
    """
    Load the Elliptic Bitcoin dataset and convert to graph format.
    
    Args:
        features_path: Path to elliptic_txs_features.csv
        classes_path: Path to elliptic_txs_classes.csv
        edgelist_path: Path to elliptic_txs_edgelist.csv
    
    Returns:
        PyTorch Geometric Data object with:
            - x: Node features [num_nodes, num_features]
            - edge_index: Edge connectivity [2, num_edges]
            - y: Node labels [num_nodes]
            - train_mask: Boolean mask for training nodes
            - val_mask: Boolean mask for validation nodes
            - test_mask: Boolean mask for test nodes
    """
    print("Loading features...")
    # Load features CSV
    features = pd.read_csv(features_path, header=None)
    
    # Name columns (first column is txId, rest are features)
    num_features = features.shape[1] - 1
    feature_cols = ["txId"] + [f"f_{i}" for i in range(1, num_features + 1)]
    features.columns = feature_cols
    
    print("Loading classes...")
    # Load classes CSV
    classes = pd.read_csv(classes_path)
    
    print("Loading edge list...")
    # Load edgelist CSV
    edgelist = pd.read_csv(edgelist_path, header=None)
    edgelist.columns = ["source", "target"]
    
    print("Merging features with labels...")
    # Merge features with labels
    data = features.merge(classes, on="txId", how="left")
    
    # Keep only labeled examples ('1' = illicit, '2' = licit, 'unknown' = unlabeled)
    # For now, we'll use all data but mark unlabeled as -1
    data["label"] = data["class"].map({'1': 1, '2': 0, 'unknown': -1})
    
    # Create node ID mapping (from txId to sequential index)
    node_ids = data["txId"].values
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    print("Creating node features...")
    # Extract node features (exclude txId column)
    feature_cols_only = [f"f_{i}" for i in range(1, num_features + 1)]
    X = data[feature_cols_only].values.astype(np.float32)
    
    # Normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    print("Creating edge index...")
    # Create edge_index from edgelist
    # Only include edges where both nodes exist in our data
    valid_edges = []
    for _, row in edgelist.iterrows():
        source_id = row["source"]
        target_id = row["target"]
        
        if source_id in node_id_to_idx and target_id in node_id_to_idx:
            source_idx = node_id_to_idx[source_id]
            target_idx = node_id_to_idx[target_id]
            valid_edges.append([source_idx, target_idx])
    
    if len(valid_edges) == 0:
        raise ValueError("No valid edges found! Check that edgelist node IDs match feature node IDs.")
    
    edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
    
    print("Creating labels...")
    # Create labels (use -1 for unlabeled nodes)
    y = data["label"].values
    y = torch.tensor(y, dtype=torch.long)
    
    # Create train/val/test splits
    # Use time-based split if available, otherwise random split
    # For Elliptic dataset, we can use the time step information if available
    # Otherwise, use random split with stratification
    
    print("Creating train/val/test splits...")
    num_nodes = len(y)
    
    # Filter out unlabeled nodes for splitting
    labeled_mask = (y != -1)
    labeled_indices = torch.where(labeled_mask)[0].numpy()
    
    # Random split: 70% train, 15% val, 15% test
    np.random.seed(42)
    np.random.shuffle(labeled_indices)
    
    train_size = int(0.7 * len(labeled_indices))
    val_size = int(0.15 * len(labeled_indices))
    
    train_indices = labeled_indices[:train_size]
    val_indices = labeled_indices[train_size:train_size+val_size]
    test_indices = labeled_indices[train_size+val_size:]
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    print(f"Dataset statistics:")
    print(f"  Total nodes: {num_nodes}")
    print(f"  Labeled nodes: {labeled_mask.sum().item()}")
    print(f"  Edges: {edge_index.size(1)}")
    print(f"  Features per node: {X.shape[1]}")
    print(f"  Training nodes: {train_mask.sum().item()}")
    print(f"  Validation nodes: {val_mask.sum().item()}")
    print(f"  Test nodes: {test_mask.sum().item()}")
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    return data


def create_data_loaders(data, batch_size=32, train_ratio=0.7, val_ratio=0.15):
    """
    Create DataLoaders for training, validation, and testing.
    
    Note: For node-level classification on a single graph, we typically
    don't use DataLoaders. This function is kept for compatibility but
    may not be needed for the Elliptic dataset.
    
    Args:
        data: PyTorch Geometric Data object
        batch_size: Batch size for DataLoader (not used for single graph)
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Remaining proportion for testing
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        For single graph, returns the data object wrapped in lists
    """
    # For node-level classification on a single graph,
    # we return the data object directly (wrapped in lists for compatibility)
    return [data], [data], [data]


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")
    try:
        data = load_elliptic_data(
            "data/elliptic_txs_features.csv",
            "data/elliptic_txs_classes.csv",
            "data/elliptic_txs_edgelist.csv"
        )
        print("\nData loaded successfully!")
        print(f"Data object: {data}")
        print(f"Number of nodes: {data.x.size(0)}")
        print(f"Number of edges: {data.edge_index.size(1)}")
        print(f"Number of features: {data.x.size(1)}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure data files are in the 'data/' directory")
