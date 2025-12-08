# Graph Neural Networks for Cryptocurrency Fraud Detection

## Overview

This branch contains implementations of **three different Graph Neural Network (GNN) architectures** for detecting fraudulent cryptocurrency transactions in the Elliptic Bitcoin dataset. Each team member is responsible for implementing one GNN model.

## Project Structure

```
GNNs/
├── data/
│   ├── elliptic_txs_features.csv      # Node features
│   ├── elliptic_txs_classes.csv       # Node labels
│   └── elliptic_txs_edgelist.csv     # Edge list (graph structure)
├── images/                            # Output visualizations
├── GNN_Model_1.py                     # First GNN implementation
├── GNN_Model_2.py                     # Second GNN implementation
├── GNN_Model_3.py                     # Third GNN implementation
├── data_loader.py                     # Data preprocessing and loading utilities
└── README_GNNs.md                     # This file
```

## Team Responsibilities

Each team member should implement one of the three GNN models:

### Model 1: [Team Member 1 Name]
- **File**: `GNN_Model_1.py`
- **Architecture**: [Specify GNN type, e.g., Graph Convolutional Network (GCN)]
- **Status**: In progress

### Model 2: [Team Member 2 Name]
- **File**: `GNN_Model_2.py`
- **Architecture**: [Specify GNN type, e.g., Graph Attention Network (GAT)]
- **Status**: In progress

### Model 3: [Team Member 3 Name]
- **File**: `GNN_Model_3.py`
- **Architecture**: [Specify GNN type, e.g., Graph Transformer]
- **Status**: In progress

## Dataset

The Elliptic Bitcoin dataset consists of:
- **Nodes**: Bitcoin transactions
- **Edges**: Transaction flows (money transfers between transactions)
- **Features**: 166 numerical features per transaction
- **Labels**: Binary classification (0 = licit, 1 = illicit/fraudulent)

### Data Files
- `data/elliptic_txs_features.csv`: Transaction features (166 features per transaction)
- `data/elliptic_txs_classes.csv`: Transaction labels ('1' = illicit, '2' = licit)
- `data/elliptic_txs_edgelist.csv`: Edge list representing transaction flows

## Requirements

### Python Packages
```bash
pip install torch torch-geometric pandas numpy matplotlib scikit-learn
```

### PyTorch Geometric Installation
```bash
# For CUDA (if using GPU):
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric

# For CPU only:
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
```

## Implementation Guidelines

### 1. Data Loading
- Use `data_loader.py` to load and preprocess the dataset
- Convert CSV data to PyTorch Geometric `Data` format
- Create train/validation/test splits
- Handle class imbalance if present

### 2. Model Architecture
Each GNN model should:
- Inherit from `torch.nn.Module`
- Implement `__init__()` for layer definition
- Implement `forward()` for forward pass
- Use appropriate GNN layers (GCN, GAT, GraphSAGE, TransformerConv, etc.)

### 3. Training
- Implement training loop with:
  - Loss function (e.g., CrossEntropyLoss)
  - Optimizer (e.g., Adam)
  - Learning rate scheduling (optional)
  - Early stopping (optional)
- Track training/validation metrics
- Save model checkpoints

### 4. Evaluation
- Evaluate on test set
- Calculate metrics:
  - Accuracy
  - F1 Score
  - Precision
  - Recall
  - ROC-AUC
  - Confusion Matrix
- Generate visualizations (ROC curve, confusion matrix, training curves)

### 5. Documentation
- Document your model architecture
- Explain design choices
- Include hyperparameters used
- Report results and comparisons

## Suggested GNN Architectures

Here are some popular GNN architectures you might consider:

1. **Graph Convolutional Network (GCN)**
   - Simple and effective
   - Good baseline model
   - Use `GCNConv` from `torch_geometric.nn`

2. **Graph Attention Network (GAT)**
   - Attention mechanism for neighbor aggregation
   - Can learn different importance for different neighbors
   - Use `GATConv` from `torch_geometric.nn`

3. **GraphSAGE**
   - Sample and aggregate approach
   - Good for large graphs
   - Use `SAGEConv` from `torch_geometric.nn`

4. **Graph Transformer**
   - Transformer architecture adapted for graphs
   - State-of-the-art performance
   - Use `TransformerConv` from `torch_geometric.nn`

5. **Graph Isomorphism Network (GIN)**
   - Powerful for graph classification
   - Use `GINConv` from `torch_geometric.nn`

## Workflow

1. **Setup** (Everyone)
   - Clone the repository
   - Checkout the `GNNs` branch
   - Install dependencies
   - Review the dataset structure

2. **Data Preprocessing** (Collaborate)
   - Implement `data_loader.py` together
   - Ensure consistent data format across all models
   - Create train/val/test splits

3. **Model Implementation** (Individual)
   - Each member implements their assigned model
   - Test on a small subset first
   - Iterate and improve

4. **Training & Evaluation** (Individual)
   - Train your model
   - Tune hyperparameters
   - Evaluate and document results

5. **Comparison** (Together)
   - Compare all three models
   - Discuss strengths/weaknesses
   - Create final report

## Code Style

- Follow PEP 8 Python style guide
- Add docstrings to all functions and classes
- Include type hints where possible
- Comment complex logic
- Use meaningful variable names

## Git Workflow

1. Create a feature branch for your model: `git checkout -b feature/model-1`
2. Make your changes and commit frequently
3. Push your branch: `git push -u origin feature/model-1`
4. Create a Pull Request to merge into `GNNs` branch
5. Review each other's code before merging

## Resources

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [PyTorch Geometric Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html)
- [Graph Neural Networks: A Review](https://arxiv.org/abs/1812.08434)
- [Elliptic Dataset Paper](https://arxiv.org/abs/1908.02591)

## Notes

- The dataset is large (~658 MB for features CSV). Consider using Git LFS if needed.
- Graph structure is important - make sure edge_index is correctly constructed.
- Handle class imbalance (there are typically more licit than illicit transactions).
- Consider using GPU for training if available.

## Questions?

If you have questions or need help, create an issue in the repository or discuss with the team.

---

**Good luck with your implementations! 🚀**

