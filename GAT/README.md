# GAT (Graph Attention Network) Implementation

## Overview

This folder contains the **Graph Attention Network (GAT)** implementation for cryptocurrency fraud detection on the Elliptic Bitcoin dataset.

## GAT Architecture

GAT uses attention mechanisms to learn different importance weights for different neighbors when aggregating information. This allows the model to focus on more relevant connections in the graph.

### Key Features:
- **Attention Mechanism**: Learns importance weights for neighbors
- **Multi-head Attention**: Captures different types of relationships
- **Flexible**: Can handle different graph structures
- **Interpretable**: Attention weights provide insights into important connections

## Implementation Details

### Model Architecture
- **Layers**: Multi-layer GAT with multi-head attention
- **Hidden Dimensions**: 64 (configurable)
- **Number of Layers**: 2 (configurable)
- **Attention Heads**: 8 (configurable)
- **Activation**: ELU
- **Dropout**: 0.5 (configurable)

### Training Parameters
- **Learning Rate**: 0.01 (configurable)
- **Batch Size**: 32 (for graph-level tasks)
- **Epochs**: 100 (configurable)
- **Optimizer**: Adam with weight decay (5e-4)
- **Loss Function**: CrossEntropyLoss
- **Learning Rate Scheduler**: ReduceLROnPlateau

## Files

- `gat_model.py` - Main GAT model implementation
- `train.py` - Training script
- `evaluate.py` - Evaluation script
- `utils.py` - Utility functions (if needed)

## Usage

### Prerequisites
Make sure you have the required packages installed:
```bash
pip install torch torch-geometric pandas numpy matplotlib scikit-learn
```

### Training
```bash
# Basic training with default parameters
python train.py --data_dir ../data

# Custom training with specific parameters
python train.py --data_dir ../data --epochs 100 --lr 0.01 --num_heads 8 --hidden_dim 64 --num_layers 2

# Use simple model (faster, fewer parameters)
python train.py --data_dir ../data --model_type simple --num_heads 4

# Use GPU (if available)
python train.py --data_dir ../data --device cuda
```

### Evaluation
```bash
# Evaluate trained model
python evaluate.py --model_path models/gat_best_model.pth --data_dir ../data

# Evaluate with specific model configuration
python evaluate.py --model_path models/gat_best_model.pth --data_dir ../data --model_type full --hidden_dim 64 --num_layers 2 --num_heads 8
```

## Results

[Results will be documented here after implementation]

### Metrics
- **Accuracy**: [To be filled]
- **F1 Score**: [To be filled]
- **Precision**: [To be filled]
- **Recall**: [To be filled]
- **ROC-AUC**: [To be filled]

## Author

[Your Name]

## References

- Veličković, P., et al. (2018). Graph Attention Networks. ICLR.
- [PyTorch Geometric GAT Documentation](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv)

