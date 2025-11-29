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
- **Hidden Dimensions**: [To be specified]
- **Number of Layers**: [To be specified]
- **Attention Heads**: [To be specified]
- **Activation**: ELU (or ReLU)
- **Dropout**: [To be specified]

### Training Parameters
- **Learning Rate**: [To be specified]
- **Batch Size**: [To be specified]
- **Epochs**: [To be specified]
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

## Files

- `gat_model.py` - Main GAT model implementation
- `train.py` - Training script
- `evaluate.py` - Evaluation script
- `utils.py` - Utility functions (if needed)

## Usage

### Training
```bash
python train.py --data_dir ../data --epochs 100 --lr 0.01 --heads 8
```

### Evaluation
```bash
python evaluate.py --model_path models/gat_model.pth --data_dir ../data
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

