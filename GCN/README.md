# GCN (Graph Convolutional Network) Implementation

## Overview

This folder contains the **Graph Convolutional Network (GCN)** implementation for cryptocurrency fraud detection on the Elliptic Bitcoin dataset.

## GCN Architecture

GCN is a semi-supervised learning method for graph-structured data. It uses a localized first-order approximation of spectral graph convolutions to learn node representations.

### Key Features:
- **Spectral Convolution**: Approximates spectral graph convolution
- **Layer-wise Propagation**: Information propagates through graph layers
- **Simple and Effective**: Good baseline for graph learning tasks
- **Efficient**: Fast training and inference

## Implementation Details

### Model Architecture
- **Layers**: Multi-layer GCN with ReLU activation
- **Hidden Dimensions**: [To be specified]
- **Number of Layers**: [To be specified]
- **Activation**: ReLU
- **Dropout**: [To be specified]

### Training Parameters
- **Learning Rate**: [To be specified]
- **Batch Size**: [To be specified]
- **Epochs**: [To be specified]
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

## Files

- `gcn_model.py` - Main GCN model implementation
- `train.py` - Training script
- `evaluate.py` - Evaluation script
- `utils.py` - Utility functions (if needed)

## Usage

### Training
```bash
python train.py --data_dir ../data --epochs 100 --lr 0.01
```

### Evaluation
```bash
python evaluate.py --model_path models/gcn_model.pth --data_dir ../data
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

- Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
- [PyTorch Geometric GCN Documentation](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)

