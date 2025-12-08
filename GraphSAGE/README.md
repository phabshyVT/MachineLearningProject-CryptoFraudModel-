# GraphSAGE Implementation

## Overview

This folder contains the **GraphSAGE (Graph Sample and Aggregate)** implementation for cryptocurrency fraud detection on the Elliptic Bitcoin dataset.

## GraphSAGE Architecture

GraphSAGE is a framework for inductive representation learning on large graphs. Unlike transductive methods, GraphSAGE can generate embeddings for nodes that were not seen during training.

### Key Features:
- **Sampling**: Samples a fixed-size neighborhood for each node
- **Aggregation**: Aggregates information from sampled neighbors
- **Inductive**: Can generalize to unseen nodes/graphs
- **Scalable**: Works well on large graphs

## Implementation Details

### Model Architecture
- **Layers**: Multi-layer GraphSAGE with mean/mean-pool aggregation
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

- `graphsage_model.py` - Main GraphSAGE model implementation
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
python evaluate.py --model_path models/graphsage_model.pth --data_dir ../data
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

- Hamilton, W., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.
- [PyTorch Geometric GraphSAGE Documentation](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv)

