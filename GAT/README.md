# GAT (Graph Attention Network) Implementation

## Overview

This folder contains a **complete Graph Attention Network (GAT)** implementation for cryptocurrency fraud detection on the Elliptic Bitcoin dataset. The code trains a GAT model to classify Bitcoin transactions as either **licit (legitimate)** or **illicit (fraudulent)** based on transaction features and graph structure.

## What the Code Does

### 1. **Data Loading and Preprocessing**
- Loads three CSV files:
  - `elliptic_txs_features.csv`: 166 features per transaction
  - `elliptic_txs_classes.csv`: Transaction labels ('1' = illicit, '2' = licit)
  - `elliptic_txs_edgelist.csv`: Graph edges representing transaction flows
- Merges features with labels
- Normalizes features (zero mean, unit variance)
- Creates node-to-index mapping for graph construction
- Builds the graph structure (edge_index) from the edgelist
- Splits data into train (70%), validation (15%), and test (15%) sets

### 2. **Model Architecture**
The GAT model consists of:
- **First GAT Layer**: Multi-head attention (8 heads) with hidden dimension 64
- **Second GAT Layer**: Single-head attention for final feature extraction
- **Fully Connected Layer**: Binary classification output (2 classes)
- **Dropout**: 0.6 for regularization
- **Activation**: ELU (Exponential Linear Unit)

### 3. **Training Process**
The code performs **3 independent training runs** to ensure reproducibility and measure variance:
- Each run trains for 40 epochs
- Uses **class-weighted loss** to handle class imbalance (more licit than illicit transactions)
- Optimizer: Adam with learning rate 0.005 and weight decay 5e-4
- Tracks training and validation loss/accuracy at each epoch
- All output is logged to both console and `sample_output.txt`

### 4. **Evaluation**
After each training run:
- Evaluates on the test set
- Computes classification metrics:
  - Precision, Recall, F1-Score for each class
  - Overall accuracy
  - Confusion matrix
- Generates visualizations showing mean ± standard deviation across runs

### 5. **Output Generation**
- **Terminal Output**: Real-time training progress and results
- **sample_output.txt**: Complete log of all output (see `sample_output.txt` for example)
- **images/gat_results_mean_std.png**: Visualization of training curves with error bars

## GAT Architecture Details

GAT uses attention mechanisms to learn different importance weights for different neighbors when aggregating information. This allows the model to focus on more relevant connections in the graph.

### Key Features:
- **Attention Mechanism**: Learns importance weights for neighbors
- **Multi-head Attention**: Captures different types of relationships (8 heads)
- **Flexible**: Can handle different graph structures
- **Class-Weighted Loss**: Handles imbalanced dataset (more licit than illicit transactions)

## Implementation Details

### Model Architecture
- **Layers**: 2-layer GAT
- **Hidden Dimensions**: 64
- **Attention Heads**: 8 (first layer), 1 (second layer)
- **Activation**: ELU
- **Dropout**: 0.6

### Training Parameters
- **Learning Rate**: 0.005
- **Epochs**: 40 per run
- **Number of Runs**: 3 (for statistical reliability)
- **Optimizer**: Adam with weight decay (5e-4)
- **Loss Function**: CrossEntropyLoss with class weights

## Files

- `Habshy_GAT_Model.py` - Complete implementation (data loading, model, training, evaluation)
- `sample_output.txt` - Example output showing training progress and results
- `images/gat_results_mean_std.png` - Training curves visualization
- `README.md` - This file

## Usage

### Prerequisites
Make sure you have the required packages installed:
```bash
pip install torch torch-geometric pandas numpy matplotlib scikit-learn tqdm
```

### Running the Code
```bash
cd GAT
python Habshy_GAT_Model.py
```

The code will:
1. Load data from `../data/` directory
2. Train 3 models independently
3. Evaluate each model on the test set
4. Generate visualizations
5. Save all output to `sample_output.txt`

### Expected Output

When you run the code, you'll see:

1. **Data Loading**:
   ```
   Loading features...
   Loading classes...
   Loading edges...
   Merging...
   
   Dataset Loaded:
    Nodes: 203769
    Edges: 234355
    Train: 32594 | Val: 6985 | Test: 6985
   ```

2. **Training Progress** (for each of 3 runs):
   ```
   ========== RUN 1/3 ==========
   ===== Training Run Start =====
   Epoch [1/40] | Train Loss: 1.4419 | Val Loss: 0.3834 | Train Acc: 79.49% | Val Acc: 79.70%
   Epoch [2/40] | Train Loss: 0.7599 | Val Loss: 0.4158 | Train Acc: 77.45% | Val Acc: 77.34%
   ...
   ```

3. **Test Results** (after each run):
   ```
   Test Results:
                 precision    recall  f1-score   support
   
              0       0.99      0.71      0.82      6273
              1       0.27      0.95      0.42       712
   
       accuracy                           0.73      6985
   ```

4. **Summary Statistics**:
   ```
   ========================================
   SUMMARY OF RESULTS ACROSS 3 RUNS:
   ========================================
   
   Average Test Accuracy: ~72%
   Images saved to: images/gat_results_mean_std.png
   Output saved to: sample_output.txt
   ```

See `sample_output.txt` for a complete example of the output.

## Results Interpretation

### Performance Metrics

Based on the sample output, the model achieves:
- **Accuracy**: ~72-73%
- **Precision (Class 0 - Licit)**: 0.99 (very high - correctly identifies legitimate transactions)
- **Precision (Class 1 - Fraud)**: 0.27 (lower - some false positives)
- **Recall (Class 0 - Licit)**: 0.69-0.71 (good)
- **Recall (Class 1 - Fraud)**: 0.95-0.96 (excellent - catches most fraud cases)

### Key Insights

1. **High Fraud Recall (95%)**: The model successfully identifies most fraudulent transactions, which is critical for fraud detection.

2. **Lower Fraud Precision (27%)**: The model flags many transactions as fraud that are actually legitimate. This is a trade-off - better to catch fraud even if it means more false alarms.

3. **High Licit Precision (99%)**: When the model says a transaction is legitimate, it's almost always correct.

4. **Confusion Matrix**: Shows the model correctly identifies most licit transactions but has some false positives for fraud detection.

### Class Imbalance Handling

The dataset is highly imbalanced (many more licit than illicit transactions). The code uses:
- **Class-weighted loss**: Gives more importance to the minority class (fraud)
- This helps the model learn to detect fraud despite the imbalance

## Output Files

- **sample_output.txt**: Complete log of all training runs, metrics, and results
- **images/gat_results_mean_std.png**: Training curves showing:
  - Training and validation loss over epochs (with error bars)
  - Training and validation accuracy over epochs (with error bars)
  - Mean and standard deviation across 3 runs

## Code Structure

The code is organized into clear sections:

1. **Imports**: All required libraries
2. **Output Logger**: TeeOutput class to write to both console and file
3. **Data Loading**: `load_elliptic_data()` function
4. **Model Definition**: `GAT` class (PyTorch module)
5. **Training Functions**: `train_one_epoch()`, `evaluate()`, `run_training()`
6. **Visualization**: `plot_mean_std()` function
7. **Main Function**: Orchestrates the entire pipeline

## Author

Phlobater Habshy

## References

- Veličković, P., et al. (2018). Graph Attention Networks. ICLR.
- [PyTorch Geometric GAT Documentation](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv)
- Elliptic Bitcoin Dataset: [Paper](https://arxiv.org/abs/1908.02591)
