# Quick Start Guide - Running the GAT Model

## Step 1: Install Dependencies

First, make sure you have all required packages installed:

```bash
# Install PyTorch (CPU version - adjust for CUDA if you have GPU)
pip install torch torchvision torchaudio

# Install PyTorch Geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric

# Install other dependencies
pip install pandas numpy matplotlib scikit-learn
```

**For GPU (CUDA) support:**
```bash
# First install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install PyTorch Geometric with CUDA
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric
```

## Step 2: Prepare Data

Make sure your data files are in the correct location:

1. Create a `data` folder in the repository root (if it doesn't exist):
   ```bash
   mkdir data
   ```

2. Copy your CSV files to the `data` folder:
   - `elliptic_txs_features.csv`
   - `elliptic_txs_classes.csv`
   - `elliptic_txs_edgelist.csv`

   The structure should look like:
   ```
   MachineLearningProject-CryptoFraudModel-/
   ├── data/
   │   ├── elliptic_txs_features.csv
   │   ├── elliptic_txs_classes.csv
   │   └── elliptic_txs_edgelist.csv
   ├── GAT/
   │   ├── train.py
   │   ├── evaluate.py
   │   └── gat_model.py
   └── data_loader.py
   ```

## Step 3: Navigate to GAT Folder

```bash
cd GAT
```

## Step 4: Run Training

### Basic Training (Default Parameters)
```bash
python train.py --data_dir ../data
```

### Training with Custom Parameters
```bash
python train.py \
    --data_dir ../data \
    --epochs 100 \
    --lr 0.01 \
    --hidden_dim 64 \
    --num_layers 2 \
    --num_heads 8 \
    --dropout 0.5 \
    --batch_size 32
```

### Training with Simple Model (Faster)
```bash
python train.py --data_dir ../data --model_type simple --num_heads 4
```

### Training on GPU (if available)
```bash
python train.py --data_dir ../data --device cuda
```

## Step 5: Evaluate the Model

After training, evaluate your model:

```bash
python evaluate.py \
    --model_path models/gat_best_model.pth \
    --data_dir ../data
```

### With Custom Model Configuration
```bash
python evaluate.py \
    --model_path models/gat_best_model.pth \
    --data_dir ../data \
    --model_type full \
    --hidden_dim 64 \
    --num_layers 2 \
    --num_heads 8
```

## Expected Output

### During Training:
- Progress updates every 10 epochs showing:
  - Epoch number
  - Training loss and accuracy
  - Validation loss and accuracy
  - Validation F1 score
- Training curves saved to `images/gat_training_curves.png`
- Best model saved to `models/gat_best_model.pth`

### During Evaluation:
- Comprehensive metrics (Accuracy, F1, Precision, Recall, ROC-AUC)
- Classification report
- Visualizations saved to `images/`:
  - `gat_confusion_matrix.png`
  - `gat_roc_curve.png`
  - `gat_precision_recall_curve.png`

## Troubleshooting

### Error: "No module named 'torch_geometric'"
- Install PyTorch Geometric (see Step 1)

### Error: "FileNotFoundError: data/elliptic_txs_features.csv"
- Make sure data files are in the `data/` folder
- Check that you're running from the `GAT/` directory
- Verify the `--data_dir` path is correct

### Error: "CUDA out of memory"
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use CPU: `--device cpu`
- Use simpler model: `--model_type simple`

### Error: "Data loading not yet implemented"
- Make sure `data_loader.py` is in the parent directory
- The data loader should be automatically imported

## Example Full Workflow

```bash
# 1. Navigate to GAT folder
cd GAT

# 2. Train the model (adjust parameters as needed)
python train.py --data_dir ../data --epochs 50 --lr 0.01

# 3. Wait for training to complete...

# 4. Evaluate the trained model
python evaluate.py --model_path models/gat_best_model.pth --data_dir ../data

# 5. Check results in images/ folder
```

## Command-Line Arguments Reference

### train.py Arguments:
- `--data_dir`: Directory containing CSV files (default: `../data`)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.01)
- `--batch_size`: Batch size (default: 32)
- `--hidden_dim`: Hidden dimension size (default: 64)
- `--num_layers`: Number of GAT layers (default: 2)
- `--num_heads`: Number of attention heads (default: 8)
- `--dropout`: Dropout probability (default: 0.5)
- `--model_type`: Model type - 'full' or 'simple' (default: 'full')
- `--device`: Device to use - 'cuda' or 'cpu' (default: auto-detect)
- `--save_dir`: Directory to save models (default: 'models')
- `--save_plots`: Directory to save plots (default: 'images')

### evaluate.py Arguments:
- `--model_path`: Path to saved model (required)
- `--data_dir`: Directory containing CSV files (default: `../data`)
- `--model_type`: Model type - 'full' or 'simple' (default: 'full')
- `--hidden_dim`: Hidden dimension size (must match training)
- `--num_layers`: Number of GAT layers (must match training)
- `--num_heads`: Number of attention heads (must match training)
- `--dropout`: Dropout probability (must match training)
- `--device`: Device to use - 'cuda' or 'cpu' (default: auto-detect)
- `--output_dir`: Directory to save plots (default: 'images')

