"""
Complete GAT (Graph Attention Network) Implementation for Cryptocurrency Fraud Detection
========================================================================================

This is a complete, self-contained implementation that includes:
- Data loading and preprocessing
- GAT model definitions
- Training functionality
- Evaluation and visualization

Run this file to train and evaluate the GAT model.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# DATA LOADING
# ============================================================================

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


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

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
        """Initialize the GAT model."""
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.use_pooling = use_pooling
        
        # First layer: input features -> hidden dimension
        self.conv1 = GATConv(
            num_features,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True
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
        if num_layers > 1:
            self.conv_final = GATConv(
                hidden_dim * num_heads if num_layers > 2 else hidden_dim * num_heads,
                hidden_dim,
                heads=1,
                dropout=dropout,
                concat=False
            )
        else:
            self.conv_final = GATConv(
                hidden_dim * num_heads,
                hidden_dim,
                heads=1,
                dropout=dropout,
                concat=False
            )
        
        # For graph-level pooling (if needed)
        if use_pooling:
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
        """Forward pass through the network."""
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
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        
        # Final classification
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        
        return x


class GATModelSimple(nn.Module):
    """Simplified GAT model with fewer parameters."""
    
    def __init__(
        self,
        num_features=166,
        hidden_dim=64,
        num_classes=2,
        num_heads=4,
        dropout=0.5
    ):
        """Initialize simplified GAT model."""
        super(GATModelSimple, self).__init__()
        
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


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch)
        
        # Get labels
        if hasattr(batch, 'y') and batch.y is not None:
            labels = batch.y
        else:
            continue
        
        # Handle different output shapes
        if out.dim() > 1 and out.size(0) != labels.size(0):
            labels = labels[:out.size(0)]
        
        # Calculate loss (only on labeled nodes)
        if hasattr(batch, 'train_mask') and batch.train_mask is not None:
            train_mask = batch.train_mask
            loss = criterion(out[train_mask], labels[train_mask])
        else:
            loss = criterion(out, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        if hasattr(batch, 'train_mask') and batch.train_mask is not None:
            preds = out[train_mask].argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels[train_mask].cpu().numpy())
        else:
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1


def validate(model, val_loader, criterion, device, mask_name='val_mask'):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            # Forward pass
            out = model(batch)
            
            # Get labels
            if hasattr(batch, 'y') and batch.y is not None:
                labels = batch.y
            else:
                continue
            
            # Get mask
            if hasattr(batch, mask_name) and getattr(batch, mask_name) is not None:
                mask = getattr(batch, mask_name)
                mask_labels = labels[mask]
                mask_out = out[mask]
            else:
                mask_labels = labels
                mask_out = out
            
            # Handle different output shapes
            if mask_out.dim() > 1 and mask_out.size(0) != mask_labels.size(0):
                mask_labels = mask_labels[:mask_out.size(0)]
            
            # Calculate loss
            loss = criterion(mask_out, mask_labels)
            total_loss += loss.item()
            
            # Get predictions and probabilities
            probs = F.softmax(mask_out, dim=1)
            preds = mask_out.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(mask_labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.0
    
    return avg_loss, accuracy, f1, precision, recall, roc_auc


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path='training_curves.png'):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Training curves saved to {save_path}")
    plt.close()


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix - GAT Model')
    plt.colorbar()
    
    labels = [
        ['True Negatives', 'False Positives'],
        ['False Negatives', 'True Positives']
    ]
    
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i,
                f"{labels[i][j]}\n{cm[i,j]}",
                ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black',
                fontsize=12,
            )
    
    plt.xticks([0, 1], ['Predicted Licit (0)', 'Predicted Fraud (1)'])
    plt.yticks([0, 1], ['Actual Licit (0)', 'Actual Fraud (1)'])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_probs, save_path='roc_curve.png'):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, color='darkorange', label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.title('ROC Curve - GAT Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to {save_path}")


def plot_precision_recall_curve(y_true, y_probs, save_path='precision_recall_curve.png'):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, color='purple')
    plt.title('Precision-Recall Curve - GAT Model')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Precision-Recall curve saved to {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Complete GAT Model for Fraud Detection')
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing data files')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GAT layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--model_type', type=str, default='full', choices=['full', 'simple'],
                       help='Model type: full or simple')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--save_plots', type=str, default='images', help='Directory to save plots')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only evaluate')
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model for evaluation')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_plots, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    try:
        data = load_elliptic_data(
            os.path.join(args.data_dir, 'elliptic_txs_features.csv'),
            os.path.join(args.data_dir, 'elliptic_txs_classes.csv'),
            os.path.join(args.data_dir, 'elliptic_txs_edgelist.csv')
        )
        data = data.to(device)
        
        # For node-level classification, we can use the full graph
        train_loader = [data]
        val_loader = [data]
        test_loader = [data]
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize model
    num_features = data.x.size(1)
    num_classes = len(torch.unique(data.y[data.y != -1])) if hasattr(data, 'y') and data.y is not None else 2
    
    if args.model_type == 'simple':
        model = GATModelSimple(
            num_features=num_features,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_heads=args.num_heads,
            dropout=args.dropout
        ).to(device)
    else:
        model = GATModel(
            num_features=num_features,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout
        ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    if not args.skip_training:
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Training loop
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        best_val_loss = float('inf')
        
        print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Val F1':<12}")
        print("-" * 70)
        
        for epoch in range(1, args.epochs + 1):
            # Train
            train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss, val_acc, val_f1, val_precision, val_recall, val_roc_auc = validate(
                model, val_loader, criterion, device
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Track metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Print progress
            if epoch % 10 == 0 or epoch == 1:
                print(f"{epoch:<8} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f} {val_f1:<12.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, os.path.join(args.save_dir, 'gat_best_model.pth'))
        
        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Plot training curves
        plot_training_curves(
            train_losses, val_losses, train_accs, val_accs,
            save_path=os.path.join(args.save_plots, 'gat_training_curves.png')
        )
        
        model_path = os.path.join(args.save_dir, 'gat_best_model.pth')
    else:
        # Load model for evaluation only
        if args.model_path is None:
            model_path = os.path.join(args.save_dir, 'gat_best_model.pth')
        else:
            model_path = args.model_path
        
        print(f"\nLoading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    
    # Evaluation
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    test_loss, test_acc, test_f1, test_precision, test_recall, test_roc_auc = validate(
        model, test_loader, criterion, device, mask_name='test_mask'
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  ROC-AUC: {test_roc_auc:.4f}")
    
    # Get detailed predictions for visualization
    model.eval()
    with torch.no_grad():
        out = model(data)
        test_mask = data.test_mask
        test_out = out[test_mask]
        test_labels = data.y[test_mask]
        
        probs = F.softmax(test_out, dim=1)
        preds = test_out.argmax(dim=1).cpu().numpy()
        labels = test_labels.cpu().numpy()
        probs_class1 = probs[:, 1].cpu().numpy()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Licit (0)', 'Fraud (1)']))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(labels, preds, 
                         save_path=os.path.join(args.save_plots, 'gat_confusion_matrix.png'))
    plot_roc_curve(labels, probs_class1,
                   save_path=os.path.join(args.save_plots, 'gat_roc_curve.png'))
    plot_precision_recall_curve(labels, probs_class1,
                                save_path=os.path.join(args.save_plots, 'gat_precision_recall_curve.png'))
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"Model saved to: {model_path}")
    print(f"Visualizations saved to: {args.save_plots}/")


if __name__ == '__main__':
    main()

