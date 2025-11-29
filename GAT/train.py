"""
Training Script for GAT Model
==============================

This script handles training the Graph Attention Network for fraud detection.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from gat_model import GATModel, GATModelSimple
import sys

# Add parent directory to path to import data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_elliptic_data, create_data_loaders


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
        
        # Get labels (node-level or graph-level)
        if hasattr(batch, 'y') and batch.y is not None:
            labels = batch.y
        else:
            # If no labels, skip this batch
            continue
        
        # Handle different output shapes
        if out.dim() > 1 and out.size(0) != labels.size(0):
            # Graph-level prediction but node-level labels
            # Use only first node's label (or average)
            labels = labels[:out.size(0)]
        
        # Calculate loss
        loss = criterion(out, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1


def validate(model, val_loader, criterion, device):
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
            
            # Handle different output shapes
            if out.dim() > 1 and out.size(0) != labels.size(0):
                labels = labels[:out.size(0)]
            
            # Calculate loss
            loss = criterion(out, labels)
            total_loss += loss.item()
            
            # Get predictions and probabilities
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Calculate ROC-AUC if binary classification
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


def main():
    parser = argparse.ArgumentParser(description='Train GAT Model for Fraud Detection')
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing data files')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
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
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_plots, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    try:
        data = load_elliptic_data(
            os.path.join(args.data_dir, 'elliptic_txs_features.csv'),
            os.path.join(args.data_dir, 'elliptic_txs_classes.csv'),
            os.path.join(args.data_dir, 'elliptic_txs_edgelist.csv')
        )
        data = data.to(device)
        
        # Create train/val/test masks if not present
        if not hasattr(data, 'train_mask'):
            # Simple split: 70% train, 15% val, 15% test
            num_nodes = data.x.size(0)
            indices = torch.randperm(num_nodes)
            train_size = int(0.7 * num_nodes)
            val_size = int(0.15 * num_nodes)
            
            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            data.train_mask[indices[:train_size]] = True
            data.val_mask[indices[train_size:train_size+val_size]] = True
            data.test_mask[indices[train_size+val_size:]] = True
        
        # For node-level classification, we can use the full graph
        train_loader = [data]
        val_loader = [data]
        test_loader = [data]
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Note: You need to implement load_elliptic_data in data_loader.py")
        return
    
    # Initialize model
    num_features = data.x.size(1)
    num_classes = len(torch.unique(data.y)) if hasattr(data, 'y') and data.y is not None else 2
    
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
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
    
    print("\nStarting training...")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Val F1':<12}")
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
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_f1, test_precision, test_recall, test_roc_auc = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  ROC-AUC: {test_roc_auc:.4f}")


if __name__ == '__main__':
    main()

