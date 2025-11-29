"""
Evaluation Script for GAT Model
================================

This script evaluates a trained GAT model on test data and generates
comprehensive evaluation metrics and visualizations.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import numpy as np
from gat_model import GATModel, GATModelSimple
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_elliptic_data


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


def main():
    parser = argparse.ArgumentParser(description='Evaluate GAT Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing data files')
    parser.add_argument('--model_type', type=str, default='full', choices=['full', 'simple'],
                       help='Model type: full or simple')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GAT layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='images', help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
        
        # Create test mask if not present
        if not hasattr(data, 'test_mask'):
            num_nodes = data.x.size(0)
            indices = torch.randperm(num_nodes)
            test_size = int(0.15 * num_nodes)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask[indices[-test_size:]] = True
        
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
    
    # Load model weights
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    if 'epoch' in checkpoint:
        print(f"Model was trained for {checkpoint['epoch']} epochs")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    with torch.no_grad():
        out = model(data)
        
        # Get test predictions
        if hasattr(data, 'test_mask') and data.test_mask is not None:
            test_mask = data.test_mask
        else:
            test_mask = torch.ones(data.x.size(0), dtype=torch.bool)
        
        test_out = out[test_mask]
        test_labels = data.y[test_mask] if hasattr(data, 'y') and data.y is not None else None
        
        if test_labels is None:
            print("No labels found in data!")
            return
        
        # Get predictions
        probs = F.softmax(test_out, dim=1)
        preds = test_out.argmax(dim=1).cpu().numpy()
        labels = test_labels.cpu().numpy()
        probs_class1 = probs[:, 1].cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(labels, probs_class1)
    except:
        roc_auc = 0.0
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("="*70)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Licit (0)', 'Fraud (1)']))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(labels, preds, 
                         save_path=os.path.join(args.output_dir, 'gat_confusion_matrix.png'))
    plot_roc_curve(labels, probs_class1,
                   save_path=os.path.join(args.output_dir, 'gat_roc_curve.png'))
    plot_precision_recall_curve(labels, probs_class1,
                                save_path=os.path.join(args.output_dir, 'gat_precision_recall_curve.png'))
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()

