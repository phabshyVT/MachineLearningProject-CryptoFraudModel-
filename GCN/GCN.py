import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def load_data():
    features = pd.read_csv("../data/elliptic_txs_features.csv", header=None)
    classes = pd.read_csv("../data/elliptic_txs_classes.csv")
    edges = pd.read_csv("../data/elliptic_txs_edgelist.csv")

    # Assign column names to the features dataframe
    col_names = ["txId", "timestep"] + [f"f_{i}" for i in range(165)]
    features.columns = col_names

    # Merge features with classes based on the transaction ID
    df = features.merge(classes, on="txId", how="left")


    return df, edges

def prepare_graph(df, edges):
    # Keep labeled nodes only
    df = df[df["class"].isin(['1', '2'])]
    df["label"] = df["class"].map({'1': 1, '2': 0})

    # Save mapping: txId → index (0..N-1)
    df = df.reset_index(drop=True)
    id_map = {txId: idx for idx, txId in enumerate(df["txId"])}

    # Create feature matrix X
    X = df[[f"f_{i}" for i in range(165)]].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float)

    # Create labels
    y = torch.tensor(df["label"].values, dtype=torch.long)

    # Build edge_index
    edge_list = edges.copy()
    edge_list = edge_list[edge_list["txId1"].isin(id_map.keys()) & edge_list["txId2"].isin(id_map.keys())]

    src = edge_list["txId1"].map(id_map).values
    dst = edge_list["txId2"].map(id_map).values

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Train/Val/Test masks
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=df["label"])
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=df.loc[train_idx]["label"])

    train_mask = torch.zeros(len(df), dtype=torch.bool)
    val_mask = torch.zeros(len(df), dtype=torch.bool)
    test_mask = torch.zeros(len(df), dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return Data(
        x=X,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

# GCN model definition
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
def train_gcn(data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(device)

    model = GCN(in_dim=data.x.size(1), hidden_dim=64, out_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        _, pred = out.max(dim=1)
        val_correct = int((pred[data.val_mask] == data.y[data.val_mask]).sum())
        val_total = int(data.val_mask.sum())
        val_acc = val_correct / val_total

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val Acc {val_acc:.4f}")

    return model

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("GCN Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def plot_roc(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')

    plt.title("GCN ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def plot_precision_recall(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="purple", linewidth=2)

    plt.title("GCN Precision–Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Graphs and model data
def evaluate_gcn(model, data):
    model.eval()

    logits = model(data.x, data.edge_index)
    probs = torch.softmax(logits, dim=1)[:, 1]  # fraud probability
    
    _, pred = logits.max(dim=1)

    test_pred = pred[data.test_mask].cpu()
    test_true = data.y[data.test_mask].cpu()
    test_probs = probs[data.test_mask].detach().cpu().numpy()

    acc = (test_pred == test_true).float().mean()

    print("\n=== GCN Test Results ===")
    print("Accuracy:", acc.item())

    plot_confusion_matrix(test_true, test_pred)
    plot_roc(test_true, test_probs)
    plot_precision_recall(test_true, test_probs)

    print("\nClassification Report:")
    print(classification_report(test_true, test_pred))



# Running gcn
df, edges = load_data()
graph_data = prepare_graph(df, edges)
model = train_gcn(graph_data)
evaluate_gcn(model, graph_data)