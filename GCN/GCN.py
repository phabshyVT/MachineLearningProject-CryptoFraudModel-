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


# ============================================================
# Load Data
# ============================================================

def load_data():
    features = pd.read_csv("../data/elliptic_txs_features.csv", header=None)
    classes = pd.read_csv("../data/elliptic_txs_classes.csv")
    edges = pd.read_csv("../data/elliptic_txs_edgelist.csv")

    col_names = ["txId", "timestep"] + [f"f_{i}" for i in range(165)]
    features.columns = col_names

    df = features.merge(classes, on="txId", how="left")
    return df, edges


# ============================================================
# Build Graph
# ============================================================

def prepare_graph(df, edges):
    df = df[df["class"].isin(['1', '2'])]
    df["label"] = df["class"].map({'1': 1, '2': 0})

    df = df.reset_index(drop=True)
    id_map = {txId: idx for idx, txId in enumerate(df["txId"])}

    # Feature matrix
    X = df[[f"f_{i}" for i in range(165)]].values
    X = StandardScaler().fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)

    y = torch.tensor(df["label"].values, dtype=torch.long)

    # Edges
    edges = edges[edges["txId1"].isin(id_map.keys()) & edges["txId2"].isin(id_map.keys())]
    src = edges["txId1"].map(id_map).values
    dst = edges["txId2"].map(id_map).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Masks (randomized each run)
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


# ============================================================
# GCN Model Definition
# ============================================================

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


# ============================================================
# Single Run Training
# ============================================================

def train_gcn_once(data, epochs=50, lr=0.01):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(device)

    model = GCN(in_dim=data.x.size(1), hidden_dim=64, out_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Validation
        model.eval()
        _, pred = out.max(dim=1)
        val_correct = int((pred[data.val_mask] == data.y[data.val_mask]).sum())
        val_total = int(data.val_mask.sum())
        val_acc = val_correct / val_total
        val_accs.append(val_acc)

    # Test accuracy (for summary statistics)
    logits = model(data.x, data.edge_index)
    _, pred = logits.max(dim=1)
    test_correct = int((pred[data.test_mask] == data.y[data.test_mask]).sum())
    test_total = int(data.test_mask.sum())
    test_acc = test_correct / test_total

    return model, np.array(train_losses), np.array(val_accs), test_acc


# ============================================================
# Multi-Run GCN Executor
# ============================================================

def run_gcn_multiple(df, edges, runs=3, epochs=50):
    all_train_losses = []
    all_val_accs = []
    test_accs = []

    for r in range(runs):
        print(f"\n=== RUN {r+1}/{runs} ===")

        # Resplit dataset each run → realistic variance
        graph_data = prepare_graph(df.copy(), edges.copy())

        model, train_loss, val_acc, test_acc = train_gcn_once(graph_data, epochs=epochs)

        all_train_losses.append(train_loss)
        all_val_accs.append(val_acc)
        test_accs.append(test_acc)

    # Convert to numpy arrays
    all_train_losses = np.array(all_train_losses)
    all_val_accs = np.array(all_val_accs)

    # Compute mean & std
    train_mean = all_train_losses.mean(axis=0)
    train_std = all_train_losses.std(axis=0)
    val_mean = all_val_accs.mean(axis=0)
    val_std = all_val_accs.std(axis=0)

    # Plot results
    plot_gcn_mean_std(train_mean, train_std, val_mean, val_std)

    print("\n=== Test Accuracy Over Runs ===")
    print("Mean Test Accuracy:", np.mean(test_accs))
    print("Std Test Accuracy:", np.std(test_accs))

    return train_mean, val_mean, test_accs


# ============================================================
# Plotting
# ============================================================

def plot_gcn_mean_std(train_mean, train_std, val_mean, val_std):
    epochs = np.arange(1, len(train_mean) + 1)

    plt.figure(figsize=(10, 5))
    plt.suptitle("GCN Mean ± Std Across Runs")

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.title("Train Loss (Mean ± Std)")
    plt.plot(epochs, train_mean, label="Loss", color="blue")
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.title("Accuracy (Mean ± Std)")
    plt.plot(epochs, val_mean, label="Val Acc", color="green")
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# Run Script
# ============================================================

if __name__ == "__main__":
    df, edges = load_data()
    run_gcn_multiple(df, edges, runs=3, epochs=50)
