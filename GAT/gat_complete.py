"""
Graph Attention Network (GAT) for Cryptocurrency Fraud Detection
Author: Phlobater Habshy
Style inspired by CS4824 HW3 MNIST Implementation
"""

# ============================================================
# Imports
# ============================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, log_loss
)

# ============================================================
# Data Loading
# ============================================================

def load_elliptic_data(DataDir):
    """Load and preprocess the Elliptic dataset"""

    FeaturesPath = os.path.join(DataDir, "elliptic_txs_features.csv")
    ClassesPath  = os.path.join(DataDir, "elliptic_txs_classes.csv")
    EdgesPath    = os.path.join(DataDir, "elliptic_txs_edgelist.csv")

    if not (os.path.exists(FeaturesPath) and 
            os.path.exists(ClassesPath) and 
            os.path.exists(EdgesPath)):
        raise FileNotFoundError("Dataset files missing.")

    print("Loading features...")
    Features = pd.read_csv(FeaturesPath, header=None)
    NumFeatures = Features.shape[1] - 1
    Features.columns = ["txId"] + [f"f_{i}" for i in range(1, NumFeatures + 1)]

    print("Loading classes...")
    Classes = pd.read_csv(ClassesPath)

    print("Loading edges...")
    Edges = pd.read_csv(EdgesPath)
    Edges.columns = ["source", "target"]
    Edges["source"] = pd.to_numeric(Edges["source"], errors="coerce")
    Edges["target"] = pd.to_numeric(Edges["target"], errors="coerce")
    Edges = Edges.dropna().astype(int)

    print("Merging...")
    DF = Features.merge(Classes, on="txId", how="left")
    DF["label"] = DF["class"].map({"1": 1, "2": 0, "unknown": -1})

    # Normalize features
    X = DF[[f"f_{i}" for i in range(1, NumFeatures + 1)]].values.astype(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    NodeIds = DF["txId"].values
    MapId = {tx: i for i, tx in enumerate(NodeIds)}

    Src = Edges["source"].map(MapId)
    Dst = Edges["target"].map(MapId)
    Mask = Src.notna() & Dst.notna()

    EdgeIndex = torch.tensor(np.vstack([Src[Mask].values, Dst[Mask].values]), dtype=torch.long)
    Y = torch.tensor(DF["label"].values, dtype=torch.long)

    Labeled = torch.where(Y != -1)[0]
    N = len(Labeled)
    Perm = np.random.permutation(Labeled)

    TrainIdx = Perm[: int(0.7 * N)]
    ValIdx   = Perm[int(0.7 * N): int(0.85 * N)]
    TestIdx  = Perm[int(0.85 * N):]

    TrainMask = torch.zeros(len(Y), dtype=torch.bool); TrainMask[TrainIdx] = True
    ValMask   = torch.zeros(len(Y), dtype=torch.bool); ValMask[ValIdx] = True
    TestMask  = torch.zeros(len(Y), dtype=torch.bool); TestMask[TestIdx] = True

    print("\nDataset Loaded:")
    print(f" Nodes: {len(Y)}")
    print(f" Edges: {EdgeIndex.shape[1]}")
    print(f" Train: {TrainMask.sum().item()} | Val: {ValMask.sum().item()} | Test: {TestMask.sum().item()}\n")

    return Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=EdgeIndex,
        y=Y,
        train_mask=TrainMask,
        val_mask=ValMask,
        test_mask=TestMask,
    )

# ============================================================
# GAT Model
# ============================================================

class GAT(nn.Module):
    def __init__(self, InputDim, HiddenDim=64, Heads=8, Dropout=0.6):
        super(GAT, self).__init__()
        self.GAT1 = GATConv(InputDim, HiddenDim, heads=Heads, dropout=Dropout)
        self.GAT2 = GATConv(HiddenDim * Heads, HiddenDim, heads=1, dropout=Dropout)
        self.FC = nn.Linear(HiddenDim, 2)
        self.Dropout = Dropout

    def forward(self, Data):
        X, Edge = Data.x, Data.edge_index
        X = F.dropout(X, p=self.Dropout, training=self.training)
        X = F.elu(self.GAT1(X, Edge))
        X = F.dropout(X, p=self.Dropout, training=self.training)
        X = F.elu(self.GAT2(X, Edge))
        return self.FC(X)

# ============================================================
# Helpers
# ============================================================

def train_one_epoch(Model, Data, Optimizer, Criterion):
    Model.train()
    Optimizer.zero_grad()
    Out = Model(Data)
    Loss = Criterion(Out[Data.train_mask], Data.y[Data.train_mask])
    Loss.backward()
    Optimizer.step()
    return Loss.item()

def evaluate(Model, Data, Mask):
    Model.eval()
    with torch.no_grad():
        Out = Model(Data)[Mask]
        Labels = Data.y[Mask]
        Prob1 = torch.softmax(Out, dim=1)[:, 1].cpu().numpy()
        Preds = Out.argmax(dim=1).cpu().numpy()
        return Preds, Prob1, Labels.cpu().numpy()

# ============================================================
# Run One Full Training Run
# ============================================================

def run_training(DataObj, Epochs, LR, HiddenSize, Heads, Device):

    LabelsTrain = DataObj.y[DataObj.train_mask]
    FraudWeight = (LabelsTrain == 0).sum().item() / max(1, (LabelsTrain == 1).sum().item())

    Criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, FraudWeight], dtype=torch.float32).to(Device))

    Model = GAT(DataObj.x.size(1), HiddenDim=HiddenSize, Heads=Heads).to(Device)
    Optimizer = optim.Adam(Model.parameters(), lr=LR, weight_decay=5e-4)

    TrainLosses, ValLosses = [], []
    TrainAccs, ValAccs = [], []

    print("\n===== Training Run Start =====")
    for Epoch in range(Epochs):
        Loss = train_one_epoch(Model, DataObj, Optimizer, Criterion)

        PredT, _, YT = evaluate(Model, DataObj, DataObj.train_mask)
        TrainAcc = 100 * accuracy_score(YT, PredT)

        PredV, ProbV, YV = evaluate(Model, DataObj, DataObj.val_mask)
        ValAcc = 100 * accuracy_score(YV, PredV)
        ValLoss = log_loss(YV, ProbV, labels=[0, 1])

        TrainLosses.append(Loss)
        ValLosses.append(ValLoss)
        TrainAccs.append(TrainAcc)
        ValAccs.append(ValAcc)

        print(f"Epoch [{Epoch+1}/{Epochs}] | Train Loss: {Loss:.4f} | Val Loss: {ValLoss:.4f} | "
              f"Train Acc: {TrainAcc:.2f}% | Val Acc: {ValAcc:.2f}%")

    return Model, TrainLosses, ValLosses, TrainAccs, ValAccs

# ============================================================
# Plot Mean ± Std
# ============================================================

def plot_mean_std(TrainLoss, TrainStd, ValLoss, ValStd, TrainAcc, TrainAccStd, ValAcc, ValAccStd):
    Epochs = range(1, len(TrainLoss) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.title("Loss (Mean ± Std) Across Runs")
    plt.plot(Epochs, TrainLoss, label="Train Loss", color="blue")
    plt.plot(Epochs, ValLoss, label="Val Loss", color="orange")
    plt.fill_between(Epochs, TrainLoss - TrainStd, TrainLoss + TrainStd, color="blue", alpha=0.2)
    plt.fill_between(Epochs, ValLoss - ValStd, ValLoss + ValStd, color="orange", alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.title("Accuracy (Mean ± Std) Across Runs")
    plt.plot(Epochs, TrainAcc, label="Train Acc", color="green")
    plt.plot(Epochs, ValAcc, label="Val Acc", color="red")
    plt.fill_between(Epochs, TrainAcc - TrainAccStd, TrainAcc + TrainAccStd, color="green", alpha=0.2)
    plt.fill_between(Epochs, ValAcc - ValAccStd, ValAcc + ValAccStd, color="red", alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/gat_results_mean_std.png", dpi=300)
    plt.close()

# ============================================================
# Main (Runs Multiple Times)
# ============================================================

def main():

    # ----------------------------
    # Hyperparameters
    # ----------------------------
    Epochs = 40
    LR = 0.005
    HiddenSize = 64
    Heads = 8
    Runs = 3

    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DataObj = load_elliptic_data("../data").to(Device)

    AllTrainLoss, AllValLoss = [], []
    AllTrainAcc, AllValAcc = [], []

    # ----------------------------
    # Multiple Training Runs
    # ----------------------------
    for R in range(Runs):
        print(f"\n========== RUN {R+1}/{Runs} ==========")
        Model, TL, VL, TA, VA = run_training(DataObj, Epochs, LR, HiddenSize, Heads, Device)

        AllTrainLoss.append(TL)
        AllValLoss.append(VL)
        AllTrainAcc.append(TA)
        AllValAcc.append(VA)

        # Test performance for this run
        Pred, Prob, Labels = evaluate(Model, DataObj, DataObj.test_mask)

        print("\nTest Results:")
        print(classification_report(Labels, Pred))
        print("Confusion Matrix:")
        print(confusion_matrix(Labels, Pred))

    # ----------------------------
    # Mean ± Std Across Runs
    # ----------------------------
    AllTrainLoss  = np.array(AllTrainLoss)
    AllValLoss    = np.array(AllValLoss)
    AllTrainAcc   = np.array(AllTrainAcc)
    AllValAcc     = np.array(AllValAcc)

    plot_mean_std(
        AllTrainLoss.mean(0), AllTrainLoss.std(0),
        AllValLoss.mean(0),   AllValLoss.std(0),
        AllTrainAcc.mean(0),  AllTrainAcc.std(0),
        AllValAcc.mean(0),    AllValAcc.std(0)
    )

# ============================================================
# Run Script
# ============================================================

if __name__ == "__main__":
    main()
