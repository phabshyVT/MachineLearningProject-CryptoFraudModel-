import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
print("Loading data...")

features = pd.read_csv("data/elliptic_txs_features.csv", header=None)
classes = pd.read_csv("data/elliptic_txs_classes.csv")

# Name columns
num_features = 166
feature_cols = ["txId"] + [f"f_{i}" for i in range(1, num_features + 1)]
features.columns = feature_cols

# Merge labels into features
data = features.merge(classes, on="txId", how="left")

# Keep only labeled examples ('1' = illicit, '2' = licit)
data = data[data["class"].isin(['1', '2'])]

# Map labels: '1' → illicit (1), '2' → licit (0)
data["label"] = data["class"].map({'1': 1, '2': 0})

# Prepare inputs
X = data[feature_cols[1:]].values  # 166 features
y = data["label"].values

print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")


# -------------------------------------------------------------
# 2. NORMALIZE FEATURES
# -------------------------------------------------------------
print("Normalizing...")

scaler = StandardScaler()
X = scaler.fit_transform(X)


# -------------------------------------------------------------
# 3. TRAIN/TEST SPLIT
# -------------------------------------------------------------
print("Splitting train/test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# -------------------------------------------------------------
# 4. SIGMOID FUNCTION
# -------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# -------------------------------------------------------------
# 5. PARAMETER INITIALIZATION
# -------------------------------------------------------------
def initialize_params(n_features):
    W = np.zeros((n_features, 1))
    b = 0
    return W, b


# -------------------------------------------------------------
# 6. FORWARD + BACKWARD PASS
# -------------------------------------------------------------
def propagate(W, b, X, y):
    m = X.shape[1]  # number of samples
    
    # Forward pass
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)

    # Cost function (log loss)
    cost = (-1 / m) * np.sum(
        y * np.log(A + 1e-9) + (1 - y) * np.log(1 - A + 1e-9)
    )
    
    # Backpropagation (gradients)
    dW = (1 / m) * np.dot(X, (A - y).T)
    db = (1 / m) * np.sum(A - y)

    return dW, db, cost


# -------------------------------------------------------------
# 7. PREDICTION FUNCTION
# -------------------------------------------------------------
def predict(W, b, X):
    X = X.T
    A = sigmoid(np.dot(W.T, X) + b)
    return (A > 0.5).astype(int).flatten()


# -------------------------------------------------------------
# 8. TRAIN (GRADIENT DESCENT)
# -------------------------------------------------------------
def train(X, y, X_val, y_val, learning_rate=0.1, iterations=500):
    X = X.T  # reshape to (features, m)
    y = y.reshape(1, -1)  # reshape to (1, m)

    n_features = X.shape[0]
    W, b = initialize_params(n_features)
    costs = []
    train_accs = []
    val_accs = []

    for i in range(iterations):
        dW, db, cost = propagate(W, b, X, y)

        # Gradient descent update
        W -= learning_rate * dW
        b -= learning_rate * db

        # Track metrics every iteration
        costs.append(cost)
        
        # Training accuracy
        train_pred = predict(W, b, X.T)
        train_accs.append(accuracy_score(y.flatten(), train_pred))
        
        # Validation accuracy
        val_pred = predict(W, b, X_val)
        val_accs.append(accuracy_score(y_val, val_pred))

        if i % 50 == 0:
            print(f"Iteration {i} | Cost={cost:.4f} | Train Acc={train_accs[-1]:.4f} | Val Acc={val_accs[-1]:.4f}")

    return W, b, costs, train_accs, val_accs


# -------------------------------------------------------------
# 9. TRAIN MODEL
# -------------------------------------------------------------
print("Training logistic regression (from scratch)...")

W, b, costs, train_accs, val_accs = train(
    X_train, y_train, X_test, y_test,
    learning_rate=0.1, iterations=500
)


# -------------------------------------------------------------
# 10. EVALUATE MODEL
# -------------------------------------------------------------
print("Evaluating model...")

y_pred = predict(W, b, X_test)
y_pred_proba = sigmoid(np.dot(W.T, X_test.T) + b).flatten()

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -------------------------------------------------------------
# 11. VISUALIZATION FUNCTIONS
# -------------------------------------------------------------
def plot_training_curves(costs, train_accs, val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(costs, 'b-', linewidth=2)
    axes[0].set_title("Training Cost vs Iterations")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost (Log Loss)")
    axes[0].grid(True)

    axes[1].plot(train_accs, 'g-', linewidth=2, label="Training Accuracy")
    axes[1].plot(val_accs, 'r-', linewidth=2, label="Validation Accuracy")
    axes[1].set_title("Accuracy vs Iterations")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("images/training_curves.png", dpi=300)
    plt.show()


def plot_confusion_matrix_emphasized(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - Fraud Detection")
    plt.colorbar()

    labels = [
        ["True Negatives", "False Positives"],
        ["False Negatives", "True Positives"]
    ]

    for i in range(2):
        for j in range(2):
            plt.text(
                j, i,
                f"{labels[i][j]}\n{cm[i,j]}",
                ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black",
                fontsize=12,
            )

    plt.xticks([0, 1], ["Predicted Licit (0)", "Predicted Fraud (1)"])
    plt.yticks([0, 1], ["Actual Licit (0)", "Actual Fraud (1)"])

    plt.tight_layout()
    plt.savefig("images/confusion_matrix_emphasized.png", dpi=300)
    plt.show()


def plot_prediction_distribution(y_pred):
    pred_counts = pd.Series(y_pred).value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    plt.bar(["Licit (0)", "Fraud (1)"], pred_counts.values,
            color=["blue", "red"], alpha=0.7)
    plt.title("Model Prediction Distribution")

    for i, v in enumerate(pred_counts.values):
        plt.text(i, v, str(v), ha="center", va="bottom")

    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("images/prediction_distribution.png", dpi=300)
    plt.show()


def plot_precision_recall(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, color='purple')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/precision_recall_curve.png", dpi=300)
    plt.show()


def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, color='darkorange', label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/roc_curve.png", dpi=300)
    plt.show()


def plot_class_distribution(y_train, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    train_counts = pd.Series(y_train).value_counts().sort_index()
    axes[0].bar(["Licit (0)", "Fraud (1)"], train_counts.values,
                color=["blue", "red"], alpha=0.7)
    axes[0].set_title("Training Class Distribution")

    test_counts = pd.Series(y_test).value_counts().sort_index()
    axes[1].bar(["Licit (0)", "Fraud (1)"], test_counts.values,
                color=["blue", "red"], alpha=0.7)
    axes[1].set_title("Test Class Distribution")

    plt.tight_layout()
    plt.savefig("images/class_distribution.png", dpi=300)
    plt.show()


# -------------------------------------------------------------
# 12. GENERATE ALL PLOTS
# -------------------------------------------------------------
print("\nGenerating graphs...")

plot_training_curves(costs, train_accs, val_accs)
plot_confusion_matrix_emphasized(y_test, y_pred)
plot_prediction_distribution(y_pred)
plot_precision_recall(y_test, y_pred_proba)
plot_roc_curve(y_test, y_pred_proba)
plot_class_distribution(y_train, y_test)

