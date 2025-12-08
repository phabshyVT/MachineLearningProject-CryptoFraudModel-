from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
import numpy as np

def load_data():
    features = pd.read_csv("./data/elliptic_txs_features.csv", header=None)
    classes = pd.read_csv("./data/elliptic_txs_classes.csv")
    edges = pd.read_csv("./data/elliptic_txs_edgelist.csv")

    # Assign column names to the features dataframe
    col_names = ["txId", "timestep"] + [f"f_{i}" for i in range(165)]
    features.columns = col_names

    # Merge features with classes based on the transaction ID
    df = features.merge(classes, on="txId", how="left")


    return df, edges

def clean_data(df):
    # Remove rows with any NaN values
    df_cleaned = df[df["class"].isin(['1', '2'])]

    # Dropping txId column
    df_cleaned = df_cleaned.drop(columns=["txId"])
    
    X = df_cleaned.drop(columns=["class"])
    y = df_cleaned["class"]

    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df_cleaned

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    # First split: train vs temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size + val_size,
        stratify=y,
        random_state=random_state
    )

    # Second split: val vs test
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_ratio,
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_naive_bayes(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # illicit class probability

    print("\n=== Naive Bayes Results ===\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    return model


def plot_confusion(y_test, y_pred):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_roc_curve(y_test, y_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


def plot_class_distribution(y):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title("Class Distribution (Legit = 0, Fraud = 1)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()


def print_key_stats(y_test, y_pred, y_proba):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = (tp + tn) / cm.sum()
    roc = roc_auc_score(y_test, y_proba)

    print("\n==== KEY NAIVE BAYES STATISTICS ====")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}  <-- MOST IMPORTANT")
    print(f"F1 Score:        {f1:.4f}")
    print(f"ROC-AUC:         {roc:.4f}")
    print("-------------------------------------")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}  <-- MISSED FRAUD")
    print(f"True Positives:  {tp}")
    print(f"True Negatives:  {tn}")
    print("-------------------------------------")
    fraud_rate = np.mean(y_test)
    print(f"Fraud % in Test Set: {fraud_rate*100:.2f}%")
    print("=====================================\n")



# Load and clean
df, edges = load_data()
X, y, df_cleaned = clean_data(df)

# Split into train/val/test
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Train & evaluate
model = train_naive_bayes(X_train, y_train, X_test, y_test)

# ======== NEW: Predictions for plots/statistics ========
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ======== GRAPHS ========
plot_confusion(y_test, y_pred)
plot_roc_curve(y_test, y_proba)
plot_class_distribution(y)

# ======== KEY STATISTICS ========
print_key_stats(y_test, y_pred, y_proba)
