# Logistic Regression Fraud Detection Model

## Overview

This project implements a **Logistic Regression classifier from scratch** (using only NumPy) to detect fraudulent cryptocurrency transactions in the Elliptic Bitcoin dataset. The model is trained to classify transactions as either **licit (0)** or **illicit/fraudulent (1)** based on 166 transaction features.

## Dataset

The model uses the **Elliptic Bitcoin Dataset**, which contains:
- **Features**: 166 numerical features per transaction (stored in `elliptic_txs_features.csv`)
- **Labels**: Transaction classes stored in `elliptic_txs_classes.csv`
  - `'1'` = Illicit (fraudulent) transactions
  - `'2'` = Licit (legitimate) transactions

## Code Structure

The implementation is organized into 12 main sections:

### 1. **Data Loading** (Lines 14-41)
- Loads transaction features and class labels from CSV files
- Merges features with labels using transaction IDs
- Filters to keep only labeled examples (classes '1' and '2')
- Maps labels: '1' → 1 (fraud), '2' → 0 (licit)
- Prepares feature matrix `X` (166 features) and label vector `y`

### 2. **Feature Normalization** (Lines 44-50)
- Applies `StandardScaler` to normalize features (zero mean, unit variance)
- Essential for stable gradient descent optimization

### 3. **Train/Test Split** (Lines 53-60)
- Splits data into 80% training and 20% testing sets
- Uses stratified sampling to maintain class distribution
- Random seed set to 42 for reproducibility

### 4. **Sigmoid Function** (Lines 63-67)
- Implements the sigmoid activation function: `σ(z) = 1 / (1 + e^(-z))`
- Maps any real number to a value between 0 and 1 (probability)

### 5. **Parameter Initialization** (Lines 70-76)
- Initializes weights `W` (166×1 matrix) to zeros
- Initializes bias `b` to zero

### 6. **Forward and Backward Pass** (Lines 79-98)
- **Forward Pass**: Computes predictions using `Z = W^T·X + b` and `A = σ(Z)`
- **Cost Function**: Calculates binary cross-entropy (log loss)
- **Backpropagation**: Computes gradients `dW` and `db` for weight updates

### 7. **Prediction Function** (Lines 101-107)
- Makes binary predictions using a 0.5 probability threshold
- Returns 1 if probability > 0.5 (fraud), 0 otherwise (licit)

### 8. **Training Function** (Lines 110-144)
- Implements gradient descent optimization
- Updates weights and bias iteratively: `W = W - α·dW`, `b = b - α·db`
- Tracks cost, training accuracy, and validation accuracy at each iteration
- Default hyperparameters:
  - Learning rate: 0.1
  - Iterations: 500

### 9. **Model Training** (Lines 147-155)
- Trains the logistic regression model on the training set
- Uses test set for validation during training

### 10. **Model Evaluation** (Lines 158-169)
- Generates predictions on the test set
- Computes evaluation metrics:
  - **Accuracy**: Overall correctness
  - **F1 Score**: Harmonic mean of precision and recall
  - **Classification Report**: Detailed precision, recall, and F1 per class

### 11. **Visualization Functions** (Lines 172-291)
The code generates 6 comprehensive visualizations:

#### `plot_training_curves()`
- Training cost (log loss) vs. iterations
- Training and validation accuracy vs. iterations

#### `plot_confusion_matrix_emphasized()`
- Confusion matrix with labeled cells:
  - True Negatives (TN): Correctly predicted licit
  - False Positives (FP): Incorrectly predicted as fraud
  - False Negatives (FN): Missed fraud cases
  - True Positives (TP): Correctly detected fraud

#### `plot_prediction_distribution()`
- Bar chart showing the distribution of predicted classes

#### `plot_precision_recall()`
- Precision-Recall curve showing the trade-off between precision and recall

#### `plot_roc_curve()`
- ROC curve with AUC (Area Under Curve) score
- Measures the model's ability to distinguish between classes

#### `plot_class_distribution()`
- Class distribution in training and test sets

### 12. **Generate All Plots** (Lines 294-304)
- Executes all visualization functions
- Saves plots as high-resolution PNG files (300 DPI)

## Output Files

Running the script generates the following visualization files:
- `training_curves.png` - Training cost and accuracy over iterations
- `confusion_matrix_emphasized.png` - Confusion matrix visualization
- `prediction_distribution.png` - Distribution of predictions
- `precision_recall_curve.png` - Precision-Recall curve
- `roc_curve.png` - ROC curve with AUC score
- `class_distribution.png` - Class distribution in train/test sets

## Dependencies

```python
numpy
pandas
scikit-learn
matplotlib
```

## Usage

1. Ensure the following CSV files are in the same directory:
   - `elliptic_txs_features.csv`
   - `elliptic_txs_classes.csv`

2. Run the script:
   ```bash
   python Habshy_LogisticRegressionFraud.py
   ```

3. The script will:
   - Load and preprocess the data
   - Train the logistic regression model
   - Evaluate performance metrics
   - Generate and save all visualizations

## Key Features

- **From Scratch Implementation**: No use of sklearn's LogisticRegression - all core functions (sigmoid, gradient descent, backpropagation) are implemented manually
- **Comprehensive Evaluation**: Multiple metrics and visualizations for thorough model assessment
- **Fraud Detection Focus**: Specifically designed for binary classification of cryptocurrency fraud
- **Educational Value**: Clear code structure with detailed comments explaining each step

## Model Performance

The model's performance can be assessed through:
- **Accuracy**: Overall percentage of correct predictions
- **F1 Score**: Balanced metric considering both precision and recall
- **AUC-ROC**: Area under the ROC curve (higher is better, max = 1.0)
- **Precision-Recall Curve**: Important for imbalanced datasets

## Author

Phlobater Habshy

## Course

CS4824 - Machine Learning Project

