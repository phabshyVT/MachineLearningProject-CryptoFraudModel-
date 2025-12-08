# Logistic Regression Model

## Overview

This folder contains a Logistic Regression classifier for detecting fraudulent Bitcoin transactions in the Elliptic dataset. Logistic Regression serves as a baseline model that uses only the tabular features of transactions to classify them as legitimate or fraudulent.

## Files

- `Habshy_LogisticRegressionFraud.py` - Main implementation script
- `README_LogisticRegression.md` - Detailed documentation
- `data/` - Dataset files (elliptic transaction features, classes, and edge list)
- `images/` - Visualization outputs including:
  - Class distribution plots
  - Confusion matrix
  - ROC curve
  - Precision-Recall curve
  - Prediction distribution
  - Training curves

## Methodology

The model uses a standard Logistic Regression classifier from scikit-learn. The implementation includes:

1. **Data Preprocessing**: 
   - Loading and merging transaction features with class labels
   - Filtering to keep only labeled examples (removing 'unknown' labels)
   - Feature normalization using StandardScaler
   - Train-test split with stratification

2. **Model Training**:
   - Standard Logistic Regression with default parameters
   - Training on normalized feature vectors

3. **Evaluation**:
   - Accuracy, F1-score, and classification report
   - Confusion matrix visualization
   - ROC curve and AUC score
   - Precision-Recall curve

## Usage

Run the main script:
```bash
python Habshy_LogisticRegressionFraud.py
```

Make sure the data files are in the `data/` folder before running.

## Results

The model generates comprehensive evaluation metrics and visualizations saved in the `images/` folder. Results demonstrate baseline performance for fraud detection using only tabular features without considering the graph structure of transactions.

