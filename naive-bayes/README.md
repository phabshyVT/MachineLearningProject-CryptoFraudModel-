# Naive Bayes Model

## Overview

This folder contains a Gaussian Naive Bayes classifier for detecting fraudulent Bitcoin transactions. Naive Bayes serves as a simple probabilistic baseline model that assumes feature independence.

## Files

- `naive_bayes_model.py` - Main implementation script
- `data/` - Dataset files (elliptic transaction features, classes, and edge list)

## Methodology

The model uses Gaussian Naive Bayes from scikit-learn, which assumes that features follow a Gaussian distribution and are conditionally independent given the class label.

The implementation includes:

1. **Data Preprocessing**:
   - Loading transaction features and class labels
   - Filtering to keep only labeled examples ('1' for illicit, '2' for licit)
   - Feature standardization using StandardScaler
   - Train-validation-test split with stratification

2. **Model Training**:
   - Gaussian Naive Bayes classifier
   - Training on standardized features

3. **Evaluation**:
   - Classification report with precision, recall, and F1-score
   - Confusion matrix visualization
   - ROC curve and AUC score
   - Performance metrics on validation and test sets

## Usage

Run the main script:
```bash
python naive_bayes_model.py
```

Ensure the data files are in the `data/` folder before running.

## Results

Naive Bayes provides a fast and simple baseline for comparison. While it makes strong independence assumptions that may not hold for transaction features, it offers a computationally efficient approach to fraud detection and helps establish baseline performance metrics.

