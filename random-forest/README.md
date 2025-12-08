# Random Forest Model

## Overview

This folder contains Random Forest classifiers for detecting fraudulent Bitcoin transactions. Random Forest provides a more flexible, non-linear approach compared to Logistic Regression by using an ensemble of decision trees.

## Files

- `Random_Forest.ipynb` - Main Random Forest implementation with standard train-test split
- `Random_Forest_Temporal_Split.ipynb` - Random Forest with temporal splitting to respect time ordering
- `data/` - Dataset files (elliptic transaction features, classes, and edge list)

## Methodology

The Random Forest models use an ensemble of decision trees to capture non-linear patterns in the transaction features. Two implementations are provided:

1. **Standard Random Forest** (`Random_Forest.ipynb`):
   - Random train-test split with stratification
   - 225 estimators
   - Evaluates performance on randomly split data

2. **Temporal Split Random Forest** (`Random_Forest_Temporal_Split.ipynb`):
   - Time-aware splitting based on transaction timesteps
   - Trains on earlier transactions and tests on later ones
   - More realistic evaluation scenario that respects temporal ordering

## Key Features

- Handles class imbalance through ensemble voting
- Captures non-linear relationships between features
- Provides feature importance scores
- Includes comprehensive evaluation metrics and visualizations

## Usage

Open and run the Jupyter notebooks:
- For standard evaluation: `Random_Forest.ipynb`
- For temporal evaluation: `Random_Forest_Temporal_Split.ipynb`

Update the file paths in the notebooks to point to your data directory.

## Results

The Random Forest models typically achieve better performance than linear models like Logistic Regression by capturing complex feature interactions. The temporal split version provides a more realistic assessment of model performance on future transactions.

