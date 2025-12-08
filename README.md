# Bitcoin Transaction Fraud Detection

## Introduction

This project uses machine learning methods to detect fraud in Bitcoin transactions. Fraud on the blockchain is especially serious because once a transaction is confirmed, it cannot be reversed. As cryptocurrency becomes more widely used, detecting fraudulent activity is becoming more important.

Criminals often hide their actions by taking advantage of the anonymity and freedom that blockchain provides. They use techniques like mixing funds from many sources, moving money across different cryptocurrencies, and breaking large amounts into many smaller transactions to avoid detection. These tactics make it hard to identify fraud using simple rules or traditional approaches.

Recent research shows that new techniques are helping improve fraud detection in cryptocurrency systems. Studies have found that models trained with only a small percentage of labeled data can perform as well as fully supervised models. Machine learning and deep learning methods are making fraud detection more accurate, less dependent on labeled data, and easier to interpret.

## Problem Description

Detecting fraud in blockchain transactions is difficult and becoming more important as cryptocurrencies continue to grow. Unlike traditional banking systems, blockchains give users global access and no central authority. These features are useful, but they also make it easier for criminals to hide illegal activity. Fraudsters use many tricks to stay hidden, such as mixing funds from different sources, moving money across different cryptocurrencies, and breaking large transfers into many smaller ones to avoid detection. These tactics make it hard for simple rules or basic machine learning methods to recognize suspicious behavior.

Another challenge is the size and imbalance of blockchain data. The networks generate massive amounts of transactions, but only a very small number are labeled as fraudulent. This makes it harder to train models that can detect fraud accurately without producing too many false alarms. Because of this, better methods are needed that can understand both the structure of the transaction network and the subtle patterns in user behavior. In this project, we test several machine learning and graph-based models to see which ones work best for identifying fraud in the Elliptic Bitcoin dataset.

## Methodology

We used a diverse set of models to compare traditional machine learning approaches with graph-based methods. Our baseline models included Naive Bayes and Logistic Regression, which rely solely on tabular features. We then incorporated a Random Forest classifier to introduce a more flexible, non-linear model. Finally, we evaluated three different Graph Neural Networks: Graph Attention Network (GAT), GraphSAGE, and Graph Convolutional Network (GCN) to capture relational patterns in the transaction graph and assess how graph structure improves fraud detection.

## Dataset

The Elliptic Dataset maps Bitcoin transactions and classifies them into licit categories (such as exchanges, mining operations, and regular transactions) or illicit categories (such as scams and Ponzi schemes). It contains roughly 200,000 transactions, each represented as a node with 166 numerical features describing attributes like the transaction amount, fee value, wallet age, and the number of inputs and outputs. The dataset also includes about 230,000 edges, representing the flow of money between transactions.

Each transaction is labeled as Fraud (1), Legitimate (2), or Unknown, but the dataset is highly imbalanced with fraudulent activity accounting for less than five percent of all labeled samples. This imbalance poses a significant challenge for machine learning models.

## Preprocessing

For preprocessing, we removed all transactions with unknown labels, keeping only the clearly labeled licit and illicit examples. We also dropped duplicate entries to ensure data consistency and prevent training bias. These cleaning steps allowed us to work with a reliable, well-defined subset of the dataset for model training and evaluation.

## Project Structure

```
MachineLearningProject-CryptoFraudModel-/
├── logistic-regression/    # Logistic Regression baseline model
├── random-forest/          # Random Forest ensemble model
├── naive-bayes/            # Naive Bayes baseline model
├── GAT/                    # Graph Attention Network
├── GCN/                    # Graph Convolutional Network
├── GraphSAGE/              # GraphSAGE model
└── data/                   # Shared dataset files
```

Each model folder contains its own implementation, documentation, and results. The graph neural network models (GAT, GCN, GraphSAGE) leverage the transaction graph structure, while the baseline models (Logistic Regression, Random Forest, Naive Bayes) use only tabular features.

## Results

The project compares the performance of traditional machine learning models with graph-based approaches. Graph Neural Networks typically show improved performance by incorporating the relational structure of transactions, while baseline models provide fast and interpretable alternatives for fraud detection.
