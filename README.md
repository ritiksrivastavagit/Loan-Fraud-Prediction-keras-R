# Loan-Fraud-Prediction-keras-R
An R project to predict loan fraud using a Keras deep learning model. Features engineered from transaction history and balanced with ROSE.


# Loan Fraud Prediction: A Deep Learning Approach

This repository contains the R code for a deep learning model designed to predict loan application fraud. The project's core strategy is to perform advanced feature engineering by combining static loan application data with historical customer transaction data to build a rich behavioral profile for each applicant.

# Project Objective

The goal is to build, train, and tune a Keras neural network that can accurately identify the complex patterns associated with fraudulent loan applications. The model is trained on a synthetically balanced dataset (using ROSE) and evaluated on a real-world, unbalanced test set to prove its effectiveness.

# Tech Stack

Language: R

Core Packages:

keras3: For building and training the neural network.

tidyverse: For all data manipulation and feature engineering.

lubridate: For handling date features.

ROSE: To correct for severe class imbalance using Random Over-Sampling.

caret: For creating the confusion matrix and calculating metrics.

ROCR: For evaluating model performance and threshold tuning.

# Project Workflow

1. Feature Engineering (The Most Critical Step)

A loan application alone provides limited context. To give the model a behavioral understanding of the applicant, we:

Joined loan_applications.csv with transactions.csv by customer_id.

Filtered transactions to include only those that occurred before the application date.

Aggregated this historical data to create new features for each applicant, such as:

txn_count_before_app

txn_avg_amount

txn_failed_count

txn_international_count

txn_distinct_merchants

Joined these new features back to the main loan dataset.

2. Data Preprocessing

One-Hot Encoding: Converted all categorical features (e.g., loan_type, employment_status) into numerical binary columns.

Scaling: Standardized all numerical features (mean of 0, std. dev. of 1), which is essential for a neural network to train properly.

3. Handling Class Imbalance

The raw data is highly imbalanced (e.g., 98% non-fraud vs. 2% fraud).

A stratified 80/20 train/test split was performed.

The ROSE (Random Over-Sampling Examples) package was used only on the training set to create a new, perfectly balanced (50/50) dataset.

The test set was left unbalanced to provide a real-world performance benchmark.

4. Model Architecture

A Keras sequential neural network was defined with the following architecture:

Input Layer

Dense Layer: 64 units, relu activation

Dropout: 30% (to prevent overfitting)

Dense Layer: 32 units, relu activation

Dropout: 20%

Dense Layer: 16 units, relu activation

Output Layer: 1 unit, sigmoid activation (to output a fraud probability between 0 and 1)

# Key Results & Evaluation

The model was trained for 30 epochs and evaluated on the unseen, unbalanced test set.

1. Threshold Tuning

For a fraud model, raw accuracy is misleading. The key is to balance Sensitivity (catching fraud) and Specificity (not flagging legitimate customers). An analysis of the trade-off identified a statistically optimal threshold of 0.025.

2. Final Model Performance

Using the optimal threshold (0.025), the model achieved the following on the real-world test set:

Balanced Accuracy: 76.2% (A random guess would be 50%)

Sensitivity (Recall): 76.2% (The model successfully identified 76.2% of all actual fraud cases).

Specificity: 76.2% (The model successfully identified 76.2% of all legitimate applications).

At a high-confidence threshold (e.g., 0.90), the model could still catch 51.5% of fraud cases while being 99.9% correct about legitimate applications, making it highly precise.

# Data Source & License

Data Files: loan_applications.csv, transactions.csv

License: CC0: Public Domain


# How to Use

Clone this repository.

Place the data files (loan_applications.csv and transactions.csv) in a /data folder (or in the root, just ensure the script's read.csv() path is correct).

Install the required R packages (listed in the "Tech Stack" section).

Run the .Rmd or .R script to perform the analysis and reproduce the results.
