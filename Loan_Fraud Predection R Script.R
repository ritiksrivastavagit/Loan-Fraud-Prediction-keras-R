## ----------------------------------------------------------------
## FRAUD DETECTION MODEL
##
## Purpose: Train a neural network to detect loan application fraud,
## using applicant data and historical transaction behavior.
## ----------------------------------------------------------------

#Installing required packages
#install.packages("caret")
#install.packages("DMwR")
#install.packages("pROC")
#install.packages("e1071")
#install.packages("ROSE")
#install.packages("keras3")
#install_keras(method = "conda") 
## --- 1. LOAD LIBRARIES ---
library(tidyverse)
library(lubridate)
library(caret)
library(ROSE)
library(ROCR)
library(keras3)
library(magrittr)


## --- 2. LOAD DATA ---
loan <- read.csv("loan_applications.csv", stringsAsFactors = FALSE)
txn <- read.csv("transactions.csv", stringsAsFactors = FALSE)


## --- 3. FEATURE ENGINEERING ---

# Convert date strings to date objects
loan <- loan %>%
  mutate(application_date = as_date(application_date))

txn <- txn %>%
  mutate(transaction_date = as_date(transaction_date))

# Join loan keys with all customer transactions
loan_keys <- loan %>%
  select(application_id, customer_id, application_date)

joined_data <- left_join(loan_keys, txn,
                         by = "customer_id",
                         relationship = "many-to-many"
) # Allow many-to-many

# Filter for transactions *before* the application date
historical_txns <- joined_data %>%
  filter(transaction_date < application_date)

# Aggregate historical transaction data by application
txn_features <- historical_txns %>%
  group_by(application_id) %>%
  summarize(
    txn_count_before_app = n(),
    txn_avg_amount = mean(transaction_amount, na.rm = TRUE),
    txn_total_amount = sum(transaction_amount, na.rm = TRUE),
    txn_failed_count = sum(transaction_status != "Success", na.rm = TRUE),
    txn_international_count = sum(is_international_transaction == 1, na.rm = TRUE),
    txn_avg_balance_after = mean(account_balance_after_transaction, na.rm = TRUE),
    txn_distinct_merchants = n_distinct(merchant_name, na.rm = TRUE),
    txn_distinct_devices = n_distinct(device_used, na.rm = TRUE)
  )

# Join new features to the main loan dataset
loan_rich <- left_join(loan, txn_features, by = "application_id")

# Impute 0 for applications with no prior transaction history
loan_rich <- loan_rich %>%
  mutate(
    txn_count_before_app = replace_na(txn_count_before_app, 0),
    txn_failed_count = replace_na(txn_failed_count, 0),
    txn_international_count = replace_na(txn_international_count, 0),
    txn_distinct_merchants = replace_na(txn_distinct_merchants, 0),
    txn_distinct_devices = replace_na(txn_distinct_devices, 0),
    txn_avg_amount = replace_na(txn_avg_amount, 0),
    txn_total_amount = replace_na(txn_total_amount, 0),
    txn_avg_balance_after = replace_na(txn_avg_balance_after, 0)
  )


## --- 4. PREPROCESSING ---

# Create the target variable
loan_rich <- loan_rich %>%
  mutate(
    fraud_flag = ifelse(grepl("Fraudulent", loan_status, ignore.case = TRUE), 1, 0)
  )
# Remove identifiers and non-predictive columns
loan_processed <- loan_rich %>%
  select(
    -application_id, -customer_id, -application_date,
    -residential_address, -loan_status, -fraud_type
  )

# One-hot encode categorical features
loan_dummies <- model.matrix(~ gender + loan_type + employment_status + property_ownership_status + purpose_of_loan - 1, data = loan_processed)
loan_processed <- cbind(loan_processed, loan_dummies)
loan_processed <- loan_processed %>%
  select(-gender, -loan_type, -employment_status, -property_ownership_status, -purpose_of_loan)

# Clean column names for R formula compatibility
names(loan_processed) <- make.names(names(loan_processed))

# Define and scale numerical features
cols_to_scale <- c(
  "loan_amount_requested", "monthly_income", "loan_tenure_months",
  "interest_rate_offered", "cibil_score", "existing_emis_monthly",
  "applicant_age", "number_of_dependents", "debt_to_income_ratio",
  "txn_count_before_app", "txn_avg_amount", "txn_total_amount",
  "txn_failed_count", "txn_international_count", "txn_avg_balance_after",
  "txn_distinct_merchants", "txn_distinct_devices"
)

cols_to_scale <- cols_to_scale[cols_to_scale %in% names(loan_processed)]
loan_processed[cols_to_scale] <- scale(loan_processed[cols_to_scale])


## --- 5. DATA SPLITTING & BALANCING ---
set.seed(823)

fraud_data <- loan_processed %>% filter(fraud_flag == 1)
nonfraud_data <- loan_processed %>% filter(fraud_flag == 0)

# 80% of each class
train_fraud_index <- sample(1:nrow(fraud_data), 0.8 * nrow(fraud_data))
train_nonfraud_index <- sample(1:nrow(nonfraud_data), 0.8 * nrow(nonfraud_data))

loantrain_rich <- rbind(
  fraud_data[train_fraud_index, ],
  nonfraud_data[train_nonfraud_index, ]
)

loantest_rich <- rbind(
  fraud_data[-train_fraud_index, ],
  nonfraud_data[-train_nonfraud_index, ]
)
loantrain_rich_balanced <- ovun.sample(
  fraud_flag ~ .,
  data = loantrain_rich,
  method = "both",
  p = 0.5,
  seed = 123
)$data



## --- 6. KERAS MODEL BUILDING ---

# Prepare data matrices for Keras
x_train <- loantrain_rich_balanced %>%
  select(-fraud_flag) %>%
  as.matrix()
y_train <- loantrain_rich_balanced$fraud_flag %>%
  as.matrix()

x_test <- loantest_rich %>%
  select(-fraud_flag) %>%
  as.matrix()
y_test <- loantest_rich$fraud_flag %>%
  as.matrix()

# Define neural network architecture
model <- keras_model_sequential() %>%
  layer_dense(
    units = 64,
    activation = "relu",
    input_shape = ncol(x_train)
  ) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(
    units = 32,
    activation = "relu"
  ) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(
    units = 16,
    activation = "relu"
  ) %>%
  layer_dense(
    units = 1,
    activation = "sigmoid"
  )

summary(model)

# Compile the Model
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)

# Train the Model
history <- model %>% fit(
  x_train, y_train,
  epochs = 30,
  batch_size = 64,
  validation_split = 0.2,
  verbose = 2
)

# Plot training & validation loss over epochs
plot(history)


## --- 7. MODEL EVALUATION & THRESHOLD ANALYSIS ---

# Get model predictions (probabilities) on the test set
keras_probabilities <- model %>% predict(x_test)

loantest_factor <- factor(y_test, levels = c(0, 1))

# --- 7a. Plot Sensitivity vs. Specificity Trade-off ---

pred_ROCR <- prediction(keras_probabilities, loantest_factor)
perf_sens_spec <- performance(pred_ROCR, "tpr", "tnr")

plot(perf_sens_spec,
     main = "Sensitivity vs. Specificity Trade-off",
     xlab = "Specificity (True Negative Rate)",
     ylab = "Sensitivity (True Positive Rate)",
     colorize = TRUE,
     print.cutoffs.at = seq(0.1, 1.0, by = 0.1),
     lwd = 2
)
abline(a = 1, b = -1, lty = 2, col = "gray") # Add diagonal reference line

# --- 7b. Generate Threshold Performance Dataframe ---

# Define the thresholds to test
thresholds_to_test <- seq(0.8, 1.0, by = 0.01)

# Create an empty list to store our results
results_list <- list()

# Loop over each threshold and calculate metrics
for (thresh in thresholds_to_test) {
  
  # Classify predictions based on the current threshold
  keras_classes <- ifelse(keras_probabilities > thresh, 1, 0)
  keras_classes_factor <- factor(keras_classes, levels = c(0, 1))
  
  # Calculate the Confusion Matrix
  cm <- confusionMatrix(keras_classes_factor, loantest_factor, positive = "1")
  
  # Extract the metrics
  results_list[[as.character(thresh)]] <- data.frame(
    Threshold = thresh,
    Accuracy = cm$overall['Accuracy'],
    Sensitivity = cm$byClass['Sensitivity'],
    Specificity = cm$byClass['Specificity']
  )
}

# Combine all results from the list into one data frame
results_df <- do.call(rbind, results_list)

# Print the final results
print("Threshold Analysis Results:")
print(results_df, row.names = FALSE)

## --- END OF SCRIPT ---


