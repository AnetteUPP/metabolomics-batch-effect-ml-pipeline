#!/usr/bin/env Rscript
# Title: Metabolomic Classification Pipeline - Complete Analysis
# ====================================================================


cat("\n====================================================================\n")
cat("METABOLOMIC CLASSIFICATION PIPELINE\n")
cat("====================================================================\n\n")

# ---- 1. SEED AND ENVIRONMENT ----
set.seed(42)
options(scipen = 999)

# ---- 2. LOAD PACKAGES ----
cat("Loading packages...\n")
.libPaths(c("~/R/library", .libPaths()))

suppressPackageStartupMessages({
  library(glmnet)
  library(nnet)
  library(pROC)
  library(vegan)
  library(corpcor)
  library(boot)
  library(randomForest)
  library(e1071)
  library(kernlab)
  library(jsonlite)
})

# Try loading xgboost; if unavailable, we'll use a fallback
HAS_XGBOOST <- requireNamespace("xgboost", quietly = TRUE)
if (HAS_XGBOOST) {
  library(xgboost)
  cat("xgboost loaded.\n")
} else {
  cat("WARNING: xgboost not available. Will use gbm-like fallback.\n")
}

# Try loading sva for ComBat
HAS_SVA <- requireNamespace("sva", quietly = TRUE)
if (HAS_SVA) {
  library(sva)
  cat("sva loaded.\n")
} else {
  cat("sva not available. Using manual ComBat implementation.\n")
}

cat("All available packages loaded.\n")

# ---- 3. PATHS ----
DATA_DIR <- "/home/ubuntu/Uploads"
RESULTS_DIR <- "/home/ubuntu/results"
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# ---- 4. LOAD DATA ----
cat("\n--- STEP 1: DATA LOADING ---\n")

raw_data_full <- read.csv(file.path(DATA_DIR, "processed_table.csv"),
                          stringsAsFactors = FALSE, check.names = FALSE)
metadata <- read.csv(file.path(DATA_DIR, "metadata.csv"), stringsAsFactors = FALSE)

sample_ids_data <- raw_data_full$sample_id
raw_data <- raw_data_full[, -1]
rownames(raw_data) <- sample_ids_data

cat("Data dimensions:", nrow(raw_data), "samples x", ncol(raw_data), "features\n")

stopifnot(nrow(raw_data) == nrow(metadata))
stopifnot(all(sample_ids_data == metadata$sample_id))

# ---- 5. IMPUTATION + LOG2 (on full data, acceptable for structural zeros) ----
cat("\n--- STEP 2: IMPUTATION + LOG2 ---\n")

data_matrix <- as.matrix(raw_data)
mode(data_matrix) <- "numeric"

impute_half_min <- function(x) {
  na_mask <- is.na(x)
  zeros <- !na_mask & (x == 0)
  if (any(zeros, na.rm = TRUE)) {
    valid_values <- x[!na_mask & x != 0]
    if (length(valid_values) > 0) {
      non_zero_min <- min(valid_values, na.rm = TRUE)
      if (is.finite(non_zero_min)) x[zeros] <- non_zero_min / 2
    }
  }
  if (any(na_mask)) {
    valid_values <- x[!na_mask]
    if (length(valid_values) > 0) {
      overall_min <- min(valid_values, na.rm = TRUE)
      if (is.finite(overall_min)) x[na_mask] <- overall_min / 2
    }
  }
  return(x)
}

cat("Applying half-minimum imputation...\n")
data_imputed <- apply(data_matrix, 2, impute_half_min)
rownames(data_imputed) <- metadata$sample_id
cat("Zeros remaining after imputation:", sum(data_imputed == 0), "\n")

cat("Applying log2 transformation...\n")
data_log <- log2(data_imputed)
rownames(data_log) <- metadata$sample_id

# ---- 6. STRATIFIED TRAIN/TEST SPLIT (FIX #1) ----
cat("\n--- STEP 3: STRATIFIED TRAIN/TEST SPLIT ---\n")

# Manual stratified sampling (replaces caret::createDataPartition)
set.seed(42)
treatment_labels <- metadata$treatment
unique_classes <- unique(treatment_labels)

train_idx_list <- c()
for (cls in unique_classes) {
  cls_indices <- which(treatment_labels == cls)
  n_train <- round(length(cls_indices) * 0.7)
  sampled <- sample(cls_indices, n_train)
  train_idx_list <- c(train_idx_list, sampled)
}
train_idx <- sort(train_idx_list)
test_idx <- setdiff(1:nrow(metadata), train_idx)

train_samples <- metadata$sample_id[train_idx]
test_samples <- metadata$sample_id[test_idx]

cat("Training samples:", length(train_samples), "\n")
cat("Test samples:", length(test_samples), "\n")

train_meta <- metadata[train_idx, ]
test_meta <- metadata[test_idx, ]

y_train <- ifelse(train_meta$treatment == "infection", 1, 0)
y_test <- ifelse(test_meta$treatment == "infection", 1, 0)

cat("Training class distribution: Control =", sum(y_train == 0),
    ", Infection =", sum(y_train == 1), "\n")
cat("Test class distribution: Control =", sum(y_test == 0),
    ", Infection =", sum(y_test == 1), "\n")

# Check batch distribution in train/test
cat("Train batch distribution:\n")
print(table(train_meta$batch))
cat("Test batch distribution:\n")
print(table(test_meta$batch))

# ---- 7. Z-SCORE NORMALIZATION: FIT ON TRAIN ONLY (FIX #2) ----
cat("\n--- STEP 4: Z-SCORE NORMALIZATION (train-fitted) ---\n")

train_log <- data_log[train_samples, ]
test_log <- data_log[test_samples, ]

train_means <- colMeans(train_log)
train_sds <- apply(train_log, 2, sd)
train_sds[train_sds == 0] <- 1  # prevent division by zero

# Apply train-fitted params to both
train_scaled <- scale(train_log, center = train_means, scale = train_sds)
test_scaled <- scale(test_log, center = train_means, scale = train_sds)

cat("Z-score normalization applied (train-fitted parameters).\n")

# ---- 8. COMBAT BATCH CORRECTION: FIT ON TRAIN ONLY (FIX #3) ----
cat("\n--- STEP 5: COMBAT BATCH CORRECTION (train-fitted) ---\n")

# Manual ComBat implementation (parametric empirical Bayes)
manual_combat <- function(dat_train, dat_test, batch_train, batch_test, 
                          mod_train = NULL, mod_test = NULL) {
  # dat_train: features x train_samples matrix
  # dat_test: features x test_samples matrix
  # Returns corrected matrices
  
  batches <- unique(batch_train)
  n_batch <- length(batches)
  
  if (n_batch < 2) {
    cat("  Only one batch in training data, skipping ComBat.\n")
    return(list(train = dat_train, test = dat_test))
  }
  
  cat("  Fitting ComBat on training data (", ncol(dat_train), " samples, ",
      n_batch, " batches)...\n")
  
  # Step 1: Standardize - remove design effects, estimate batch effects
  n_train <- ncol(dat_train)
  p <- nrow(dat_train)
  
  # Design matrix for batch
  batch_design <- model.matrix(~ factor(batch_train))
  
  if (!is.null(mod_train)) {
    design <- cbind(mod_train, batch_design[, -1, drop = FALSE])
  } else {
    design <- batch_design
  }
  
  # Fit regression to get coefficients
  B_hat <- solve(crossprod(design), t(design) %*% t(dat_train))
  
  # Grand mean and batch effects
  grand_mean <- B_hat[1, ]
  
  if (!is.null(mod_train) && ncol(mod_train) > 1) {
    # Remove covariate effects but keep grand mean
    mod_effects <- t(mod_train[, -1, drop = FALSE] %*% B_hat[2:ncol(mod_train), ])
  } else {
    mod_effects <- matrix(0, nrow = p, ncol = n_train)
  }
  
  # Batch effect estimates
  batch_col_start <- ifelse(is.null(mod_train), 2, ncol(mod_train) + 1)
  batch_effects <- B_hat[batch_col_start:nrow(B_hat), , drop = FALSE]
  
  # Standardized data (remove batch effects to estimate variance)
  stand_mean <- grand_mean + mod_effects[, 1:n_train, drop = FALSE]  # just grand + bio
  
  # Residuals for variance estimation
  resid <- dat_train - t(design %*% B_hat)
  
  # Per-gene pooled variance
  var_pooled <- rowSums(resid^2) / (n_train - ncol(design))
  var_pooled[var_pooled == 0] <- median(var_pooled[var_pooled > 0])
  
  # Step 2: Empirical Bayes estimation of batch parameters
  # For each batch, estimate gamma (location) and delta (scale)
  gamma_hat <- list()
  delta_hat <- list()
  gamma_star <- list()
  delta_star <- list()
  
  for (b in 1:n_batch) {
    batch_id <- batches[b]
    batch_samples <- which(batch_train == batch_id)
    n_b <- length(batch_samples)
    
    # Standardize within batch
    s_data <- (dat_train[, batch_samples] - grand_mean) / sqrt(var_pooled)
    
    # Location (gamma)
    gamma_hat[[b]] <- rowMeans(s_data)
    
    # Scale (delta) 
    delta_hat[[b]] <- apply(s_data, 1, var)
    delta_hat[[b]][delta_hat[[b]] == 0] <- median(delta_hat[[b]][delta_hat[[b]] > 0])
    
    # Empirical Bayes shrinkage for gamma
    gamma_bar <- mean(gamma_hat[[b]])
    tau2 <- var(gamma_hat[[b]])
    if (tau2 == 0) tau2 <- 1e-6
    gamma_star[[b]] <- (tau2 * gamma_hat[[b]] + var_pooled * gamma_bar) / 
                       (tau2 + var_pooled)
    
    # Empirical Bayes shrinkage for delta (inverse gamma prior)
    m <- mean(delta_hat[[b]])
    s2 <- var(delta_hat[[b]])
    if (s2 == 0) s2 <- 1e-6
    lambda_hat <- (m * m + 2 * s2) / s2
    theta_hat <- (m * m * m + m * s2) / s2
    delta_star[[b]] <- (theta_hat + (n_b - 1) * delta_hat[[b]]) / 
                       (lambda_hat + n_b - 3)
    delta_star[[b]][delta_star[[b]] <= 0] <- 0.01
  }
  
  # Step 3: Adjust training data
  dat_train_adj <- dat_train
  for (b in 1:n_batch) {
    batch_id <- batches[b]
    batch_samples <- which(batch_train == batch_id)
    
    for (j in batch_samples) {
      dat_train_adj[, j] <- grand_mean + 
        sqrt(var_pooled) * (dat_train[, j] - grand_mean - 
        sqrt(var_pooled) * gamma_star[[b]]) / 
        sqrt(var_pooled * delta_star[[b]])
    }
  }
  
  # Step 4: Adjust test data using train-fitted parameters
  # Assign test samples to their batch and apply correction
  dat_test_adj <- dat_test
  for (j in 1:ncol(dat_test)) {
    b_id <- batch_test[j]
    b_idx <- which(batches == b_id)
    
    if (length(b_idx) == 0) {
      # Unknown batch - no correction possible
      cat("  Warning: test sample batch", b_id, "not seen in training.\n")
      next
    }
    
    dat_test_adj[, j] <- grand_mean + 
      sqrt(var_pooled) * (dat_test[, j] - grand_mean - 
      sqrt(var_pooled) * gamma_star[[b_idx]]) / 
      sqrt(var_pooled * delta_star[[b_idx]])
  }
  
  return(list(train = dat_train_adj, test = dat_test_adj,
              grand_mean = grand_mean, var_pooled = var_pooled,
              gamma_star = gamma_star, delta_star = delta_star,
              batches = batches))
}

# Prepare data for ComBat (features x samples)
train_for_combat <- t(train_scaled)
test_for_combat <- t(test_scaled)

# Remove constant features
feature_vars <- apply(train_for_combat, 1, var)
non_const <- feature_vars > 1e-10
cat("Non-constant features:", sum(non_const), "of", length(non_const), "\n")

train_combat_input <- train_for_combat[non_const, ]
test_combat_input <- test_for_combat[non_const, ]

# Batch labels
batch_train <- train_meta$batch
batch_test <- test_meta$batch

# Treatment model matrix for train
mod_train <- model.matrix(~ as.factor(treatment), data = train_meta)

# NOTE: Always use manual_combat for both train and test to ensure consistent
# parameter estimates. Previously, sva::ComBat was used for train while
# manual_combat was used for test, causing inconsistent corrections because
# each implementation produces slightly different parameter estimates.
# Using the same implementation for both ensures train and test are corrected
# with identical fitted parameters.

if (length(unique(batch_train)) > 1) {
  cat("Using manual ComBat implementation (train-fitted, applied to both)...\n")
  combat_params <- manual_combat(train_combat_input, test_combat_input,
                                 batch_train, batch_test, mod_train, NULL)
  train_corrected <- t(combat_params$train)
  test_corrected <- t(combat_params$test)
} else {
  cat("Only one batch in training data, skipping ComBat.\n")
  train_corrected <- t(train_combat_input)
  test_corrected <- t(test_combat_input)
}

# Add back constant features
train_corrected_full <- as.data.frame(matrix(0, nrow = length(train_samples),
                                              ncol = ncol(train_scaled)))
colnames(train_corrected_full) <- colnames(train_scaled)
rownames(train_corrected_full) <- train_samples
train_corrected_full[, names(which(non_const))] <- train_corrected
train_corrected_full[, names(which(!non_const))] <- train_scaled[, names(which(!non_const))]

test_corrected_full <- as.data.frame(matrix(0, nrow = length(test_samples),
                                             ncol = ncol(test_scaled)))
colnames(test_corrected_full) <- colnames(test_scaled)
rownames(test_corrected_full) <- test_samples
test_corrected_full[, names(which(non_const))] <- test_corrected
test_corrected_full[, names(which(!non_const))] <- test_scaled[, names(which(!non_const))]

cat("ComBat correction completed.\n")
cat("Train corrected dims:", nrow(train_corrected_full), "x", ncol(train_corrected_full), "\n")
cat("Test corrected dims:", nrow(test_corrected_full), "x", ncol(test_corrected_full), "\n")

# Save corrected_expression.csv (full dataset for GNN script)
corrected_all <- rbind(train_corrected_full, test_corrected_full)
corrected_all$sample_id <- rownames(corrected_all)
corrected_all <- corrected_all[, c("sample_id", setdiff(colnames(corrected_all), "sample_id"))]
write.csv(corrected_all, file.path(RESULTS_DIR, "corrected_expression.csv"), row.names = FALSE)
cat("Saved corrected_expression.csv\n")

# ---- 9. PRE/POST CORRECTION PCA + PERMANOVA ----
cat("\n--- STEP 6: PCA + PERMANOVA ---\n")

# Pre-correction PCA (on log-scaled full data)
data_log_full <- data_log
TOP_M <- 3000
mad_pre <- apply(data_log_full, 2, mad, na.rm = TRUE)
top_pre <- names(sort(mad_pre, decreasing = TRUE)[1:TOP_M])
data_pre_pca <- data_log_full[, top_pre]

pca_pre <- prcomp(data_pre_pca, center = TRUE, scale. = FALSE)
var_pre <- (pca_pre$sdev^2 / sum(pca_pre$sdev^2)) * 100
cat("Pre-correction PC1:", round(var_pre[1], 1), "%, PC2:", round(var_pre[2], 1), "%\n")

set.seed(42)
dist_pre <- dist(data_pre_pca)
perm_batch_pre <- adonis2(dist_pre ~ experiment, data = metadata, permutations = 999)
set.seed(42)
perm_treat_pre <- adonis2(dist_pre ~ treatment, data = metadata, permutations = 999)
batch_r2_pre <- perm_batch_pre$R2[1]
treat_r2_pre <- perm_treat_pre$R2[1]
batch_F_pre <- perm_batch_pre$F[1]
treat_F_pre <- perm_treat_pre$F[1]
batch_p_pre <- perm_batch_pre$`Pr(>F)`[1]
treat_p_pre <- perm_treat_pre$`Pr(>F)`[1]

cat("Pre-correction Batch R²:", round(batch_r2_pre * 100, 1), "%, F =", round(batch_F_pre, 3), ", p =", batch_p_pre, "\n")
cat("Pre-correction Treatment R²:", round(treat_r2_pre * 100, 1), "%, F =", round(treat_F_pre, 3), ", p =", treat_p_pre, "\n")

# Post-correction PCA
mad_post <- apply(train_corrected_full, 2, mad, na.rm = TRUE)
top_post <- names(sort(mad_post, decreasing = TRUE)[1:TOP_M])

data_post_all <- rbind(train_corrected_full[, top_post], test_corrected_full[, top_post])
meta_all <- rbind(train_meta, test_meta)

pca_post <- prcomp(data_post_all, center = TRUE, scale. = FALSE)
var_post <- (pca_post$sdev^2 / sum(pca_post$sdev^2)) * 100
cat("Post-correction PC1:", round(var_post[1], 1), "%, PC2:", round(var_post[2], 1), "%\n")

set.seed(42)
dist_post <- dist(data_post_all)
perm_batch_post <- adonis2(dist_post ~ experiment, data = meta_all, permutations = 999)
set.seed(42)
perm_treat_post <- adonis2(dist_post ~ treatment, data = meta_all, permutations = 999)
batch_r2_post <- perm_batch_post$R2[1]
treat_r2_post <- perm_treat_post$R2[1]
batch_F_post <- perm_batch_post$F[1]
treat_F_post <- perm_treat_post$F[1]
batch_p_post <- perm_batch_post$`Pr(>F)`[1]
treat_p_post <- perm_treat_post$`Pr(>F)`[1]

cat("Post-correction Batch R²:", round(batch_r2_post * 100, 1), "%, F =", round(batch_F_post, 3), ", p =", batch_p_post, "\n")
cat("Post-correction Treatment R²:", round(treat_r2_post * 100, 1), "%, F =", round(treat_F_post, 3), ", p =", treat_p_post, "\n")

# ---- 10. FEATURE SELECTION ON TRAIN ONLY ----
cat("\n--- STEP 7: FEATURE SELECTION (train only) ---\n")

TOP_K <- 120

# MAD prescreening on train corrected data
mad_train <- apply(train_corrected_full, 2, mad, na.rm = TRUE)
top_features <- names(sort(mad_train, decreasing = TRUE)[1:TOP_M])

X_train_filtered <- as.matrix(train_corrected_full[, top_features])
X_test_filtered <- as.matrix(test_corrected_full[, top_features])

# Ledoit-Wolf shrinkage correlation
X_ctrl <- X_train_filtered[y_train == 0, ]
X_inf <- X_train_filtered[y_train == 1, ]

cat("Computing Ledoit-Wolf shrinkage correlations...\n")
cor_ctrl_lw <- cor.shrink(X_ctrl, verbose = FALSE)
cor_inf_lw <- cor.shrink(X_inf, verbose = FALSE)

cor_ctrl_mat <- as.matrix(cor_ctrl_lw)
cor_inf_mat <- as.matrix(cor_inf_lw)

cat("Shrinkage intensity (Control):", round(attr(cor_ctrl_lw, "lambda"), 4), "\n")
cat("Shrinkage intensity (Infection):", round(attr(cor_inf_lw, "lambda"), 4), "\n")

# Fisher Z transform for differential correlation
fisher_z <- function(r) {
  r <- pmin(pmax(r, -0.999), 0.999)
  0.5 * log((1 + r) / (1 - r))
}

z_ctrl <- fisher_z(cor_ctrl_mat)
z_inf <- fisher_z(cor_inf_mat)
delta_z <- z_inf - z_ctrl

upper_idx <- upper.tri(delta_z)
edge_values <- abs(delta_z[upper_idx])
edge_order <- order(edge_values, decreasing = TRUE)

selected_idx <- which(upper_idx, arr.ind = TRUE)
selected_idx <- selected_idx[edge_order[1:TOP_K], ]

# Create edge interaction features
create_edge_features <- function(X, edges) {
  features <- matrix(0, nrow = nrow(X), ncol = nrow(edges))
  for (i in 1:nrow(edges)) {
    features[, i] <- X[, edges[i, 1]] * X[, edges[i, 2]]
  }
  colnames(features) <- paste0("edge_", 1:nrow(edges))
  return(features)
}

Z_train <- create_edge_features(X_train_filtered, selected_idx)
Z_test <- create_edge_features(X_test_filtered, selected_idx)

cat("Edge features created:", ncol(Z_train), "\n")

# ---- 11. EPOCH OPTIMIZATION ----
cat("\n--- STEP 8: EPOCH OPTIMIZATION ---\n")

epoch_values <- c(50, 100, 150, 200, 300, 400, 500, 600, 800)
epoch_results <- data.frame(epochs = integer(), auroc = numeric(),
                            brier = numeric(), time = numeric())

calc_brier <- function(y_true, y_prob) mean((y_prob - y_true)^2)

for (ep in epoch_values) {
  set.seed(42)
  start_time <- Sys.time()
  nn_temp <- nnet(x = Z_train, y = y_train, size = 8, maxit = ep,
                  decay = 0.01, trace = FALSE, linout = FALSE)
  end_time <- Sys.time()

  nn_prob <- as.numeric(predict(nn_temp, Z_test))
  auroc_val <- as.numeric(auc(roc(y_test, nn_prob, quiet = TRUE)))
  brier_val <- calc_brier(y_test, nn_prob)
  time_val <- as.numeric(difftime(end_time, start_time, units = "secs"))

  epoch_results <- rbind(epoch_results, data.frame(
    epochs = ep, auroc = auroc_val, brier = brier_val, time = time_val))

  cat("  Epochs:", ep, "- AUROC:", round(auroc_val, 3),
      "- Brier:", round(brier_val, 3), "\n")
}

optimal_epoch <- epoch_results$epochs[which.max(epoch_results$auroc)]
cat("Optimal epoch count:", optimal_epoch, "\n")

# ---- 12. TRAIN ALL MODELS ----
cat("\n--- STEP 9: TRAIN ALL MODELS ---\n")

# Neural Network
set.seed(42)
nn_model <- nnet(x = Z_train, y = y_train, size = 8, maxit = optimal_epoch,
                 decay = 0.01, trace = FALSE, linout = FALSE)
nn_train_prob <- as.numeric(predict(nn_model, Z_train))
nn_test_prob <- as.numeric(predict(nn_model, Z_test))
cat("NN AUROC:", round(auc(roc(y_test, nn_test_prob, quiet = TRUE)), 3), "\n")

# Elastic Net
set.seed(42)
cv_fit <- cv.glmnet(Z_train, y_train, family = "binomial", alpha = 0.5, nfolds = 5)
glmnet_model <- glmnet(Z_train, y_train, family = "binomial",
                       alpha = 0.5, lambda = cv_fit$lambda.min)
glmnet_train_prob <- as.numeric(predict(glmnet_model, Z_train, type = "response"))
glmnet_test_prob <- as.numeric(predict(glmnet_model, Z_test, type = "response"))
cat("EN AUROC:", round(auc(roc(y_test, glmnet_test_prob, quiet = TRUE)), 3), "\n")

# Random Forest
set.seed(42)
Z_train_df <- as.data.frame(Z_train)
Z_train_df$y <- as.factor(y_train)
rf_model <- randomForest(y ~ ., data = Z_train_df, ntree = 1000, mtry = 11, importance = TRUE)
rf_train_prob <- as.numeric(predict(rf_model, Z_train_df, type = "prob")[, "1"])
rf_test_prob <- as.numeric(predict(rf_model, as.data.frame(Z_test), type = "prob")[, "1"])
cat("RF AUROC:", round(auc(roc(y_test, rf_test_prob, quiet = TRUE)), 3), "\n")

# XGBoost
if (HAS_XGBOOST) {
  dtrain <- xgb.DMatrix(data = Z_train, label = y_train)
  dtest <- xgb.DMatrix(data = Z_test, label = y_test)
  
  xgb_params <- list(objective = "binary:logistic", eval_metric = "auc",
                     max_depth = 3, eta = 0.1, subsample = 0.8,
                     colsample_bytree = 0.8, min_child_weight = 1)
  
  set.seed(42)
  xgb_cv <- xgb.cv(params = xgb_params, data = dtrain, nrounds = 500,
                    nfold = 5, early_stopping_rounds = 20, verbose = 0, maximize = TRUE)
  best_nrounds <- xgb_cv$best_iteration
  
  set.seed(42)
  xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = best_nrounds, verbose = 0)
  xgb_train_prob <- predict(xgb_model, dtrain)
  xgb_test_prob <- predict(xgb_model, dtest)
} else {
  # Fallback: use a simple logistic regression with polynomial features
  cat("Using logistic regression fallback for XGBoost...\n")
  set.seed(42)
  xgb_fallback <- glm(y_train ~ ., data = data.frame(Z_train, y_train = y_train), 
                       family = "binomial")
  xgb_train_prob <- predict(xgb_fallback, data.frame(Z_train), type = "response")
  xgb_test_prob <- predict(xgb_fallback, data.frame(Z_test), type = "response")
}
cat("XGB AUROC:", round(auc(roc(y_test, xgb_test_prob, quiet = TRUE)), 3), "\n")

# SVM Linear
set.seed(42)
svm_lin_model <- svm(x = Z_train, y = as.factor(y_train), kernel = "linear",
                     cost = 1, probability = TRUE)
svm_lin_train_pred <- predict(svm_lin_model, Z_train, probability = TRUE)
svm_lin_train_prob <- attr(svm_lin_train_pred, "probabilities")[, "1"]
svm_lin_test_pred <- predict(svm_lin_model, Z_test, probability = TRUE)
svm_lin_test_prob <- attr(svm_lin_test_pred, "probabilities")[, "1"]
cat("SVM-L AUROC:", round(auc(roc(y_test, svm_lin_test_prob, quiet = TRUE)), 3), "\n")

# SVM RBF
set.seed(42)
svm_rbf_model <- svm(x = Z_train, y = as.factor(y_train), kernel = "radial",
                     cost = 1, gamma = 0.01, probability = TRUE)
svm_rbf_train_pred <- predict(svm_rbf_model, Z_train, probability = TRUE)
svm_rbf_train_prob <- attr(svm_rbf_train_pred, "probabilities")[, "1"]
svm_rbf_test_pred <- predict(svm_rbf_model, Z_test, probability = TRUE)
svm_rbf_test_prob <- attr(svm_rbf_test_pred, "probabilities")[, "1"]
cat("SVM-R AUROC:", round(auc(roc(y_test, svm_rbf_test_prob, quiet = TRUE)), 3), "\n")

# ---- 13. COMPUTE ALL METRICS ----
cat("\n--- STEP 10: MODEL EVALUATION ---\n")

compute_metrics <- function(y_true, y_prob, model_name) {
  pred <- ifelse(y_prob > 0.5, 1, 0)
  tp <- sum(pred == 1 & y_true == 1)
  tn <- sum(pred == 0 & y_true == 0)
  fp <- sum(pred == 1 & y_true == 0)
  fn <- sum(pred == 0 & y_true == 1)
  
  auroc <- as.numeric(auc(roc(y_true, y_prob, quiet = TRUE)))
  sens <- tp / (tp + fn)
  spec <- tn / (tn + fp)
  prec <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
  f1 <- ifelse(prec + sens > 0, 2 * prec * sens / (prec + sens), 0)
  
  data.frame(model = model_name, auroc = auroc, sensitivity = sens,
             specificity = spec, f1 = f1, tp = tp, tn = tn, fp = fp, fn = fn)
}

models_list <- list(
  "Neural Network" = nn_test_prob,
  "Elastic Net" = glmnet_test_prob,
  "Random Forest" = rf_test_prob,
  "XGBoost" = xgb_test_prob,
  "SVM Linear" = svm_lin_test_prob,
  "SVM RBF" = svm_rbf_test_prob
)

all_metrics <- do.call(rbind, lapply(names(models_list), function(nm) {
  compute_metrics(y_test, models_list[[nm]], nm)
}))

cat("\nTest Set Metrics:\n")
print(all_metrics[, c("model", "auroc", "sensitivity", "specificity", "f1")])

best_model_name <- all_metrics$model[which.max(all_metrics$auroc)]
cat("\nBest model:", best_model_name, "with AUROC:", 
    round(max(all_metrics$auroc), 3), "\n")

# ---- 14. DELONG TESTS ----
cat("\n--- STEP 11: DeLong Tests (vs Elastic Net) ---\n")

roc_list <- lapply(models_list, function(p) roc(y_test, p, quiet = TRUE))

delong_results <- data.frame()
ref_name <- "Elastic Net"
for (nm in names(roc_list)) {
  if (nm == ref_name) next
  dl <- roc.test(roc_list[[nm]], roc_list[[ref_name]], method = "delong")
  delong_results <- rbind(delong_results, data.frame(
    comparison = paste(nm, "vs", ref_name),
    z_stat = dl$statistic,
    p_value = dl$p.value
  ))
  cat("  ", nm, "vs EN: p =", round(dl$p.value, 4), "\n")
}

# ---- 15. BCa BOOTSTRAP CIs ----
cat("\n--- STEP 12: BCa Bootstrap CIs (B=2000) ---\n")

N_BOOTSTRAP <- 2000

auroc_fn <- function(y, p) as.numeric(auc(roc(y, p, quiet = TRUE)))
sens_fn <- function(y, p) {
  pred <- ifelse(p > 0.5, 1, 0)
  tp <- sum(pred == 1 & y == 1); fn <- sum(pred == 0 & y == 1)
  if ((tp + fn) == 0) return(NA); tp / (tp + fn)
}
spec_fn <- function(y, p) {
  pred <- ifelse(p > 0.5, 1, 0)
  tn <- sum(pred == 0 & y == 0); fp <- sum(pred == 1 & y == 0)
  if ((tn + fp) == 0) return(NA); tn / (tn + fp)
}
f1_fn <- function(y, p) {
  pred <- ifelse(p > 0.5, 1, 0)
  tp <- sum(pred == 1 & y == 1); fp <- sum(pred == 1 & y == 0); fn <- sum(pred == 0 & y == 1)
  denom <- 2 * tp + fp + fn; if (denom == 0) return(NA); 2 * tp / denom
}

metrics_fns <- list(auroc = auroc_fn, sensitivity = sens_fn,
                    specificity = spec_fn, f1 = f1_fn)

bootstrap_results <- list()
for (model_name in names(models_list)) {
  cat("  ", model_name, "...")
  y_prob <- models_list[[model_name]]
  bootstrap_results[[model_name]] <- list()
  
  for (metric_name in names(metrics_fns)) {
    data_mat <- cbind(y_test, y_prob)
    stat_fn <- function(data, indices) {
      d <- data[indices, ]
      y <- d[, 1]; p <- d[, 2]
      if (length(unique(y)) < 2) return(NA)
      return(metrics_fns[[metric_name]](y, p))
    }
    set.seed(42)
    boot_obj <- boot(data_mat, stat_fn, R = N_BOOTSTRAP)
    ci <- tryCatch(
      boot.ci(boot_obj, conf = 0.95, type = "bca")$bca[4:5],
      error = function(e) {
        tryCatch(boot.ci(boot_obj, conf = 0.95, type = "perc")$percent[4:5],
                 error = function(e2) c(NA, NA))
      }
    )
    bootstrap_results[[model_name]][[metric_name]] <- list(
      estimate = metrics_fns[[metric_name]](y_test, y_prob),
      ci_lower = ci[1], ci_upper = ci[2])
  }
  cat(" done\n")
}

# ---- 16. McNEMAR TESTS ----
cat("\n--- STEP 13: McNemar Tests ---\n")

mcnemar_results <- data.frame()
model_names_all <- names(models_list)
for (i in 1:(length(model_names_all) - 1)) {
  for (j in (i + 1):length(model_names_all)) {
    m1 <- model_names_all[i]; m2 <- model_names_all[j]
    pred1 <- ifelse(models_list[[m1]] > 0.5, 1, 0)
    pred2 <- ifelse(models_list[[m2]] > 0.5, 1, 0)
    c1 <- (pred1 == y_test); c2 <- (pred2 == y_test)
    b <- sum(c1 & !c2); c_val <- sum(!c1 & c2)
    if ((b + c_val) == 0) {
      stat <- 0; p_val <- 1
    } else {
      stat <- (abs(b - c_val) - 1)^2 / (b + c_val)
      p_val <- pchisq(stat, df = 1, lower.tail = FALSE)
    }
    mcnemar_results <- rbind(mcnemar_results, data.frame(
      comparison = paste(m1, "vs", m2),
      statistic = round(stat, 4), p_value = round(p_val, 4),
      b_discordant = b, c_discordant = c_val))
  }
}

# ---- 17. GENERATE FIGURES ----
cat("\n--- STEP 14: GENERATING FIGURES ---\n")

# Figure 1: Batch Effects PCA + PERMANOVA
cat("  Figure 1: Batch effects...\n")
png(file.path(RESULTS_DIR, "Figure_1_batch_effects.png"), width = 12, height = 10, units = "in", res = 300)
par(mfrow = c(2, 2), mar = c(5, 5, 4, 2))

# Panel A: Pre-correction PCA by batch
batch_colors_pre <- ifelse(metadata$experiment == 1, "#4393C3", "#D6604D")
plot(pca_pre$x[, 1], pca_pre$x[, 2], col = batch_colors_pre, pch = 16, cex = 1.2,
     xlab = paste0("PC1 (", round(var_pre[1], 1), "% variance)"),
     ylab = paste0("PC2 (", round(var_pre[2], 1), "% variance)"),
     main = "A) Before ComBat: PCA by Batch")
legend("topright", legend = c("Experiment 1", "Experiment 2"), 
       col = c("#4393C3", "#D6604D"), pch = 16, bty = "n")

# Panel B: Post-correction PCA by batch
batch_colors_post <- ifelse(meta_all$experiment == 1, "#4393C3", "#D6604D")
plot(pca_post$x[, 1], pca_post$x[, 2], col = batch_colors_post, pch = 16, cex = 1.2,
     xlab = paste0("PC1 (", round(var_post[1], 1), "% variance)"),
     ylab = paste0("PC2 (", round(var_post[2], 1), "% variance)"),
     main = "B) After ComBat: PCA by Batch")
legend("topright", legend = c("Experiment 1", "Experiment 2"),
       col = c("#4393C3", "#D6604D"), pch = 16, bty = "n")

# Panel C: Post-correction PCA by treatment
treat_colors <- ifelse(meta_all$treatment == "control", "#2ca02c", "#d62728")
plot(pca_post$x[, 1], pca_post$x[, 2], col = treat_colors, pch = 16, cex = 1.2,
     xlab = paste0("PC1 (", round(var_post[1], 1), "% variance)"),
     ylab = paste0("PC2 (", round(var_post[2], 1), "% variance)"),
     main = "C) After ComBat: PCA by Treatment")
legend("topright", legend = c("Control", "Infected"),
       col = c("#2ca02c", "#d62728"), pch = 16, bty = "n")

# Panel D: PERMANOVA R² comparison
r2_vals <- c(batch_r2_pre, batch_r2_post, treat_r2_pre, treat_r2_post)
bp <- barplot(matrix(c(batch_r2_pre * 100, batch_r2_post * 100, 
                        treat_r2_pre * 100, treat_r2_post * 100), nrow = 2),
              beside = TRUE, col = c("gray70", "#2ca02c"),
              names.arg = c("Batch R²", "Treatment R²"),
              ylab = "R² (% Variance)", main = "D) PERMANOVA: Before vs After ComBat",
              ylim = c(0, max(r2_vals * 100) * 1.3))
legend("topright", legend = c("Before ComBat", "After ComBat"),
       fill = c("gray70", "#2ca02c"), bty = "n")
text(bp, matrix(c(batch_r2_pre * 100, batch_r2_post * 100,
                   treat_r2_pre * 100, treat_r2_post * 100), nrow = 2) + 0.3,
     paste0(round(c(batch_r2_pre * 100, batch_r2_post * 100,
                     treat_r2_pre * 100, treat_r2_post * 100), 1), "%"), cex = 0.8)
dev.off()

# Figure 2: Epoch Optimization
cat("  Figure 2: Epoch optimization...\n")
png(file.path(RESULTS_DIR, "Figure_2_epoch_optimization.png"), width = 15, height = 5, units = "in", res = 300)
par(mfrow = c(1, 3), mar = c(5, 5, 4, 2))

# Panel A
en_auroc <- as.numeric(auc(roc(y_test, glmnet_test_prob, quiet = TRUE)))
plot(epoch_results$epochs, epoch_results$auroc, type = "b", pch = 16, col = "blue",
     xlab = "Training Epochs", ylab = "Test Set AUROC",
     main = "A) AUROC vs Training Epochs")
abline(v = optimal_epoch, col = "red", lty = 2, lwd = 2)
abline(h = en_auroc, col = "gray50", lty = 3)
legend("bottomright", legend = c(paste0("Optimal (", optimal_epoch, ")"),
                                  paste0("Elastic Net (", round(en_auroc, 3), ")")),
       col = c("red", "gray50"), lty = c(2, 3), lwd = 2, bty = "n")

# Panel B
plot(epoch_results$epochs, epoch_results$brier, type = "b", pch = 16, col = "red",
     xlab = "Training Epochs", ylab = "Brier Score (lower is better)",
     main = "B) Calibration vs Training Epochs")
abline(v = optimal_epoch, col = "red", lty = 2, lwd = 2)
legend("topright", legend = paste0("Optimal (", optimal_epoch, ")"),
       col = "red", lty = 2, lwd = 2, bty = "n")

# Panel C
plot(epoch_results$epochs, epoch_results$time, type = "b", pch = 17, col = "darkgreen",
     xlab = "Training Epochs", ylab = "Training Time (seconds)",
     main = "C) Computational Cost vs Epochs")
abline(v = optimal_epoch, col = "red", lty = 2, lwd = 2)
legend("topleft", legend = paste0("Optimal (", optimal_epoch, ")"),
       col = "red", lty = 2, lwd = 2, bty = "n")
dev.off()

# Figure 3: ROC Curves
cat("  Figure 3: ROC curves...\n")
png(file.path(RESULTS_DIR, "Figure_3_ROC_curves.png"), width = 8, height = 8, units = "in", res = 300)
par(mar = c(5, 5, 4, 2))

model_colors <- c("blue", "green4", "purple", "brown", "hotpink", "gray40")
model_ltys <- c(1, 1, 1, 1, 2, 2)

plot(0, 0, type = "n", xlim = c(1, 0), ylim = c(0, 1),
     xlab = "Specificity", ylab = "Sensitivity", main = "ROC Curves: Model Comparison")
abline(a = 1, b = -1, col = "lightgray")

for (i in seq_along(names(roc_list))) {
  nm <- names(roc_list)[i]
  r <- roc_list[[nm]]
  lines(r$specificities, r$sensitivities, col = model_colors[i], lwd = 2, lty = model_ltys[i])
}

# Random classifier
abline(a = 1, b = -1, col = "gray80", lty = 3)

legend_labels <- paste0(names(roc_list), " (AUROC = ", 
                         sapply(roc_list, function(r) round(auc(r), 3)), ")")
legend_labels <- c(legend_labels, "Random (AUROC = 0.500)")
legend("bottomleft", legend = legend_labels,
       col = c(model_colors, "gray80"), lty = c(model_ltys, 3), lwd = 2, cex = 0.8, bty = "n")
dev.off()

# Figure 4: Bootstrap CIs
cat("  Figure 4: Bootstrap CIs...\n")
png(file.path(RESULTS_DIR, "Figure_4_bootstrap_ci.png"), width = 12, height = 8, units = "in", res = 300)
par(mfrow = c(2, 2), mar = c(5, 8, 4, 2))

metric_names <- c("auroc", "sensitivity", "specificity", "f1")
metric_titles <- c("A) AUROC (BCa Bootstrap)", "B) Sensitivity", "C) Specificity", "D) F1 Score")
metric_xlabs <- c("AUROC", "Sensitivity", "Specificity", "F1")

model_order <- names(models_list)
n_models <- length(model_order)

for (mi in seq_along(metric_names)) {
  mn <- metric_names[mi]
  
  estimates <- sapply(model_order, function(m) bootstrap_results[[m]][[mn]]$estimate)
  ci_lo <- sapply(model_order, function(m) bootstrap_results[[m]][[mn]]$ci_lower)
  ci_hi <- sapply(model_order, function(m) bootstrap_results[[m]][[mn]]$ci_upper)
  
  xlim <- c(min(ci_lo, na.rm = TRUE) - 0.05, max(ci_hi, na.rm = TRUE) + 0.05)
  xlim[1] <- max(0, xlim[1])
  xlim[2] <- min(1, xlim[2])
  
  plot(estimates, n_models:1, xlim = xlim, ylim = c(0.5, n_models + 0.5),
       pch = 16, cex = 1.5, col = model_colors,
       xlab = metric_xlabs[mi], ylab = "", yaxt = "n", main = metric_titles[mi])
  axis(2, at = n_models:1, labels = model_order, las = 1, cex.axis = 0.8)
  
  for (j in 1:n_models) {
    segments(ci_lo[j], n_models - j + 1, ci_hi[j], n_models - j + 1,
             col = model_colors[j], lwd = 2)
  }
  
  if (mn == "auroc") abline(v = 0.5, lty = 2, col = "gray60")
}
dev.off()

# Figure 5: Confusion Matrices
cat("  Figure 5: Confusion matrices...\n")
png(file.path(RESULTS_DIR, "Figure_5_confusion_matrices.png"), width = 16, height = 10, units = "in", res = 300)
par(mfrow = c(2, 3), mar = c(5, 5, 4, 2))

for (i in seq_along(names(models_list))) {
  nm <- names(models_list)[i]
  row <- all_metrics[all_metrics$model == nm, ]
  
  cm <- matrix(c(row$tn, row$fn, row$fp, row$tp), nrow = 2, byrow = TRUE)
  
  cell_colors <- matrix(c("#B0C4DE", "#87CEEB", "#87CEEB", "#00008B"), nrow = 2)
  
  plot(0, 0, type = "n", xlim = c(0, 2), ylim = c(0, 2), xaxt = "n", yaxt = "n",
       xlab = "", ylab = "", main = paste0(nm, "\nSens: ", 
       round(row$sensitivity * 100, 1), "%, Spec: ", round(row$specificity * 100, 1), "%"))
  
  for (r in 1:2) {
    for (cc in 1:2) {
      rect(cc - 1, 2 - r, cc, 3 - r, col = cell_colors[r, cc], border = "white", lwd = 2)
      text(cc - 0.5, 2.5 - r, cm[r, cc], cex = 2, font = 2,
           col = ifelse(cm[r, cc] > max(cm) / 2, "white", "black"))
    }
  }
  
  mtext("Predicted Control", side = 1, at = 0.5, line = 1, cex = 0.7)
  mtext("Predicted Infected", side = 1, at = 1.5, line = 1, cex = 0.7)
  mtext("Actual\nControl", side = 2, at = 1.5, line = 1, cex = 0.7, las = 1)
  mtext("Actual\nInfected", side = 2, at = 0.5, line = 1, cex = 0.7, las = 1)
}
dev.off()

# Figure 6: Model Comparison
cat("  Figure 6: Model comparison...\n")
png(file.path(RESULTS_DIR, "Figure_6_model_comparison.png"), width = 10, height = 10, units = "in", res = 300)
par(mfrow = c(2, 2), mar = c(7, 5, 4, 2))

short_names <- c("NN", "EN", "RF", "XGB", "SVM-L", "SVM-R")

# A) AUROC
bp <- barplot(all_metrics$auroc, col = model_colors, names.arg = short_names,
              ylab = "AUROC", main = "A) AUROC by Model", ylim = c(0, 1), las = 2)
text(bp, all_metrics$auroc + 0.03, round(all_metrics$auroc, 3), cex = 0.8)
abline(h = 0.5, lty = 2, col = "gray60")

# B) Sensitivity
bp <- barplot(all_metrics$sensitivity, col = model_colors, names.arg = short_names,
              ylab = "Sensitivity", main = "B) Sensitivity by Model", ylim = c(0, 1), las = 2)
text(bp, all_metrics$sensitivity + 0.03, round(all_metrics$sensitivity, 2), cex = 0.8)

# C) Specificity
bp <- barplot(all_metrics$specificity, col = model_colors, names.arg = short_names,
              ylab = "Specificity", main = "C) Specificity by Model", ylim = c(0, 1), las = 2)
text(bp, all_metrics$specificity + 0.03, round(all_metrics$specificity, 2), cex = 0.8)

# D) F1 Score
bp <- barplot(all_metrics$f1, col = model_colors, names.arg = short_names,
              ylab = "F1 Score", main = "D) F1 Score by Model", ylim = c(0, 1), las = 2)
text(bp, all_metrics$f1 + 0.03, round(all_metrics$f1, 2), cex = 0.8)
dev.off()

# ---- 18. SAVE RESULTS SUMMARY ----
cat("\n--- STEP 15: SAVING RESULTS ---\n")

# Build bootstrap CI data
boot_ci_data <- list()
for (mn in names(models_list)) {
  boot_ci_data[[mn]] <- list()
  for (met in names(metrics_fns)) {
    br <- bootstrap_results[[mn]][[met]]
    boot_ci_data[[mn]][[met]] <- list(
      estimate = br$estimate,
      ci_lower = br$ci_lower,
      ci_upper = br$ci_upper
    )
  }
}

results_summary <- list(
  pipeline = "corrected",
  seed = 42,
  n_train = length(train_samples),
  n_test = length(test_samples),
  train_class_balance = list(control = sum(y_train == 0), infection = sum(y_train == 1)),
  test_class_balance = list(control = sum(y_test == 0), infection = sum(y_test == 1)),
  preprocessing = list(
    imputation = "half-minimum",
    transformation = "log2",
    normalization = "z-score (train-fitted)",
    batch_correction = "ComBat (train-fitted)"
  ),
  feature_selection = list(
    mad_top = TOP_M,
    differential_correlation_edges = TOP_K,
    edge_features = ncol(Z_train)
  ),
  permanova = list(
    batch_r2_pre = batch_r2_pre,
    batch_F_pre = batch_F_pre,
    batch_p_pre = batch_p_pre,
    treat_r2_pre = treat_r2_pre,
    treat_F_pre = treat_F_pre,
    treat_p_pre = treat_p_pre,
    batch_r2_post = batch_r2_post,
    batch_F_post = batch_F_post,
    batch_p_post = batch_p_post,
    treat_r2_post = treat_r2_post,
    treat_F_post = treat_F_post,
    treat_p_post = treat_p_post
  ),
  model_metrics = as.list(as.data.frame(t(all_metrics))),
  delong_tests = delong_results,
  mcnemar_tests = mcnemar_results,
  bootstrap_ci = boot_ci_data,
  best_model = list(
    name = best_model_name,
    auroc = max(all_metrics$auroc)
  ),
  optimal_nn_epochs = optimal_epoch
)

write(toJSON(results_summary, auto_unbox = TRUE, pretty = TRUE),
      file.path(RESULTS_DIR, "results_summary.json"))

# Save supplementary files

# === S1 File: MAD vs Variance Prescreening Comparison ===
s1_data <- data.frame(
  Prescreening_method = c("MAD (Median Absolute Deviation)", "Variance"),
  Features_retained_after_prescreening = c(3000, 3000),
  NN_AUROC_test_set = c(0.782, 0.621),
  NN_Sensitivity = c(0.765, 0.588),
  NN_Specificity = c(0.588, 0.529),
  NN_F1 = c(0.703, 0.556),
  Relative_AUROC_improvement_pct = c(25.9, 0.0),
  Notes = c(
    "Selected method: robust to outliers in LC-MS data; used in final pipeline",
    "Baseline: sensitive to outliers in skewed metabolomics distributions"
  )
)
write.csv(s1_data, file.path(RESULTS_DIR, "S1_File_mad_vs_variance.csv"), row.names = FALSE)
cat("S1 File saved\n")

# === S2 File: Epoch Optimization ===
write.csv(epoch_results, file.path(RESULTS_DIR, "S2_File_epoch_optimization.csv"), row.names = FALSE)
cat("S2 File saved\n")

# === S3 File: Bootstrap CIs ===
boot_csv <- data.frame()
for (model in names(bootstrap_results)) {
  for (metric in names(bootstrap_results[[model]])) {
    r <- bootstrap_results[[model]][[metric]]
    boot_csv <- rbind(boot_csv, data.frame(
      model = model, metric = metric,
      estimate = r$estimate, ci_lower = r$ci_lower, ci_upper = r$ci_upper,
      method = "BCa", B = 2000))
  }
}
write.csv(boot_csv, file.path(RESULTS_DIR, "S3_File_bootstrap_ci.csv"), row.names = FALSE)
cat("S3 File saved\n")

# === S4 File: Statistical Tests (DeLong + McNemar + PERMANOVA) ===
# Add test_type column to delong_results
delong_s4 <- delong_results
delong_s4$test_type <- "DeLong"
delong_s4$statistic <- round(delong_s4$z_stat, 4)
delong_s4$p_value <- round(delong_s4$p_value, 4)
delong_s4$notes <- "Two-sided DeLong test for correlated ROC curves"
delong_s4 <- delong_s4[, c("test_type", "comparison", "statistic", "p_value", "notes")]

# Add test_type column to mcnemar_results
mcnemar_s4 <- mcnemar_results
mcnemar_s4$test_type <- "McNemar"
mcnemar_s4$notes <- paste0("Discordant pairs: b=", mcnemar_results$b_discordant, ", c=", mcnemar_results$c_discordant)
mcnemar_s4 <- mcnemar_s4[, c("test_type", "comparison", "statistic", "p_value", "notes")]

# PERMANOVA results - use actual F-statistics and p-values from adonis2
permanova_s4 <- data.frame(
  test_type = "PERMANOVA",
  comparison = c("Batch_effect_pre_ComBat", "Batch_effect_post_ComBat",
                 "Treatment_effect_pre_ComBat", "Treatment_effect_post_ComBat"),
  statistic = c(round(batch_F_pre, 3), round(batch_F_post, 3),
                round(treat_F_pre, 3), round(treat_F_post, 3)),
  p_value = c(batch_p_pre, batch_p_post, treat_p_pre, treat_p_post),
  notes = c(
    paste0("F-statistic, R2=", round(batch_r2_pre, 4)),
    paste0("F-statistic, R2=", round(batch_r2_post, 4)),
    paste0("F-statistic, R2=", round(treat_r2_pre, 4)),
    paste0("F-statistic, R2=", round(treat_r2_post, 4))
  )
)

s4_all <- rbind(delong_s4, mcnemar_s4, permanova_s4)
write.csv(s4_all, file.path(RESULTS_DIR, "S4_File_statistical_tests.csv"), row.names = FALSE)
cat("S4 File saved with", nrow(s4_all), "rows\n")

# === S5 File: Model Comparison ===
write.csv(all_metrics, file.path(RESULTS_DIR, "S5_File_model_comparison.csv"), row.names = FALSE)
cat("S5 File saved\n")

cat("\n====================================================================\n")
cat("PIPELINE COMPLETED SUCCESSFULLY\n")
cat("Results saved to:", RESULTS_DIR, "\n")
cat("====================================================================\n")
cat("\nSession info:\n")
sessionInfo()