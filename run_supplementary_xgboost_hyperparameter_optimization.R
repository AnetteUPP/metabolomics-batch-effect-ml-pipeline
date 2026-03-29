#!/usr/bin/env Rscript
# Title: XGBoost Hyperparameter Optimization for Metabolomic Classification
# Date: 2026
# Description: Grid search over XGBoost hyperparameters using 5-fold CV on training set.
#              Identifies optimal configuration and validates on held-out test set.
# PLOS Computational Biology Supporting Information 
# ====================================================================

cat("\n====================================================================\n")
cat("XGBoost Hyperparameter Optimization (Staged Grid Search)\n")
cat("====================================================================\n\n")

# ---- 0. ENVIRONMENT ----
set.seed(42)
options(scipen = 999)
.libPaths(c("~/R/library", .libPaths()))

suppressPackageStartupMessages({
  library(xgboost)
  library(pROC)
  library(corpcor)
  library(ggplot2)
  library(reshape2)
  library(gridExtra)
  library(grid)
})

RESULTS_DIR <- "/home/ubuntu/results"
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# ---- 1. LOAD DATA ----
cat("Step 1: Loading corrected data...\n")
expr_corrected <- read.csv(file.path(RESULTS_DIR, "corrected_expression.csv"),
                           row.names = 1, check.names = FALSE)
meta <- read.csv("/home/ubuntu/Uploads/metadata.csv", stringsAsFactors = FALSE)
rownames(meta) <- meta$sample_id

cat(sprintf("  Data: %d samples x %d features\n", nrow(expr_corrected), ncol(expr_corrected)))

# ---- 2. REPRODUCE STRATIFIED SPLIT ----
cat("Step 2: Reproducing stratified train/test split...\n")
set.seed(42)
treatment_labels <- meta$treatment
unique_classes <- unique(treatment_labels)
train_idx_list <- c()
for (cls in unique_classes) {
  cls_indices <- which(treatment_labels == cls)
  n_train <- round(length(cls_indices) * 0.7)
  sampled <- sample(cls_indices, n_train)
  train_idx_list <- c(train_idx_list, sampled)
}
train_idx <- sort(train_idx_list)
test_idx <- setdiff(1:nrow(meta), train_idx)

train_ids <- meta$sample_id[train_idx]
test_ids <- meta$sample_id[test_idx]
y_train <- ifelse(meta$treatment[train_idx] == "infection", 1, 0)
y_test <- ifelse(meta$treatment[test_idx] == "infection", 1, 0)

cat(sprintf("  Train: %d (Ctrl=%d, Inf=%d) | Test: %d (Ctrl=%d, Inf=%d)\n",
            length(train_ids), sum(y_train == 0), sum(y_train == 1),
            length(test_ids), sum(y_test == 0), sum(y_test == 1)))

# ---- 3. FEATURE SELECTION (same as main pipeline) ----
cat("Step 3: Feature selection (MAD + differential correlation)...\n")
train_expr <- as.matrix(expr_corrected[train_ids, ])
test_expr <- as.matrix(expr_corrected[test_ids, ])

mad_vals <- apply(train_expr, 2, mad)
top_M <- 3000
top_features <- names(sort(mad_vals, decreasing = TRUE)[1:top_M])

train_ctrl <- train_expr[y_train == 0, top_features]
train_inf <- train_expr[y_train == 1, top_features]

cor_ctrl <- cor.shrink(train_ctrl, verbose = FALSE)
cor_inf <- cor.shrink(train_inf, verbose = FALSE)

diff_cor <- abs(as.matrix(cor_inf) - as.matrix(cor_ctrl))
diag(diff_cor) <- 0

K <- 120
upper_tri <- diff_cor[upper.tri(diff_cor)]
names_mat <- which(upper.tri(diff_cor), arr.ind = TRUE)
edge_scores <- data.frame(row = names_mat[, 1], col = names_mat[, 2], diff = upper_tri)
edge_scores <- edge_scores[order(-edge_scores$diff), ]
top_edges <- edge_scores[1:K, ]

sel_idx <- unique(c(top_edges$row, top_edges$col))
selected_features <- top_features[sel_idx]
cat(sprintf("  Selected %d features from %d edges\n", length(selected_features), K))

# Use edge interaction features (product of correlated pairs) as in main pipeline
# Fisher z-transform for differential correlation
fisher_z <- function(r) 0.5 * log((1 + r) / (1 - r))
z_ctrl <- fisher_z(as.matrix(cor_ctrl))
z_inf <- fisher_z(as.matrix(cor_inf))
delta_z <- z_inf - z_ctrl
upper_idx_mat <- upper.tri(delta_z)
edge_values <- abs(delta_z[upper_idx_mat])
edge_order <- order(edge_values, decreasing = TRUE)
edge_indices <- which(upper_idx_mat, arr.ind = TRUE)
edge_indices <- edge_indices[edge_order[1:K], ]

create_edge_features <- function(X, edges) {
  features <- matrix(0, nrow = nrow(X), ncol = nrow(edges))
  for (i in 1:nrow(edges)) {
    features[, i] <- X[, edges[i, 1]] * X[, edges[i, 2]]
  }
  colnames(features) <- paste0("edge_", 1:nrow(edges))
  return(features)
}

X_train_filtered <- train_expr[, top_features]
X_test_filtered <- test_expr[, top_features]
X_train <- create_edge_features(X_train_filtered, edge_indices)
X_test <- create_edge_features(X_test_filtered, edge_indices)
cat(sprintf("  Edge interaction features created: %d\n", ncol(X_train)))

# Create DMatrix objects
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)

# ---- 4. CV FOLD CREATION ----
n_folds <- 5
set.seed(42)
folds <- sample(rep(1:n_folds, length.out = nrow(X_train)))
cat(sprintf("  5-fold CV created (fold sizes: %s)\n",
            paste(table(folds), collapse = ", ")))

# Helper: run CV for a given param set, return mean and sd AUROC
run_cv <- function(params, nrounds_val) {
  fold_aurocs <- numeric(n_folds)
  for (k in 1:n_folds) {
    train_k <- which(folds != k)
    val_k <- which(folds == k)
    
    d_train_k <- xgb.DMatrix(data = X_train[train_k, , drop = FALSE], label = y_train[train_k])
    d_val_k <- xgb.DMatrix(data = X_train[val_k, , drop = FALSE], label = y_train[val_k])
    
    model <- tryCatch({
      xgb.train(
        params = params,
        data = d_train_k,
        nrounds = nrounds_val,
        verbose = 0
      )
    }, error = function(e) NULL)
    
    if (!is.null(model)) {
      probs <- predict(model, d_val_k)
      fold_aurocs[k] <- tryCatch(
        as.numeric(roc(y_train[val_k], probs, quiet = TRUE)$auc),
        error = function(e) 0.5)
    } else {
      fold_aurocs[k] <- 0.5
    }
  }
  return(c(mean = mean(fold_aurocs), sd = sd(fold_aurocs)))
}

# ---- 5. STAGE 1: max_depth x eta ----
cat("\n--- STAGE 1: Learning Rate (eta) x Tree Depth (max_depth) ---\n")
cat("  Fixed: subsample=0.8, colsample_bytree=0.8, nrounds=100\n")

depth_values <- c(2, 3, 4, 6, 8)
eta_values <- c(0.01, 0.05, 0.1, 0.2, 0.3)

stage1_results <- expand.grid(max_depth = depth_values, eta = eta_values)
stage1_results$mean_cv_auroc <- NA
stage1_results$sd_cv_auroc <- NA

for (i in 1:nrow(stage1_results)) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = stage1_results$max_depth[i],
    eta = stage1_results$eta[i],
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 1
  )
  res <- run_cv(params, nrounds_val = 100)
  stage1_results$mean_cv_auroc[i] <- res["mean"]
  stage1_results$sd_cv_auroc[i] <- res["sd"]
  if (i %% 5 == 0) cat(sprintf("  Progress: %d/%d\n", i, nrow(stage1_results)))
}

best_s1_idx <- which.max(stage1_results$mean_cv_auroc)
best_depth <- stage1_results$max_depth[best_s1_idx]
best_eta <- stage1_results$eta[best_s1_idx]
cat(sprintf("  STAGE 1 BEST: max_depth=%d, eta=%.2f, CV AUROC=%.3f (±%.3f)\n",
            best_depth, best_eta,
            stage1_results$mean_cv_auroc[best_s1_idx],
            stage1_results$sd_cv_auroc[best_s1_idx]))

# ---- 6. STAGE 2: nrounds x subsample ----
cat("\n--- STAGE 2: Number of Trees (nrounds) x Subsample Rate ---\n")
cat(sprintf("  Fixed: max_depth=%d, eta=%.2f, colsample_bytree=0.8\n", best_depth, best_eta))

nrounds_values <- c(50, 100, 200, 300)
subsample_values <- c(0.6, 0.8, 1.0)

stage2_results <- expand.grid(nrounds = nrounds_values, subsample = subsample_values)
stage2_results$mean_cv_auroc <- NA
stage2_results$sd_cv_auroc <- NA

for (i in 1:nrow(stage2_results)) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = best_depth,
    eta = best_eta,
    subsample = stage2_results$subsample[i],
    colsample_bytree = 0.8,
    min_child_weight = 1
  )
  res <- run_cv(params, nrounds_val = stage2_results$nrounds[i])
  stage2_results$mean_cv_auroc[i] <- res["mean"]
  stage2_results$sd_cv_auroc[i] <- res["sd"]
}

best_s2_idx <- which.max(stage2_results$mean_cv_auroc)
best_nrounds <- stage2_results$nrounds[best_s2_idx]
best_subsample <- stage2_results$subsample[best_s2_idx]
cat(sprintf("  STAGE 2 BEST: nrounds=%d, subsample=%.1f, CV AUROC=%.3f (±%.3f)\n",
            best_nrounds, best_subsample,
            stage2_results$mean_cv_auroc[best_s2_idx],
            stage2_results$sd_cv_auroc[best_s2_idx]))

# ---- 7. STAGE 3: colsample_bytree x min_child_weight ----
cat("\n--- STAGE 3: Column Sampling x Min Child Weight ---\n")
cat(sprintf("  Fixed: max_depth=%d, eta=%.2f, nrounds=%d, subsample=%.1f\n",
            best_depth, best_eta, best_nrounds, best_subsample))

colsample_values <- c(0.5, 0.8, 1.0)
min_child_values <- c(1, 3, 5)

stage3_results <- expand.grid(colsample_bytree = colsample_values,
                               min_child_weight = min_child_values)
stage3_results$mean_cv_auroc <- NA
stage3_results$sd_cv_auroc <- NA

for (i in 1:nrow(stage3_results)) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = best_depth,
    eta = best_eta,
    subsample = best_subsample,
    colsample_bytree = stage3_results$colsample_bytree[i],
    min_child_weight = stage3_results$min_child_weight[i]
  )
  res <- run_cv(params, nrounds_val = best_nrounds)
  stage3_results$mean_cv_auroc[i] <- res["mean"]
  stage3_results$sd_cv_auroc[i] <- res["sd"]
}

best_s3_idx <- which.max(stage3_results$mean_cv_auroc)
best_colsample <- stage3_results$colsample_bytree[best_s3_idx]
best_min_child <- stage3_results$min_child_weight[best_s3_idx]
cat(sprintf("  STAGE 3 BEST: colsample_bytree=%.1f, min_child_weight=%d, CV AUROC=%.3f (±%.3f)\n",
            best_colsample, best_min_child,
            stage3_results$mean_cv_auroc[best_s3_idx],
            stage3_results$sd_cv_auroc[best_s3_idx]))

# ---- 8. FINAL OPTIMAL CONFIGURATION ----
cat("\n====================================================================\n")
cat("FINAL OPTIMAL CONFIGURATION:\n")
cat(sprintf("  max_depth     = %d\n", best_depth))
cat(sprintf("  eta           = %.2f\n", best_eta))
cat(sprintf("  nrounds       = %d\n", best_nrounds))
cat(sprintf("  subsample     = %.1f\n", best_subsample))
cat(sprintf("  colsample_bytree = %.1f\n", best_colsample))
cat(sprintf("  min_child_weight = %d\n", best_min_child))
cat("====================================================================\n")

# ---- 9. FINAL VALIDATION ON TEST SET ----
cat("\nStep 9: Final validation on test set...\n")

# Default model (from main pipeline: max_depth=3, early stopping via xgb.cv)
default_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 3,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1
)

set.seed(42)
default_cv <- xgb.cv(params = default_params, data = dtrain, nrounds = 500,
                      nfold = 5, early_stopping_rounds = 20, verbose = 0, maximize = TRUE)
default_nrounds <- default_cv$best_iteration
cat(sprintf("  Default early-stopping nrounds: %d\n", default_nrounds))

set.seed(42)
default_model <- xgb.train(params = default_params, data = dtrain, nrounds = default_nrounds, verbose = 0)
default_probs <- predict(default_model, dtest)
default_auroc <- as.numeric(roc(y_test, default_probs, quiet = TRUE)$auc)

# Optimized model
optimal_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = best_depth,
  eta = best_eta,
  subsample = best_subsample,
  colsample_bytree = best_colsample,
  min_child_weight = best_min_child
)

set.seed(42)
optimal_model <- xgb.train(params = optimal_params, data = dtrain, nrounds = best_nrounds, verbose = 0)
optimal_probs <- predict(optimal_model, dtest)
optimal_auroc <- as.numeric(roc(y_test, optimal_probs, quiet = TRUE)$auc)

# ROC objects for DeLong
roc_default <- roc(y_test, default_probs, quiet = TRUE)
roc_optimal <- roc(y_test, optimal_probs, quiet = TRUE)

# Youden's J for optimal threshold
coords_opt <- coords(roc_optimal, "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"))
# coords may return multiple rows; take the first
if (is.data.frame(coords_opt) && nrow(coords_opt) > 1) coords_opt <- coords_opt[1, ]
opt_threshold <- as.numeric(coords_opt$threshold)
opt_sensitivity <- as.numeric(coords_opt$sensitivity)
opt_specificity <- as.numeric(coords_opt$specificity)
pred_classes <- ifelse(optimal_probs >= opt_threshold, 1, 0)
tp <- sum(pred_classes == 1 & y_test == 1)
fp <- sum(pred_classes == 1 & y_test == 0)
fn <- sum(pred_classes == 0 & y_test == 1)
opt_precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
opt_f1 <- ifelse(opt_precision + opt_sensitivity > 0,
                 2 * opt_precision * opt_sensitivity / (opt_precision + opt_sensitivity), 0)

cat(sprintf("\n  Default XGBoost AUROC on test set:   %.3f\n", default_auroc))
cat(sprintf("  Optimized XGBoost AUROC on test set: %.3f\n", optimal_auroc))

# DeLong test
delong_result <- tryCatch(
  roc.test(roc_default, roc_optimal, method = "delong"),
  error = function(e) { cat("  DeLong test failed:", e$message, "\n"); NULL }
)
if (!is.null(delong_result)) {
  cat(sprintf("  DeLong p-value: %.4f\n", delong_result$p.value))
}

cat(sprintf("\n  At optimal threshold (%.3f):\n", opt_threshold))
cat(sprintf("    Sensitivity: %.3f\n", opt_sensitivity))
cat(sprintf("    Specificity: %.3f\n", opt_specificity))
cat(sprintf("    F1 Score:    %.3f\n", opt_f1))

if (optimal_auroc > default_auroc) {
  cat("\n  >> Optimization IMPROVED performance.\n")
} else if (optimal_auroc == default_auroc) {
  cat("\n  >> Optimization did NOT change performance.\n")
} else {
  cat("\n  >> Optimization did NOT improve performance (default was better).\n")
}

# ---- 10. BOOTSTRAP CIs ----
cat("\nStep 10: Bootstrap confidence intervals (B=500)...\n")
set.seed(42)
B <- 500
boot_default <- numeric(B)
boot_optimal <- numeric(B)

for (b in 1:B) {
  idx <- sample(1:length(y_test), replace = TRUE)
  if (length(unique(y_test[idx])) < 2) next
  boot_default[b] <- tryCatch(as.numeric(roc(y_test[idx], default_probs[idx], quiet = TRUE)$auc),
                                error = function(e) NA)
  boot_optimal[b] <- tryCatch(as.numeric(roc(y_test[idx], optimal_probs[idx], quiet = TRUE)$auc),
                                error = function(e) NA)
}
boot_default <- boot_default[!is.na(boot_default)]
boot_optimal <- boot_optimal[!is.na(boot_optimal)]

ci_default <- quantile(boot_default, c(0.025, 0.975))
ci_optimal <- quantile(boot_optimal, c(0.025, 0.975))

cat(sprintf("  Default  95%% CI: [%.3f, %.3f]\n", ci_default[1], ci_default[2]))
cat(sprintf("  Optimized 95%% CI: [%.3f, %.3f]\n", ci_optimal[1], ci_optimal[2]))

# ---- 11. SAVE CSV RESULTS ----
cat("\nStep 11: Saving grid search results...\n")

# Build combined CSV
s1_csv <- data.frame(
  stage = "Stage1_depth_eta",
  max_depth = stage1_results$max_depth,
  eta = stage1_results$eta,
  nrounds = 100,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  mean_cv_auroc = stage1_results$mean_cv_auroc,
  sd_cv_auroc = stage1_results$sd_cv_auroc
)

s2_csv <- data.frame(
  stage = "Stage2_nrounds_subsample",
  max_depth = best_depth,
  eta = best_eta,
  nrounds = stage2_results$nrounds,
  subsample = stage2_results$subsample,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  mean_cv_auroc = stage2_results$mean_cv_auroc,
  sd_cv_auroc = stage2_results$sd_cv_auroc
)

s3_csv <- data.frame(
  stage = "Stage3_colsample_minchild",
  max_depth = best_depth,
  eta = best_eta,
  nrounds = best_nrounds,
  subsample = best_subsample,
  colsample_bytree = stage3_results$colsample_bytree,
  min_child_weight = stage3_results$min_child_weight,
  mean_cv_auroc = stage3_results$mean_cv_auroc,
  sd_cv_auroc = stage3_results$sd_cv_auroc
)

all_results <- rbind(s1_csv, s2_csv, s3_csv)

# Add test_auroc and test_auroc_note columns
all_results$test_auroc <- NA
all_results$test_auroc_note <- ""

# Mark the best CV row with optimized test AUROC
best_cv_row <- which.max(all_results$mean_cv_auroc)
all_results$test_auroc[best_cv_row] <- optimal_auroc
all_results$test_auroc_note[best_cv_row] <- "CV-optimal config evaluated on test set"

# Mark default params row (max_depth=3, eta=0.1 from stage 1)
default_row <- which(all_results$max_depth == 3 & abs(all_results$eta - 0.1) < 0.01)
if (length(default_row) > 0) {
  all_results$test_auroc[default_row[1]] <- default_auroc
  all_results$test_auroc_note[default_row[1]] <- "Default XGBoost config on test set"
}

write.csv(all_results, file.path(RESULTS_DIR, "S6_File_xgboost_hyperparameter_grid_results.csv"),
          row.names = FALSE)
cat("  Saved to S6_File_xgboost_hyperparameter_grid_results.csv\n")

# ---- 12. GENERATE FIGURES ----
cat("\nStep 12: Generating publication-quality figure...\n")

# --- Panel A: Stage 1 Heatmap ---
s1_mat <- dcast(stage1_results, max_depth ~ eta, value.var = "mean_cv_auroc")
s1_melt <- melt(stage1_results, id.vars = c("max_depth", "eta"),
                measure.vars = "mean_cv_auroc")

# Find best cell
s1_best <- stage1_results[best_s1_idx, ]

pA <- ggplot(stage1_results, aes(x = factor(eta), y = factor(max_depth), fill = mean_cv_auroc)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = sprintf("%.3f", mean_cv_auroc)), color = "white", size = 3.2) +
  geom_tile(data = s1_best, aes(x = factor(eta), y = factor(max_depth)),
            fill = NA, color = "gold", size = 1.5) +
  scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
                       midpoint = median(stage1_results$mean_cv_auroc),
                       name = "CV\nAUROC") +
  labs(title = "A) Stage 1: Learning Rate × Tree Depth",
       x = "Learning Rate (eta)", y = "Tree Depth (max_depth)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 12),
        panel.grid = element_blank())

# --- Panel B: Stage 2 Line Plot ---
stage2_results$subsample_f <- factor(paste0("subsample = ", stage2_results$subsample))

# Mark best point
s2_best <- stage2_results[best_s2_idx, ]

pB <- ggplot(stage2_results, aes(x = nrounds, y = mean_cv_auroc,
                                  color = subsample_f, group = subsample_f)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  geom_errorbar(aes(ymin = mean_cv_auroc - sd_cv_auroc,
                    ymax = mean_cv_auroc + sd_cv_auroc),
                width = 15, size = 0.5) +
  geom_point(data = s2_best, aes(x = nrounds, y = mean_cv_auroc),
             shape = 8, size = 4, color = "red", stroke = 1.5) +
  scale_color_manual(values = c("#2166AC", "#4DAF4A", "#E7298A"),
                     name = "") +
  labs(title = "B) Stage 2: Number of Trees × Subsample Rate",
       x = "Number of Trees (nrounds)", y = "Mean CV AUROC (±1 SD)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 12),
        legend.position = c(0.7, 0.25))

# --- Panel C: Stage 3 Heatmap ---
s3_best <- stage3_results[best_s3_idx, ]

pC <- ggplot(stage3_results, aes(x = factor(min_child_weight),
                                  y = factor(colsample_bytree),
                                  fill = mean_cv_auroc)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = sprintf("%.3f", mean_cv_auroc)), color = "white", size = 3.5) +
  geom_tile(data = s3_best, aes(x = factor(min_child_weight),
                                 y = factor(colsample_bytree)),
            fill = NA, color = "gold", size = 1.5) +
  scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
                       midpoint = median(stage3_results$mean_cv_auroc),
                       name = "CV\nAUROC") +
  labs(title = "C) Stage 3: Column Sampling × Min Child Weight",
       x = "Min Child Weight", y = "Column Sample Rate") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 12),
        panel.grid = element_blank())

# --- Panel D: Default vs Optimized Bar Chart ---
bar_df <- data.frame(
  Model = c("Default\nXGBoost", "Optimized\nXGBoost"),
  AUROC = c(default_auroc, optimal_auroc),
  CI_lower = c(ci_default[1], ci_optimal[1]),
  CI_upper = c(ci_default[2], ci_optimal[2])
)
bar_df$Model <- factor(bar_df$Model, levels = bar_df$Model)

delong_label <- if (!is.null(delong_result)) {
  sprintf("DeLong p = %.3f", delong_result$p.value)
} else {
  ""
}

pD <- ggplot(bar_df, aes(x = Model, y = AUROC, fill = Model)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper), width = 0.15, size = 0.7) +
  geom_text(aes(label = sprintf("%.3f", AUROC)), vjust = -1.8, size = 4) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray60") +
  scale_fill_manual(values = c("#8C510A", "#01665E")) +
  scale_y_continuous(limits = c(0, 1.05), expand = c(0, 0)) +
  annotate("text", x = 1.5, y = 0.98, label = delong_label, size = 3.5, color = "red") +
  labs(title = "D) Default vs. Optimized XGBoost",
       x = "", y = "Test Set AUROC") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 12))

# --- Combine panels ---
combined <- arrangeGrob(pA, pB, pC, pD, ncol = 2,
                        top = textGrob("S4 Fig. XGBoost Hyperparameter Optimization",
                                       gp = gpar(fontsize = 16, fontface = "bold"),
                                       vjust = 1))

ggsave(file.path(RESULTS_DIR, "S4_Fig.png"),
       plot = combined, width = 12, height = 10, dpi = 300)

cat("  Figure saved to S4_Fig.png\n")

# ---- 13. SUMMARY ----
cat("\n====================================================================\n")
cat("SUMMARY\n")
cat("====================================================================\n")
cat(sprintf("Default XGBoost AUROC:   %.3f [95%% CI: %.3f-%.3f]\n",
            default_auroc, ci_default[1], ci_default[2]))
cat(sprintf("Optimized XGBoost AUROC: %.3f [95%% CI: %.3f-%.3f]\n",
            optimal_auroc, ci_optimal[1], ci_optimal[2]))
if (!is.null(delong_result)) {
  cat(sprintf("DeLong p-value:          %.4f\n", delong_result$p.value))
}
cat(sprintf("Sensitivity: %.3f | Specificity: %.3f | F1: %.3f\n",
            opt_sensitivity, opt_specificity, opt_f1))
cat("====================================================================\n")
cat("Done.\n")
