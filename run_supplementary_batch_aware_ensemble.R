#!/usr/bin/env Rscript
# Title: Batch-Aware Ensemble Learning Analysis
# ====================================================================
# Supplementary Analysis: Batch-Aware Ensemble Learning
# ====================================================================
# Uses corrected_expression.csv and same stratified split (set.seed(42))

set.seed(42)
.libPaths(c("~/R/library", .libPaths()))
cat("=== Batch-Aware Ensemble Learning Analysis ===\n")

suppressPackageStartupMessages({
  library(e1071)
  library(kernlab)
  library(pROC)
  library(corpcor)
  library(vegan)
})

RESULTS_DIR <- "/home/ubuntu/results"
DATA_DIR <- "/home/ubuntu/Uploads"

# ---- Load Data ----
cat("Step 1: Loading data...\n")
expr_corrected <- read.csv(file.path(RESULTS_DIR, "corrected_expression.csv"),
                           row.names = 1, check.names = FALSE)
meta <- read.csv(file.path(DATA_DIR, "metadata.csv"))
rownames(meta) <- meta$sample_id

# Also load raw for no-correction strategy
expr_raw <- read.csv(file.path(DATA_DIR, "processed_table.csv"), row.names = 1, check.names = FALSE)

# Reproduce stratified split
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

cat(sprintf("  Train: %d | Test: %d\n", length(train_ids), length(test_ids)))

# Batch info
batch_train <- meta$batch[train_idx]
batch_test <- meta$batch[test_idx]
cat(sprintf("  Train batches: B1=%d, B2=%d\n", sum(batch_train == 1), sum(batch_train == 2)))
cat(sprintf("  Test batches: B1=%d, B2=%d\n", sum(batch_test == 1), sum(batch_test == 2)))

# ---- Helper Functions ----
select_features <- function(X_train, y_train, M = 3000, K = 120) {
  mad_vals <- apply(X_train, 2, mad)
  top_features <- names(sort(mad_vals, decreasing = TRUE)[1:M])
  X_ctrl <- X_train[y_train == 0, top_features]
  X_inf <- X_train[y_train == 1, top_features]
  cor_ctrl <- cor.shrink(X_ctrl, verbose = FALSE)
  cor_inf <- cor.shrink(X_inf, verbose = FALSE)
  diff_cor <- abs(as.matrix(cor_inf) - as.matrix(cor_ctrl))
  diag(diff_cor) <- 0
  upper_tri <- diff_cor[upper.tri(diff_cor)]
  names_mat <- which(upper.tri(diff_cor), arr.ind = TRUE)
  edge_scores <- data.frame(row = names_mat[, 1], col = names_mat[, 2], diff = upper_tri)
  edge_scores <- edge_scores[order(-edge_scores$diff), ]
  top_edges <- edge_scores[1:K, ]
  sel_idx <- unique(c(top_edges$row, top_edges$col))
  return(top_features[sel_idx])
}

train_svm_rbf <- function(X_train, y_train, X_test) {
  model <- ksvm(x = as.matrix(X_train), y = factor(y_train),
                kernel = "rbfdot", C = 1, prob.model = TRUE)
  probs <- predict(model, as.matrix(X_test), type = "probabilities")[, "1"]
  return(probs)
}

evaluate_model <- function(probs, y_true) {
  roc_obj <- roc(y_true, probs, quiet = TRUE)
  preds <- ifelse(probs >= 0.5, 1, 0)
  tp <- sum(preds == 1 & y_true == 1)
  fn <- sum(preds == 0 & y_true == 1)
  tn <- sum(preds == 0 & y_true == 0)
  fp <- sum(preds == 1 & y_true == 0)
  sens <- tp / (tp + fn)
  spec <- tn / (tn + fp)
  prec <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
  f1 <- ifelse(prec + sens > 0, 2 * prec * sens / (prec + sens), 0)
  list(auroc = as.numeric(roc_obj$auc), sensitivity = sens,
       specificity = spec, f1 = f1, roc = roc_obj)
}

# ---- Preprocess raw data for no-correction strategy ----
expr_imp <- expr_raw
for (j in 1:ncol(expr_imp)) {
  vals <- expr_imp[, j]
  pos_vals <- vals[vals > 0 & !is.na(vals)]
  if (length(pos_vals) > 0) {
    half_min <- min(pos_vals) / 2
    expr_imp[is.na(vals) | vals <= 0, j] <- half_min
  }
}
expr_log <- log2(expr_imp)
# Z-score from train only
train_means <- colMeans(expr_log[train_ids, ])
train_sds <- apply(expr_log[train_ids, ], 2, sd)
train_sds[train_sds == 0] <- 1
expr_z_train <- scale(expr_log[train_ids, ], center = train_means, scale = train_sds)
expr_z_test <- scale(expr_log[test_ids, ], center = train_means, scale = train_sds)
expr_z_train[is.na(expr_z_train)] <- 0
expr_z_test[is.na(expr_z_test)] <- 0

# ---- Strategy 1: No Correction ----
cat("\nStrategy 1: No batch correction...\n")
features_nocorr <- select_features(expr_z_train, y_train)
probs_nocorr <- train_svm_rbf(expr_z_train[, features_nocorr], y_train,
                               expr_z_test[, features_nocorr])
results_nocorr <- evaluate_model(probs_nocorr, y_test)
cat(sprintf("  AUROC: %.3f\n", results_nocorr$auroc))

# ---- Strategy 2: ComBat (from corrected_expression.csv) ----
cat("Strategy 2: ComBat correction...\n")
features_combat <- select_features(as.matrix(expr_corrected[train_ids, ]), y_train)
probs_combat <- train_svm_rbf(expr_corrected[train_ids, features_combat], y_train,
                               expr_corrected[test_ids, features_combat])
results_combat <- evaluate_model(probs_combat, y_test)
cat(sprintf("  AUROC: %.3f\n", results_combat$auroc))

# ---- Strategy 3: Batch-Invariant ----
cat("Strategy 3: Batch-invariant features...\n")
train_expr_z <- expr_z_train
mad_vals <- apply(train_expr_z, 2, mad)
top3k <- names(sort(mad_vals, decreasing = TRUE)[1:3000])

treatment_f <- numeric(length(top3k))
batch_f <- numeric(length(top3k))
names(treatment_f) <- names(batch_f) <- top3k

for (feat in top3k) {
  vals <- train_expr_z[, feat]
  treat <- factor(y_train)
  bat <- factor(batch_train)
  treatment_f[feat] <- tryCatch(summary(aov(vals ~ treat))[[1]][1, "F value"], error = function(e) 0)
  batch_f[feat] <- tryCatch(summary(aov(vals ~ bat))[[1]][1, "F value"], error = function(e) 0)
}

invariance_score <- treatment_f / (batch_f + 1)
top_invariant <- names(sort(invariance_score, decreasing = TRUE)[1:200])

probs_invariant <- train_svm_rbf(train_expr_z[, top_invariant], y_train,
                                  expr_z_test[, top_invariant])
results_invariant <- evaluate_model(probs_invariant, y_test)
cat(sprintf("  AUROC: %.3f\n", results_invariant$auroc))

# ---- Strategy 4: Domain Adaptation ----
cat("Strategy 4: Domain adaptation...\n")
domain_labels <- c(rep(0, length(train_ids)), rep(1, length(test_ids)))
all_features <- features_combat
X_domain <- rbind(as.matrix(expr_corrected[train_ids, all_features]),
                  as.matrix(expr_corrected[test_ids, all_features]))

domain_model <- tryCatch({
  glm(domain_labels ~ ., data = data.frame(X_domain), family = "binomial")
}, warning = function(w) {
  suppressWarnings(glm(domain_labels ~ ., data = data.frame(X_domain), family = "binomial"))
})

domain_probs <- predict(domain_model, type = "response")
train_domain_probs <- domain_probs[1:length(train_ids)]
weights <- train_domain_probs / (1 - train_domain_probs + 1e-6)
weights <- pmin(pmax(weights, 0.1), 10)
weights <- weights / mean(weights)

model_da <- svm(x = as.matrix(expr_corrected[train_ids, features_combat]),
                y = factor(y_train), kernel = "radial",
                probability = TRUE, scale = FALSE,
                class.weights = c("0" = mean(weights[y_train == 0]),
                                  "1" = mean(weights[y_train == 1])))
pred_da <- predict(model_da, as.matrix(expr_corrected[test_ids, features_combat]), probability = TRUE)
probs_da <- attr(pred_da, "probabilities")[, "1"]
results_da <- evaluate_model(probs_da, y_test)
cat(sprintf("  AUROC: %.3f\n", results_da$auroc))

# ---- Strategy 5: Meta-Ensemble ----
cat("Strategy 5: Meta-ensemble...\n")
probs_ensemble <- (probs_combat + probs_invariant + probs_da) / 3
results_ensemble <- evaluate_model(probs_ensemble, y_test)
cat(sprintf("  AUROC: %.3f\n", results_ensemble$auroc))

# ---- DeLong Tests ----
cat("\nDeLong tests vs ComBat baseline...\n")
strategies <- list(
  "No Correction" = results_nocorr,
  "ComBat (Baseline)" = results_combat,
  "Batch-Invariant" = results_invariant,
  "Domain Adaptation" = results_da,
  "Meta-Ensemble" = results_ensemble
)

delong_results <- data.frame()
for (nm in names(strategies)) {
  if (nm == "ComBat (Baseline)") next
  test_result <- roc.test(strategies[["ComBat (Baseline)"]]$roc,
                          strategies[[nm]]$roc, method = "delong")
  delong_results <- rbind(delong_results, data.frame(
    Comparison = paste("ComBat vs", nm),
    AUROC_ComBat = strategies[["ComBat (Baseline)"]]$auroc,
    AUROC_Other = strategies[[nm]]$auroc,
    Z_statistic = test_result$statistic,
    p_value = test_result$p.value, stringsAsFactors = FALSE))
  cat(sprintf("  ComBat vs %s: p = %.4f\n", nm, test_result$p.value))
}

# ---- Save Files ----
comparison_df <- data.frame(
  Strategy = names(strategies),
  AUROC = sapply(strategies, function(x) round(x$auroc, 4)),
  Sensitivity = sapply(strategies, function(x) round(x$sensitivity, 4)),
  Specificity = sapply(strategies, function(x) round(x$specificity, 4)),
  F1 = sapply(strategies, function(x) round(x$f1, 4)),
  row.names = NULL
)
write.csv(comparison_df, file.path(RESULTS_DIR, "S7_File_batch_aware_comparison.csv"), row.names = FALSE)
write.csv(delong_results, file.path(RESULTS_DIR, "S8_File_batch_aware_delong_tests.csv"), row.names = FALSE)

# ---- Generate Figure ----
cat("Generating S2 Fig...\n")
png(file.path(RESULTS_DIR, "S2_Fig_batch_aware_ensemble.png"),
    width = 16, height = 6, units = "in", res = 300)
par(mfrow = c(1, 3), mar = c(6, 5, 4, 2), oma = c(0, 0, 2, 0))

# Panel A
aurocs <- sapply(strategies, function(x) x$auroc)
bar_colors <- c("#999999", "#4393C3", "#2166AC", "#D6604D", "#B2182B")
bp <- barplot(aurocs, col = bar_colors, ylim = c(0, 1),
              ylab = "AUROC", main = "A) AUROC by Strategy",
              names.arg = rep("", length(aurocs)), border = NA)
text(bp, aurocs + 0.03, sprintf("%.3f", aurocs), cex = 0.8, font = 2)
text(bp, par("usr")[3] - 0.05, srt = 35, adj = 1, xpd = TRUE,
     labels = names(strategies), cex = 0.75)
abline(h = 0.5, lty = 2, col = "gray60")
abline(h = results_combat$auroc, lty = 3, col = "#4393C3", lwd = 1.5)

# Panel B
batch_split_table <- matrix(c(
  sum(batch_train == 1), sum(batch_test == 1),
  sum(batch_train == 2), sum(batch_test == 2)
), nrow = 2, byrow = TRUE, dimnames = list(c("Batch 1", "Batch 2"), c("Train", "Test")))
barplot(batch_split_table, beside = TRUE, col = c("#4393C3", "#D6604D"),
        ylab = "Number of Samples", main = "B) Batch-Split Confounding",
        legend.text = rownames(batch_split_table),
        args.legend = list(x = "topright", bty = "n"))

# Panel C
if (nrow(delong_results) > 0) {
  n_comp <- nrow(delong_results)
  plot(delong_results$p_value, n_comp:1,
       xlim = c(0, 1), ylim = c(0.5, n_comp + 0.5),
       pch = 16, cex = 1.5, col = "#2166AC",
       xlab = "DeLong p-value", ylab = "", yaxt = "n",
       main = "C) DeLong Tests vs ComBat")
  axis(2, at = n_comp:1, labels = gsub("ComBat vs ", "", delong_results$Comparison),
       las = 1, cex.axis = 0.8)
  abline(v = 0.05, lty = 2, col = "red", lwd = 1.5)
  text(0.05, n_comp + 0.3, expression(alpha == 0.05), col = "red", cex = 0.8, pos = 4)
  for (i in 1:n_comp) {
    segments(0, n_comp - i + 1, delong_results$p_value[i], n_comp - i + 1,
             col = "#67A9CF", lwd = 2)
  }
}
mtext("S2 Fig. Batch-Aware Ensemble Learning Analysis", outer = TRUE, cex = 1.2, font = 2)
dev.off()
cat("Done.\n")
