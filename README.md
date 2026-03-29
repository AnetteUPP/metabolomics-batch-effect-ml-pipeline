# LC-MS Metabolomics Classification Pipeline

## Overview
Machine learning pipeline for binary classification (Control vs Infection) of LC-MS metabolomic data, featuring differential correlation network features and six classifier architectures.

**Best Model: XGBoost (AUROC = 0.869, 95% BCa CI: 0.700–0.956)**

## Key Results

| Model | AUROC | Sensitivity | Specificity | F1 |
|-------|-------|------------|-------------|-----|
| XGBoost | 0.869 | 0.882 | 0.706 | 0.811 |
| Random Forest | 0.825 | 0.765 | 0.647 | 0.722 |
| SVM Linear | 0.785 | 0.647 | 0.647 | 0.647 |
| Neural Network | 0.782 | 0.765 | 0.588 | 0.703 |
| SVM RBF | 0.768 | 0.647 | 0.706 | 0.667 |
| Elastic Net | 0.754 | 0.706 | 0.647 | 0.686 |

## Pipeline Methodology
1. **Preprocessing**: Imputation (half-minimum) → Log₂ → Z-score normalization (train-fitted) → ComBat batch correction (train-fitted)
2. **Data Partitioning**: Stratified 70/30 split (set.seed(42)), 82 train / 34 test
3. **Feature Selection** (train only): MAD prescreening (top 3 000) → Ledoit-Wolf shrinkage correlation → Top 120 differential correlation edges
4. **Model Training**: Neural Network, Elastic Net, Random Forest, XGBoost, SVM Linear, SVM RBF
5. **Evaluation**: AUROC, Sensitivity, Specificity, F1, BCa Bootstrap CIs (B=2000), DeLong tests, McNemar tests

## File Structure

### Analysis Scripts (in S1_Code.zip)
| Script | Purpose | Generates |
|--------|---------|-----------|
| `run_complete_analysis.R` | Core pipeline | Fig 1–6, S1–S5 Files |
| `run_supplementary_xgboost_hyperparameter_optimization.R` | XGBoost grid search | S6 File |
| `run_supplementary_batch_aware_ensemble.R` | Batch-aware strategies | S2 Fig, S7–S8 Files |
| `run_supplementary_gnn_analysis.py` | Graph Neural Networks | S3 Fig, S9 File |

### Main Figures
- **Fig 1**: Batch effects PCA + PERMANOVA
- **Fig 2**: Neural network epoch optimization
- **Fig 3**: ROC curves (all 6 models)
- **Fig 4**: BCa Bootstrap confidence intervals
- **Fig 5**: Confusion matrices
- **Fig 6**: Model comparison (AUROC, Sensitivity, Specificity, F1)

### Supplementary Figures
- **S1 Fig**: Pipeline schematic overview
- **S2 Fig**: Batch-aware ensemble learning analysis
- **S3 Fig**: Graph Neural Network analysis

### Supplementary Data Files
| File | Description |
|------|-------------|
| S1 File | MAD vs variance prescreening comparison (2 rows) |
| S2 File | NN epoch-by-epoch training logs |
| S3 File | BCa Bootstrap confidence intervals |
| S4 File | Statistical tests (DeLong + McNemar + PERMANOVA) |
| S5 File | Model comparison summary |
| S6 File | XGBoost hyperparameter grid search with test AUROC |
| S7 File | Batch-aware ensemble comparison |
| S8 File | Batch-aware DeLong tests |
| S9 File | GNN model comparison with XGBoost baseline |

## Execution Order
```bash
# 1. Core pipeline (requires R 4.x)
Rscript run_complete_analysis.R

# 2. Supplementary analyses (after core pipeline generates corrected_expression.csv)
Rscript run_supplementary_xgboost_hyperparameter_optimization.R
Rscript run_supplementary_batch_aware_ensemble.R
python3 run_supplementary_gnn_analysis.py
```

## Dependencies
- **R ≥ 4.4**: glmnet, nnet, pROC, vegan, corpcor, boot, randomForest, e1071, xgboost, jsonlite
- **Python ≥ 3.8**: torch, torch_geometric, numpy, pandas, scikit-learn, matplotlib

## Note
All supplementary files are generated directly by the 4 analysis scripts in S1_Code.zip. The train/test split is reproduced deterministically via `set.seed(42)` in each script — no external split manifest file is needed.
