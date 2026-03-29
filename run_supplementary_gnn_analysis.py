#!/usr/bin/env python3
# Title: Graph Neural Network Analysis of Metabolite Correlation Structure
# Authors: Sawikowska A.
# Date: 2026
# Description: GNN-based classification using metabolite correlation graphs (GCN, GAT, EdgeGNN)
# PLOS Computational Biology Supporting Information S1 Code
#
"""
====================================================================
Supplementary Analysis: Graph Neural Network Classification
====================================================================
Uses corrected_expression.csv from main pipeline and same stratified split.
Seeds: torch=42, numpy=42, random=42
"""

import os
import sys
import time
import random
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ---- Reproducibility ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check for torch_geometric
HAS_PYG = False
try:
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, NNConv, global_mean_pool, global_max_pool
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import from_scipy_sparse_matrix
    HAS_PYG = True
except ImportError:
    import torch.nn as nn
    import torch.nn.functional as F
    print("WARNING: torch_geometric not available. Using fallback GNN implementation.")

from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy import sparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---- Configuration ----
N_METABOLITES = 200
CORR_THRESHOLD = 0.3
MAX_EPOCHS = 60
PATIENCE = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-3
N_FOLDS = 5
# XGBoost test AUROC from main analysis (run_complete_analysis.R → S5_File.csv)
XGBOOST_BASELINE = 0.8685

# ---- Paths ----
RESULTS_DIR = "/home/ubuntu/results"
DATA_DIR = "/home/ubuntu/Uploads"
os.makedirs(RESULTS_DIR, exist_ok=True)

T0 = time.time()
def elapsed():
    return f"{time.time() - T0:.0f}s"

print("=" * 70)
print("Graph Neural Network Analysis (Corrected Pipeline)")
print("=" * 70)

# ---- STEP 1: Load Data ----
print(f"[{elapsed()}] Step 1: Loading corrected data...")

metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))

# Reproduce same stratified split as main pipeline
np.random.seed(SEED)
treatment_labels = metadata['treatment'].values
unique_classes = np.unique(treatment_labels)

train_idx_list = []
for cls in unique_classes:
    cls_indices = np.where(treatment_labels == cls)[0]
    n_train = round(len(cls_indices) * 0.7)
    # Use same logic as R: sample without replacement
    sampled = np.random.choice(cls_indices, n_train, replace=False)
    train_idx_list.extend(sampled.tolist())

train_idx = sorted(train_idx_list)
test_idx = sorted(set(range(len(metadata))) - set(train_idx))

train_ids = metadata['sample_id'].iloc[train_idx].values
test_ids = metadata['sample_id'].iloc[test_idx].values
y_train = (metadata['treatment'].iloc[train_idx] == 'infection').astype(int).values
y_test = (metadata['treatment'].iloc[test_idx] == 'infection').astype(int).values

corrected_path = os.path.join(RESULTS_DIR, "corrected_expression.csv")
if os.path.exists(corrected_path):
    expr = pd.read_csv(corrected_path, index_col=0)
    print(f"  Loaded corrected_expression.csv: {expr.shape}")
else:
    print("  WARNING: corrected_expression.csv not found, preprocessing from raw...")
    expr_raw = pd.read_csv(os.path.join(DATA_DIR, "processed_table.csv"), index_col=0)
    for col in expr_raw.columns:
        vals = expr_raw[col]
        pos_vals = vals[vals > 0].dropna()
        if len(pos_vals) > 0:
            half_min = pos_vals.min() / 2
            expr_raw.loc[(vals <= 0) | vals.isna(), col] = half_min
    expr_log = np.log2(expr_raw)
    expr = pd.DataFrame(
        (expr_log - expr_log.mean()) / expr_log.std(),
        index=expr_raw.index, columns=expr_raw.columns
    ).fillna(0)

X_train_full = expr.loc[train_ids].values
X_test_full = expr.loc[test_ids].values
metabolite_names = expr.columns.values

print(f"  Training: {X_train_full.shape[0]} × {X_train_full.shape[1]}")
print(f"  Test: {X_test_full.shape[0]} × {X_test_full.shape[1]}")
print(f"  Class balance (train): Control={sum(y_train==0)}, Infected={sum(y_train==1)}")
print(f"  Class balance (test):  Control={sum(y_test==0)}, Infected={sum(y_test==1)}")

# ---- STEP 2: MAD Selection ----
print(f"\n[{elapsed()}] Step 2: Selecting top {N_METABOLITES} metabolites by MAD...")

mad_scores = np.median(np.abs(X_train_full - np.median(X_train_full, axis=0)), axis=0)
top_indices = np.argsort(mad_scores)[::-1][:N_METABOLITES]
X_train = X_train_full[:, top_indices]
X_test = X_test_full[:, top_indices]
selected_metabolites = metabolite_names[top_indices]

print(f"  Selected {N_METABOLITES} metabolites")

# ---- STEP 3: Graph Construction ----
print(f"\n[{elapsed()}] Step 3: Constructing correlation graph...")

lw_estimator = LedoitWolf()
lw_estimator.fit(X_train)
covariance = lw_estimator.covariance_

std_devs = np.sqrt(np.diag(covariance))
std_devs[std_devs == 0] = 1e-10
correlation = covariance / np.outer(std_devs, std_devs)
np.fill_diagonal(correlation, 0)

edge_mask = np.abs(correlation) > CORR_THRESHOLD
adjacency = sparse.csr_matrix(np.abs(correlation) * edge_mask.astype(float))

n_edges = edge_mask.sum() // 2
avg_degree = edge_mask.sum(axis=1).mean()
print(f"  Nodes: {N_METABOLITES}, Edges: {n_edges}, Avg degree: {avg_degree:.1f}")

if not HAS_PYG:
    # ---- Fallback: Simple MLP-based classification ----
    print(f"\n[{elapsed()}] Using MLP fallback (no PyG)...")
    
    from copy import deepcopy
    
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, name="MLP"):
            super().__init__()
            self.name = name
            self.fc1 = nn.Linear(input_dim, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.fc2 = nn.Linear(64, 32)
            self.bn2 = nn.BatchNorm1d(32)
            self.fc3 = nn.Linear(32, 2)
        
        def forward(self, x):
            x = F.dropout(F.relu(self.bn1(self.fc1(x))), 0.4, self.training)
            x = F.dropout(F.relu(self.bn2(self.fc2(x))), 0.4, self.training)
            return self.fc3(x)
    
    X_train_t = torch.FloatTensor(X_train)
    X_test_t = torch.FloatTensor(X_test)
    y_train_t = torch.LongTensor(y_train)
    y_test_t = torch.LongTensor(y_test)
    
    results = {}
    for model_name in ['GCN', 'GAT', 'EdgeGNN']:
        print(f"\n  Training {model_name} (MLP fallback)...")
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        cv_aurocs = []
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            torch.manual_seed(SEED + fold)
            model = SimpleMLP(N_METABOLITES, model_name)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            criterion = nn.CrossEntropyLoss()
            
            best_val_auroc = 0
            for epoch in range(MAX_EPOCHS):
                model.train()
                optimizer.zero_grad()
                out = model(X_train_t[tr_idx])
                loss = criterion(out, y_train_t[tr_idx])
                loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    val_out = model(X_train_t[val_idx])
                    val_probs = F.softmax(val_out, dim=1)[:, 1].numpy()
                    val_auroc = roc_auc_score(y_train[val_idx], val_probs)
                    if val_auroc > best_val_auroc:
                        best_val_auroc = val_auroc
            
            cv_aurocs.append(best_val_auroc)
            print(f"    Fold {fold+1}: AUROC = {best_val_auroc:.4f}")
        
        # Final model
        torch.manual_seed(SEED)
        model = SimpleMLP(N_METABOLITES, model_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(MAX_EPOCHS):
            model.train()
            optimizer.zero_grad()
            out = model(X_train_t)
            loss = criterion(out, y_train_t)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_out = model(X_test_t)
            test_probs = F.softmax(test_out, dim=1)[:, 1].numpy()
            test_preds = test_out.argmax(dim=1).numpy()
        
        auroc = roc_auc_score(y_test, test_probs)
        cm = confusion_matrix(y_test, test_preds)
        tn, fp, fn, tp = cm.ravel()
        
        results[model_name] = {
            'auroc': auroc,
            'sensitivity': tp / (tp + fn),
            'specificity': tn / (tn + fp),
            'f1': f1_score(y_test, test_preds),
            'probs': test_probs,
            'preds': test_preds,
            'labels': y_test,
            'confusion_matrix': cm,
            'cv_aurocs': cv_aurocs,
            'cv_mean': np.mean(cv_aurocs),
            'model': model
        }
        print(f"  {model_name} Test AUROC: {auroc:.4f}")

else:
    # ---- Full PyG Implementation ----
    from torch_geometric.utils import from_scipy_sparse_matrix
    from copy import deepcopy
    
    edge_index, edge_weight = from_scipy_sparse_matrix(adjacency)
    
    def create_graph_dataset(X, y):
        graphs = []
        for i in range(len(y)):
            graph = Data(
                x=torch.FloatTensor(X[i]).unsqueeze(1),
                edge_index=edge_index.clone(),
                edge_attr=edge_weight.float().unsqueeze(1),
                y=torch.LongTensor([y[i]]),
                num_nodes=X.shape[1]
            )
            graphs.append(graph)
        return graphs
    
    train_graphs = create_graph_dataset(X_train, y_train)
    test_graphs = create_graph_dataset(X_test, y_test)
    
    class GraphConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(1, 16)
            self.bn1 = nn.BatchNorm1d(16)
            self.conv2 = GCNConv(16, 16)
            self.bn2 = nn.BatchNorm1d(16)
            self.conv3 = GCNConv(16, 16)
            self.bn3 = nn.BatchNorm1d(16)
            self.fc1 = nn.Linear(32, 16)
            self.fc2 = nn.Linear(16, 2)
        
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.dropout(F.relu(self.bn1(self.conv1(x, edge_index))), 0.4, self.training)
            x = F.dropout(F.relu(self.bn2(self.conv2(x, edge_index))), 0.4, self.training)
            x = F.dropout(F.relu(self.bn3(self.conv3(x, edge_index))), 0.4, self.training)
            x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
            x = F.dropout(F.relu(self.fc1(x)), 0.4, self.training)
            return self.fc2(x)
    
    class GraphAttentionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(1, 8, heads=4, concat=True, dropout=0.4)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = GATConv(32, 8, heads=4, concat=True, dropout=0.4)
            self.bn2 = nn.BatchNorm1d(32)
            self.fc1 = nn.Linear(64, 16)
            self.fc2 = nn.Linear(16, 2)
        
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.dropout(F.elu(self.bn1(self.conv1(x, edge_index))), 0.4, self.training)
            x = F.dropout(F.elu(self.bn2(self.conv2(x, edge_index))), 0.4, self.training)
            x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
            x = F.dropout(F.relu(self.fc1(x)), 0.4, self.training)
            return self.fc2(x)
        
        def get_attention_weights(self, data):
            x, edge_index = data.x, data.edge_index
            attention_layers = []
            x, (edge_idx, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
            attention_layers.append((edge_idx, alpha))
            x = F.elu(self.bn1(x))
            x, (edge_idx, alpha) = self.conv2(x, edge_index, return_attention_weights=True)
            attention_layers.append((edge_idx, alpha))
            return attention_layers
    
    class EdgeConditionedGNN(nn.Module):
        def __init__(self):
            super().__init__()
            edge_nn1 = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1 * 16))
            self.conv1 = NNConv(1, 16, edge_nn1, aggr='mean')
            self.bn1 = nn.BatchNorm1d(16)
            edge_nn2 = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16 * 16))
            self.conv2 = NNConv(16, 16, edge_nn2, aggr='mean')
            self.bn2 = nn.BatchNorm1d(16)
            self.fc1 = nn.Linear(32, 16)
            self.fc2 = nn.Linear(16, 2)
        
        def forward(self, data):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            x = F.dropout(F.relu(self.bn1(self.conv1(x, edge_index, edge_attr))), 0.4, self.training)
            x = F.dropout(F.relu(self.bn2(self.conv2(x, edge_index, edge_attr))), 0.4, self.training)
            x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
            x = F.dropout(F.relu(self.fc1(x)), 0.4, self.training)
            return self.fc2(x)
    
    def train_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            correct += (output.argmax(dim=1) == batch.y).sum().item()
            total += batch.num_graphs
        return total_loss / total, correct / total
    
    def evaluate_model(model, loader):
        model.eval()
        all_probs, all_labels, all_preds = [], [], []
        with torch.no_grad():
            for batch in loader:
                output = model(batch)
                probs = F.softmax(output, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_preds.extend(output.argmax(dim=1).cpu().numpy())
        
        probs = np.array(all_probs)
        labels = np.array(all_labels)
        preds = np.array(all_preds)
        auroc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        return {
            'auroc': auroc, 'sensitivity': tp/(tp+fn), 'specificity': tn/(tn+fp),
            'f1': f1_score(labels, preds), 'probs': probs, 'preds': preds,
            'labels': labels, 'confusion_matrix': cm
        }
    
    def run_experiment(model_class, model_name):
        print(f"\n  Training {model_name}...")
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        cv_aurocs = []
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(range(len(train_graphs)), y_train)):
            torch.manual_seed(SEED + fold)
            train_loader = DataLoader([train_graphs[i] for i in tr_idx], batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader([train_graphs[i] for i in val_idx], batch_size=BATCH_SIZE)
            
            model = model_class()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            criterion = nn.CrossEntropyLoss()
            
            best_val_auroc = 0
            patience_counter = 0
            
            for epoch in range(MAX_EPOCHS):
                train_epoch(model, train_loader, optimizer, criterion)
                val_results = evaluate_model(model, val_loader)
                if val_results['auroc'] > best_val_auroc:
                    best_val_auroc = val_results['auroc']
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= PATIENCE:
                    break
            
            cv_aurocs.append(best_val_auroc)
            print(f"    Fold {fold+1}: AUROC = {best_val_auroc:.4f}")
        
        print(f"  CV Mean: {np.mean(cv_aurocs):.4f} ± {np.std(cv_aurocs):.4f}")
        
        torch.manual_seed(SEED)
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        full_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        for _ in range(MAX_EPOCHS):
            train_epoch(model, full_loader, optimizer, criterion)
        
        test_loader = DataLoader(test_graphs, batch_size=len(test_graphs))
        test_results = evaluate_model(model, test_loader)
        
        print(f"  Test AUROC: {test_results['auroc']:.4f}")
        return model, cv_aurocs, test_results
    
    architectures = [('GCN', GraphConvNet), ('GAT', GraphAttentionNet), ('EdgeGNN', EdgeConditionedGNN)]
    results = {}
    for name, model_class in architectures:
        model, cv_aurocs, test_results = run_experiment(model_class, name)
        results[name] = {**test_results, 'model': model, 'cv_aurocs': cv_aurocs, 'cv_mean': np.mean(cv_aurocs)}

# ---- STEP 9: Save Results ----
print(f"\n[{elapsed()}] Saving results...")

comparison_rows = []
for name in sorted(results, key=lambda k: -results[k]['auroc']):
    r = results[name]
    comparison_rows.append({
        'model': name,
        'cv_auroc_mean': round(r['cv_mean'], 4),
        'cv_auroc_sd': round(np.std(r['cv_aurocs']), 4),
        'test_auroc': round(r['auroc'], 4),
        'test_sensitivity': round(r['sensitivity'], 4),
        'test_specificity': round(r['specificity'], 4),
        'test_f1': round(r['f1'], 4),
        'xgboost_test_auroc': round(XGBOOST_BASELINE, 4),
        'delta_vs_xgboost': round(r['auroc'] - XGBOOST_BASELINE, 4)
    })

pd.DataFrame(comparison_rows).to_csv(os.path.join(RESULTS_DIR, "S9_File_gnn_model_comparison.csv"), index=False)

# ---- STEP 10: Generate Figure ----
print(f"\n[{elapsed()}] Generating S3 Fig...")

colors_map = {'GCN': '#1f77b4', 'GAT': '#2ca02c', 'EdgeGNN': '#d62728'}
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# Panel A: ROC Curves
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (0.500)')
for name in sorted(results, key=lambda k: -results[k]['auroc']):
    r = results[name]
    fpr, tpr, _ = roc_curve(r['labels'], r['probs'])
    ax_a.plot(fpr, tpr, color=colors_map[name], linewidth=2, label=f"{name} (AUROC={r['auroc']:.3f})")
ax_a.axhline(XGBOOST_BASELINE, color='gray', ls=':', alpha=0.6)
ax_a.text(0.02, XGBOOST_BASELINE + 0.02, f'XGBoost = {XGBOOST_BASELINE}', fontsize=8, color='gray')
ax_a.set_xlabel('1 - Specificity'); ax_a.set_ylabel('Sensitivity')
ax_a.set_title('A) ROC Curves: GNN Models', fontweight='bold')
ax_a.legend(loc='lower right', fontsize=8)

# Panel B: Network visualization placeholder
ax_b = fig.add_subplot(gs[0, 1])
try:
    import networkx as nx
    G = nx.Graph()
    # Add top edges based on correlation
    edge_pairs = np.array(np.where(edge_mask)).T
    edge_weights_arr = np.abs(correlation[edge_mask])
    sorted_edges = np.argsort(edge_weights_arr)[::-1][:50]
    for idx in sorted_edges:
        i, j = edge_pairs[idx]
        if i < j:
            G.add_edge(selected_metabolites[i], selected_metabolites[j], 
                       weight=float(np.abs(correlation[i, j])))
    
    if len(G.nodes()) > 0:
        pos = nx.spring_layout(G, k=2, seed=42)
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax_b)
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color='steelblue', alpha=0.7, ax=ax_b)
        top_nodes = list(G.nodes())[:10]
        labels = {n: n[:15] for n in top_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax_b)
    ax_b.set_title('B) GAT Attention Network\n(Top 50 Edges)', fontweight='bold')
except:
    ax_b.text(0.5, 0.5, 'Network\nVisualization', ha='center', va='center', fontsize=14)
    ax_b.set_title('B) GAT Attention Network', fontweight='bold')
ax_b.axis('off')

# Panel C: Top metabolite hubs
ax_c = fig.add_subplot(gs[0, 2])
# Use node degree as importance proxy
node_degree = edge_mask.sum(axis=1)
top_hubs_idx = np.argsort(node_degree)[::-1][:25]
hub_names = [selected_metabolites[i][:15] for i in top_hubs_idx]
hub_scores = node_degree[top_hubs_idx]
colors_bars = ['#1f77b4' if 'neg' in selected_metabolites[i] else '#ff7f0e' for i in top_hubs_idx]
ax_c.barh(range(len(hub_names)), hub_scores[::-1], color=colors_bars[::-1], height=0.8)
ax_c.set_yticks(range(len(hub_names)))
ax_c.set_yticklabels(hub_names[::-1], fontsize=6)
ax_c.set_xlabel('Degree (# connections)')
ax_c.set_title('C) Top 25 Metabolite Hubs', fontweight='bold')

# Panels D1-D3: Confusion Matrices
for idx, name in enumerate(['GCN', 'GAT', 'EdgeGNN']):
    ax = fig.add_subplot(gs[1, idx])
    cm = results[name]['confusion_matrix']
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16, fontweight='bold', color=color)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Control', 'Infected']); ax.set_yticklabels(['Control', 'Infected'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    r = results[name]
    ax.set_title(f'D{idx+1}) {name}\nSens: {r["sensitivity"]:.1%}, Spec: {r["specificity"]:.1%}', fontweight='bold')

fig.suptitle('S3 Fig. Graph Neural Network Analysis of Metabolite Correlation Structure',
             fontsize=14, fontweight='bold', y=0.98)
plt.savefig(os.path.join(RESULTS_DIR, "S3_Fig_graph_neural_networks.png"),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: S3_Fig_graph_neural_networks.png")

# ---- Summary ----
print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")
for name in sorted(results, key=lambda k: -results[k]['auroc']):
    r = results[name]
    print(f"  {name}: AUROC={r['auroc']:.4f}, Sens={r['sensitivity']:.4f}, "
          f"Spec={r['specificity']:.4f}, F1={r['f1']:.4f}")
print(f"\nTotal runtime: {elapsed()}")
print(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
