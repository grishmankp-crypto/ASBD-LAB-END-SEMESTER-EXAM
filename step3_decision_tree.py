"""
STEP 3B — Decision Tree Classification (CART)
   - Target: Predict whether the FAVORITE wins (Favorite_Won ∈ {0,1})
   - This is the most meaningful, domain-valid classification target because:
     • The class is derived from actual ranking ground truth (not arbitrary label assignment)
     • It is naturally imbalanced 67/33 reflecting real match predictability
     • It has strong predictive features (odds tiers, rank diff, surface, series)
   - Features: Continuous (Log ranks, odds, implied probs) + One-hot encoded categoricals
   - Output: If-Then rules, Confusion Matrix, Accuracy, Precision, Recall, F1, ROC-AUC
"""

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, accuracy_score, f1_score, precision_score, recall_score,
    ConfusionMatrixDisplay
)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'savefig.dpi': 300, 'savefig.bbox': 'tight'
})

BASE_DIR = r"D:\ASBD-LAB"
df_raw   = pd.read_csv(os.path.join(BASE_DIR, "atp_cleaned_transformed.csv"))

print("=" * 72)
print("STEP 3B — DECISION TREE CLASSIFICATION")
print("=" * 72)
print(f"Dataset: {len(df_raw):,} records\n")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  — only retain features with clear causal meaning
# ─────────────────────────────────────────────────────────────────────────────
# EXCLUDED intentionally:
#   Player1_Won  — tautological with Player1_Rank direction + Favorite_Won
#   Outcome_Label, Upset_Label — derived from target, would cause data leakage
#   Tournament, Winner, Player1, Player2 — identifiers, not causal attributes
#   *_Int columns — arbitrary float encodings from source, no semantic meaning
#   Higher_Ranked_Player — directly encodes target construction logic

# Remove 2 corrupted odds records (negative bookmaker margin)
df = df_raw[df_raw['Bookmaker_Margin'] >= 0].copy()
print(f"After removing corrupted margin records: {len(df):,}")

# Continuous numerical features
num_feats = [
    'Log_Rank1',            # ln(1 + P1 rank) — compresses extreme rank skew
    'Log_Rank2',            # ln(1 + P2 rank)
    'Abs_Rank_Diff',        # |Rank1 - Rank2| — captures relative competitive gap
    'Player1_Implied_Prob', # market's estimate of P1 winning
    'Player2_Implied_Prob', # market's estimate of P2 winning
    'Bookmaker_Margin',     # over-round (reflects certainty of market)
    'Implied_Prob_Diff',    # P1_prob - P2_prob (signed direction of market lean)
    'Odds_Ratio',           # P1_Odds / P2_Odds (continuous odds signal)
]

# Categorical features — one-hot encoded
cat_feats = ['Series', 'Court', 'Surface', 'Round', 'Best_of']

TARGET = 'Favorite_Won'  # 1 = higher-ranked player won, 0 = upset

# One-hot encode categoricals
df_enc  = pd.get_dummies(df[num_feats + cat_feats + [TARGET]], columns=cat_feats, drop_first=False)
feature_cols = [c for c in df_enc.columns if c != TARGET]
X = df_enc[feature_cols].values
y = df_enc[TARGET].values

print(f"\nFeature matrix: {X.shape[0]} samples × {X.shape[1]} features")
print(f"Target distribution: Favorite_Won=1 → {y.sum():,} ({y.mean()*100:.2f}%)  |  Upset=0 → {(1-y).sum():,} ({(1-y).mean()*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER SELECTION
# Decision: We select max_depth=6 for the following reasons:
#   1. The dataset has 9 original attributes + one-hot expansions → ~30 final features
#   2. max_depth=6 allows up to 64 leaf nodes — enough to capture complex patterns
#   3. It prevents severe overfitting (a fully-grown tree would memorize training noise)
#   4. We validate this via 5-fold cross-validation on accuracy and AUC
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Hyperparameter Selection via 5-fold Cross-Validation ---")
depths_to_test = [3, 4, 5, 6, 7, 8, 10]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for d in depths_to_test:
    dt = DecisionTreeClassifier(
        criterion='gini', max_depth=d, min_samples_split=50,
        min_samples_leaf=25, random_state=42, class_weight='balanced'
    )
    acc_scores  = cross_val_score(dt, X, y, cv=cv, scoring='accuracy')
    auc_scores  = cross_val_score(dt, X, y, cv=cv, scoring='roc_auc')
    cv_results[d] = {'acc': acc_scores.mean(), 'auc': auc_scores.mean()}
    print(f"  max_depth={d:2d}: Acc={acc_scores.mean():.4f} ± {acc_scores.std():.4f}  |  AUC={auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

# Select best depth (highest AUC)
best_depth = max(cv_results, key=lambda d: cv_results[d]['auc'])
print(f"\nSelected max_depth = {best_depth} (best cross-validated AUC)")

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT & FINAL MODEL
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print(f"\nTrain set: {X_train.shape[0]:,}  |  Test set: {X_test.shape[0]:,}")

clf = DecisionTreeClassifier(
    criterion='gini', max_depth=best_depth,
    min_samples_split=50, min_samples_leaf=25,
    random_state=42, class_weight='balanced'
)
t0 = time.perf_counter()
clf.fit(X_train, y_train)
train_time = time.perf_counter() - t0

y_pred     = clf.predict(X_test)
y_prob     = clf.predict_proba(X_test)[:, 1]

acc    = accuracy_score(y_test, y_pred)
prec   = precision_score(y_test, y_pred)
rec    = recall_score(y_test, y_pred)
f1     = f1_score(y_test, y_pred)
auc    = roc_auc_score(y_test, y_prob)

print(f"\n=== CLASSIFICATION PERFORMANCE METRICS ===")
print(f"  Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision  : {prec:.4f}  (Of predicted Favorite-Wins, {prec*100:.2f}% were correct)")
print(f"  Recall     : {rec:.4f}  (Correctly identified {rec*100:.2f}% of all actual Favorite-Wins)")
print(f"  F1-Score   : {f1:.4f}  (Harmonic mean of Precision & Recall)")
print(f"  ROC-AUC    : {auc:.4f}")
print(f"  Train time : {train_time:.4f}s")

print(f"\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

print(f"\n=== FULL CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=['Upset (0)', 'Favorite_Won (1)']))

# ─────────────────────────────────────────────────────────────────────────────
# IF-THEN RULE EXTRACTION  (from tree structure)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== IF-THEN DECISION RULES (from CART tree) ===\n")
tree_rules = export_text(clf, feature_names=feature_cols, max_depth=best_depth, show_weights=True)
print(tree_rules)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
importances = clf.feature_importances_
feat_imp    = pd.DataFrame({'feature': feature_cols, 'importance': importances})
feat_imp    = feat_imp.sort_values('importance', ascending=False).head(15)
print("\n=== TOP 15 FEATURE IMPORTANCES (Gini Impurity Reduction) ===")
for _, row in feat_imp.iterrows():
    bar = "█" * int(row['importance'] * 200)
    print(f"  {row['feature']:<40} {row['importance']:.4f}  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
# P11: ROC Curve
print("\nGenerating Decision Tree Plots...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, color='#2c7bb6', linewidth=2.0, label=f'ROC Curve (AUC = {auc:.4f})')
axes[0].plot([0, 1], [0, 1], color='#777777', linestyle='--', linewidth=1.2, label='Random Classifier (AUC = 0.5)')
axes[0].fill_between(fpr, tpr, alpha=0.1, color='#2c7bb6')
axes[0].set_xlabel("False Positive Rate (1 − Specificity)", fontsize=11)
axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
axes[0].set_title(f"Figure 11: ROC Curve — CART Decision Tree\n(Predicting Higher-Ranked Favorite Win)", fontsize=11, fontweight='bold')
axes[0].legend(loc='lower right', frameon=True)
axes[0].grid(True, linestyle=':', alpha=0.6)

# Right: Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Upset (0)', 'Fav. Won (1)'])
disp.plot(ax=axes[1], colorbar=False, cmap='Blues')
axes[1].set_title(f"Figure 12: Confusion Matrix — CART Decision Tree\n(Test Set, N={len(y_test):,})", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot11_12_decision_tree_roc_cm.pdf"), format='pdf')
plt.close()
print(f"  -> Saved: plot11_12_decision_tree_roc_cm.pdf")

# P13: Feature Importance
fig, ax = plt.subplots(figsize=(9, 6))
colors_fi = ['#1b4965' if i == 0 else '#5fa8d3' for i in range(len(feat_imp))]
ax.barh(feat_imp['feature'][::-1], feat_imp['importance'][::-1],
        color=colors_fi[::-1], edgecolor='#222222', linewidth=0.7)
ax.set_xlabel("Gini Impurity Reduction (Feature Importance Score)", fontsize=11)
ax.set_title("Figure 13: CART Decision Tree — Top 15 Feature Importances\n(Gini-Based Importance, Predicting Favorite_Won)", fontsize=11, fontweight='bold')
ax.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot13_decision_tree_feature_importance.pdf"), format='pdf')
plt.close()
print(f"  -> Saved: plot13_decision_tree_feature_importance.pdf")

# P14: Pruned tree visualization
fig, ax = plt.subplots(figsize=(22, 9))
plot_tree(clf, feature_names=feature_cols, class_names=['Upset', 'Fav_Won'],
          filled=True, rounded=True, impurity=True, proportion=False,
          fontsize=7, ax=ax, max_depth=4)
ax.set_title("Figure 14: CART Decision Tree Structure (Rendered to max_depth=4)\n(Gini split criterion, class_weight='balanced')", fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot14_decision_tree_structure.pdf"), format='pdf')
plt.close()
print(f"  -> Saved: plot14_decision_tree_structure.pdf")

# Save all metrics to file
metrics_txt = os.path.join(BASE_DIR, "step3b_dt_results.txt")
with open(metrics_txt, "w", encoding="utf-8") as f:
    f.write("=" * 72 + "\n")
    f.write("STEP 3B — DECISION TREE CLASSIFICATION RESULTS\n")
    f.write(f"ATP Tennis Dataset  |  N = {len(df):,}  |  2012–2017\n")
    f.write("=" * 72 + "\n\n")
    f.write("TARGET: Favorite_Won (1 = higher-ranked player won, 0 = upset)\n")
    f.write("REASON: Derived from ranking ground truth; not arbitrary; naturally\n")
    f.write("        imbalanced 67/33 mirroring real-world match predictability.\n\n")
    f.write("HYPERPARAMETERS:\n")
    f.write(f"  criterion        : gini (Gini Impurity)\n")
    f.write(f"  max_depth        : {best_depth} (selected by 5-fold CV AUC maximization)\n")
    f.write(f"  min_samples_split: 50\n")
    f.write(f"  min_samples_leaf : 25\n")
    f.write(f"  class_weight     : balanced (upweights minority Upset class)\n\n")
    f.write("PERFORMANCE METRICS:\n")
    f.write(f"  Accuracy  : {acc:.4f}\n")
    f.write(f"  Precision : {prec:.4f}\n")
    f.write(f"  Recall    : {rec:.4f}\n")
    f.write(f"  F1-Score  : {f1:.4f}\n")
    f.write(f"  ROC-AUC   : {auc:.4f}\n\n")
    f.write("CONFUSION MATRIX:\n")
    f.write(f"  TN={cm[0,0]}  FP={cm[0,1]}\n")
    f.write(f"  FN={cm[1,0]}  TP={cm[1,1]}\n\n")
    f.write("CLASSIFICATION REPORT:\n")
    f.write(classification_report(y_test, y_pred, target_names=['Upset (0)', 'Favorite_Won (1)']))
    f.write("\n\nIF-THEN DECISION RULES (CART tree text):\n")
    f.write(tree_rules)

print(f"\nFull results saved → {metrics_txt}")
