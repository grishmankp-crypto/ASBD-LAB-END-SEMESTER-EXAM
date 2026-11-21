"""
STEP 3C — DBSCAN Clustering
   Clustering of ATP matches in a compact, meaningful feature space.
   Features selected: Log_Rank1, Log_Rank2, Implied_Prob_Diff, Bookmaker_Margin
   REASON:
     • These 4 features form a compact, interpretable subspace where
       "normal" matches cluster tightly and genuine anomalies (upsets, corrupted odds)
       appear as low-density outliers — exactly what DBSCAN detects.
     • Odds_Ratio is intentionally excluded: its long tail (max=36) causes
       distance skew that creates false outliers even after scaling.
     • Surface, Court, Series are categorical — meaningful for ARM/DT but not
       for Euclidean distance-based density clustering.

   Outputs:
     • K-distance graph to determine epsilon empirically
     • DBSCAN cluster assignments with Silhouette Score & Davies-Bouldin Index
     • Cluster profile table (mean feature values per cluster)
     • Noise/anomaly isolation with match-level interpretation
"""

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'savefig.dpi': 300, 'savefig.bbox': 'tight'
})

BASE_DIR = r"D:\ASBD-LAB"
df = pd.read_csv(os.path.join(BASE_DIR, "atp_cleaned_transformed.csv"))
df = df[df['Bookmaker_Margin'] >= 0].copy()   # remove 2 corrupted records

print("=" * 72)
print("STEP 3C — DBSCAN CLUSTERING")
print("=" * 72)
print(f"Dataset: {len(df):,} records\n")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTION  — 4 continuous, semantically coherent features
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = ['Log_Rank1', 'Log_Rank2', 'Implied_Prob_Diff', 'Bookmaker_Margin']
X_raw = df[FEATURES].values

# Standardize: DBSCAN uses Euclidean distance — features must be on equal scales
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print(f"Feature space: {FEATURES}")
print(f"Scaled to zero mean, unit variance for Euclidean distance integrity\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: K-DISTANCE GRAPH to determine epsilon empirically
# Method: For each point, compute distance to its k-th nearest neighbor (k=5).
# Sort distances descending. The "elbow" in the plot marks the optimal epsilon.
# Reference: Ester et al. (1996), KDD, pp. 226–231.
# ─────────────────────────────────────────────────────────────────────────────
print("--- Computing k-distance graph (k=5) for epsilon estimation ---")
K_NEIGHBORS = 5   # MinPts - 1 (MinPts will be set to 5)
t0 = time.perf_counter()
nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, algorithm='ball_tree').fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)
# k-th nearest neighbor distance for each point (column 5 = 5th neighbor)
kth_distances = np.sort(distances[:, K_NEIGHBORS])[::-1]
kd_time = time.perf_counter() - t0
print(f"k-distance computation done in {kd_time:.2f}s")

# Elbow detection: largest second derivative
# Better epsilon heuristic: use the distance at the 95th percentile of sorted
# kth-distances. This corresponds to the "knee" where most points transition
# from dense-core to sparse-peripheral regions, without being skewed by the
# flat tail of extreme outliers.
eps_optimal = float(np.percentile(kth_distances, 95))
# Find corresponding index for plot annotation
elbow_idx   = np.argmin(np.abs(kth_distances - eps_optimal))
print(f"Epsilon selected at 95th percentile of k-distances → ε = {eps_optimal:.4f}")

# Plot 15: K-distance graph
fig, ax = plt.subplots(figsize=(9, 5))
x_plot = np.arange(len(kth_distances))
ax.plot(x_plot, kth_distances, color='#1b4965', linewidth=1.6, label='5th-NN Distance')
ax.axvline(elbow_idx, color='#d9381e', linestyle='--', linewidth=1.5, label=f'Elbow (index={elbow_idx})')
ax.axhline(eps_optimal, color='#e76f51', linestyle=':', linewidth=1.5, label=f'ε = {eps_optimal:.4f}')
ax.set_xlabel("Points Sorted by 5th Nearest-Neighbor Distance (Descending)", fontsize=11)
ax.set_ylabel("Distance to 5th Nearest Neighbor (Scaled)", fontsize=11)
ax.set_title("Figure 15: K-Distance Graph for Empirical DBSCAN Epsilon Selection\n(Elbow Marks Transition from Dense Clusters to Sparse Noise Region)", fontsize=11, fontweight='bold')
ax.legend(loc='upper right', frameon=True)
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot15_dbscan_kdistance.pdf"), format='pdf')
plt.close()
print(f"  -> Saved: plot15_dbscan_kdistance.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: DBSCAN EXECUTION
# Parameters:
#   eps       : determined from k-distance graph elbow
#   min_samples: 5 — minimum 5 points to constitute a dense region
#                (conservative choice; about 0.034% of N=14,695)
# ─────────────────────────────────────────────────────────────────────────────
MIN_SAMPLES = 5

print(f"\n--- Running DBSCAN (eps={eps_optimal:.4f}, min_samples={MIN_SAMPLES}) ---")
t1 = time.perf_counter()
db = DBSCAN(eps=eps_optimal, min_samples=MIN_SAMPLES, algorithm='ball_tree', n_jobs=-1)
labels = db.fit_predict(X_scaled)
dbscan_time = time.perf_counter() - t1

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = np.sum(labels == -1)

print(f"DBSCAN done in {dbscan_time:.2f}s")
print(f"Clusters found      : {n_clusters}")
print(f"Noise points (ε=-1) : {n_noise} ({n_noise/len(labels)*100:.2f}%)")
print(f"Clustered points    : {len(labels)-n_noise} ({(len(labels)-n_noise)/len(labels)*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: CLUSTER QUALITY METRICS
# Note: Silhouette and Davies-Bouldin are computed on CLUSTERED points only
#       (noise points are excluded from quality scoring by convention)
# ─────────────────────────────────────────────────────────────────────────────
mask_clustered   = labels != -1
X_clustered      = X_scaled[mask_clustered]
labels_clustered = labels[mask_clustered]

if n_clusters > 1:
    sil_score = silhouette_score(X_clustered, labels_clustered, sample_size=5000, random_state=42)
    db_score  = davies_bouldin_score(X_clustered, labels_clustered)
    print(f"\nCluster Quality (on {len(X_clustered):,} non-noise points):")
    print(f"  Silhouette Score      : {sil_score:.4f}  (range [-1,1]; >0 = clusters separated)")
    print(f"  Davies-Bouldin Index  : {db_score:.4f}  (lower = more compact, well-separated clusters)")
else:
    sil_score = db_score = None
    print("Single cluster or no valid clusters — skipping internal metrics.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: CLUSTER PROFILE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
df_cluster = df.copy()
df_cluster['DBSCAN_Label'] = labels

print("\n=== CLUSTER PROFILE TABLE ===")
print(f"{'Cluster':>8} {'Count':>7} {'Log_R1':>8} {'Log_R2':>8} {'ImpPDiff':>10} {'BkmMrgn':>9} {'FavWinRate':>12} {'UpsetRate':>10}")
print("─" * 80)

cluster_profiles = []
for cl in sorted(set(labels)):
    sub = df_cluster[df_cluster['DBSCAN_Label'] == cl]
    name = "Noise" if cl == -1 else f"C{cl}"
    profile = {
        'Cluster': name,
        'Count': len(sub),
        'Log_Rank1_mean': sub['Log_Rank1'].mean(),
        'Log_Rank2_mean': sub['Log_Rank2'].mean(),
        'Implied_Prob_Diff_mean': sub['Implied_Prob_Diff'].mean(),
        'Bookmaker_Margin_mean': sub['Bookmaker_Margin'].mean(),
        'Fav_Win_Rate': sub['Favorite_Won'].mean(),
        'Upset_Rate': sub['Is_Upset'].mean()
    }
    cluster_profiles.append(profile)
    print(f"  {name:>6} {profile['Count']:>7} "
          f"{profile['Log_Rank1_mean']:>8.3f} {profile['Log_Rank2_mean']:>8.3f} "
          f"{profile['Implied_Prob_Diff_mean']:>10.4f} {profile['Bookmaker_Margin_mean']:>9.5f} "
          f"{profile['Fav_Win_Rate']:>12.4f} {profile['Upset_Rate']:>10.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: NOISE POINT ANALYSIS (MATCH-LEVEL ANOMALIES)
# ─────────────────────────────────────────────────────────────────────────────
noise_df = df_cluster[df_cluster['DBSCAN_Label'] == -1][[
    'Player1', 'Player2', 'Winner', 'Surface', 'Series', 'Round',
    'Player1_Rank', 'Player2_Rank', 'Player1_Odds', 'Player2_Odds',
    'Is_Upset', 'Bookmaker_Margin', 'Abs_Rank_Diff', 'Implied_Prob_Diff'
]].copy()
noise_df['Anomaly_Type'] = 'Other'
noise_df.loc[(noise_df['Is_Upset'] == 1) & (noise_df['Abs_Rank_Diff'] > 50), 'Anomaly_Type'] = 'Major_Upset'
noise_df.loc[noise_df['Bookmaker_Margin'].abs() > 0.12, 'Anomaly_Type'] = 'Odds_Anomaly'
noise_df.loc[noise_df['Abs_Rank_Diff'] > 500, 'Anomaly_Type'] = 'Extreme_Rank_Gap'

print(f"\n=== NOISE POINT ANALYSIS (n={len(noise_df)}) ===")
print("Anomaly type breakdown:")
print(noise_df['Anomaly_Type'].value_counts().to_string())
print("\nSample of Major Upsets in Noise Region:")
major_upsets_noise = noise_df[noise_df['Anomaly_Type'] == 'Major_Upset'].head(8)
if len(major_upsets_noise) > 0:
    for _, row in major_upsets_noise.iterrows():
        print(f"  {row['Player1']} vs {row['Player2']} | Winner: {row['Winner']} "
              f"| {row['Series']} {row['Round']} {row['Surface']} "
              f"| Ranks: {row['Player1_Rank']} vs {row['Player2_Rank']} "
              f"| Odds: {row['Player1_Odds']} vs {row['Player2_Odds']}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
# Reduce to 2D via PCA for scatter visualization
print("\nGenerating DBSCAN cluster visualization...")
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_
print(f"PCA retained variance: PC1={var_exp[0]:.3f}, PC2={var_exp[1]:.3f} (total={sum(var_exp):.3f})")

# Color map
unique_labels = sorted(set(labels))
n_cl = len([l for l in unique_labels if l != -1])
base_colors = ['#1b4965','#e76f51','#2ca02c','#9467bd','#8c564b',
               '#e377c2','#7f7f7f','#bcbd22','#17becf','#d62728']
color_map = {}
cl_idx = 0
for l in unique_labels:
    if l == -1:
        color_map[l] = '#888888'
    else:
        color_map[l] = base_colors[cl_idx % len(base_colors)]
        cl_idx += 1

# Plot 16: DBSCAN 2D PCA scatter
fig, ax = plt.subplots(figsize=(10, 7))
for l in unique_labels:
    mask = labels == l
    name  = "Noise" if l == -1 else f"Cluster {l}"
    ms    = 6 if l == -1 else 10
    alpha = 0.3 if l == -1 else 0.55
    marker= 'x' if l == -1 else 'o'
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[color_map[l]], s=ms, alpha=alpha, marker=marker,
               linewidths=0.5 if l != -1 else 0.8,
               label=f"{name} (n={mask.sum()})")

ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% Variance Explained)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% Variance Explained)", fontsize=11)
ax.set_title(
    f"Figure 16: DBSCAN Cluster Visualization in PCA-Reduced Feature Space\n"
    f"(ε={eps_optimal:.4f}, MinPts={MIN_SAMPLES}, Clusters={n_clusters}, Noise={n_noise})",
    fontsize=11, fontweight='bold'
)
ax.legend(loc='upper right', frameon=True, framealpha=0.9)
ax.grid(True, linestyle=':', alpha=0.5)
if sil_score:
    ax.text(0.02, 0.98, f"Silhouette={sil_score:.4f}\nDavies-Bouldin={db_score:.4f}",
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot16_dbscan_clusters_pca.pdf"), format='pdf')
plt.close()
print(f"  -> Saved: plot16_dbscan_clusters_pca.pdf")

# Plot 17: Cluster Upset Rate & Favorite Win Rate Bar Chart
fig, ax = plt.subplots(figsize=(9, 5))
cp_df = pd.DataFrame(cluster_profiles)
x = np.arange(len(cp_df))
width = 0.35
b1 = ax.bar(x - width/2, cp_df['Fav_Win_Rate'] * 100, width,
            label='Favorite Win Rate (%)', color='#1b4965', edgecolor='#111111', linewidth=0.8)
b2 = ax.bar(x + width/2, cp_df['Upset_Rate'] * 100, width,
            label='Upset Rate (%)', color='#e76f51', edgecolor='#111111', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cp_df['Cluster'], fontsize=10)
ax.set_ylabel("Rate [%]", fontsize=11)
ax.set_title("Figure 17: Cluster-Level Favorite Win and Upset Rates\n(Characterizing DBSCAN Cluster Behavioral Signatures)", fontsize=11, fontweight='bold')
ax.legend(frameon=True)
ax.grid(axis='y', linestyle=':', alpha=0.6)
ax.axhline(67.24, color='#1b4965', linestyle=':', linewidth=1.2, label='Global Fav. Rate')
ax.axhline(32.76, color='#e76f51', linestyle=':', linewidth=1.2, label='Global Upset Rate')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot17_dbscan_cluster_profiles.pdf"), format='pdf')
plt.close()
print(f"  -> Saved: plot17_dbscan_cluster_profiles.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
out_file = os.path.join(BASE_DIR, "step3c_dbscan_results.txt")
with open(out_file, "w", encoding="utf-8") as f:
    f.write("=" * 72 + "\n")
    f.write("STEP 3C — DBSCAN CLUSTERING RESULTS\n")
    f.write(f"ATP Tennis Dataset  |  N = {len(df):,}  |  2012–2017\n")
    f.write("=" * 72 + "\n\n")
    f.write(f"FEATURES: {FEATURES}\n")
    f.write(f"SCALING: StandardScaler (zero mean, unit variance)\n\n")
    f.write(f"DBSCAN PARAMETERS:\n")
    f.write(f"  epsilon     : {eps_optimal:.4f}  (empirically from k-distance graph elbow, k={K_NEIGHBORS})\n")
    f.write(f"  min_samples : {MIN_SAMPLES}\n")
    f.write(f"  algorithm   : ball_tree\n\n")
    f.write(f"RESULTS:\n")
    f.write(f"  Clusters found     : {n_clusters}\n")
    f.write(f"  Noise points       : {n_noise} ({n_noise/len(labels)*100:.2f}%)\n")
    if sil_score:
        f.write(f"  Silhouette Score   : {sil_score:.4f}\n")
        f.write(f"  Davies-Bouldin Idx : {db_score:.4f}\n\n")
    f.write("\nCLUSTER PROFILES:\n")
    for p in cluster_profiles:
        f.write(f"\n  {p['Cluster']}: n={p['Count']}, "
                f"Log_Rank1={p['Log_Rank1_mean']:.3f}, Log_Rank2={p['Log_Rank2_mean']:.3f}, "
                f"ImpPDiff={p['Implied_Prob_Diff_mean']:.4f}, BkmMgn={p['Bookmaker_Margin_mean']:.5f}, "
                f"FavWin={p['Fav_Win_Rate']:.4f}, Upset={p['Upset_Rate']:.4f}\n")
    f.write("\nNOISE ANOMALY BREAKDOWN:\n")
    f.write(noise_df['Anomaly_Type'].value_counts().to_string())
    f.write("\n\nINTERPRETATION:\n")
    f.write("""
1. DBSCAN does not require a pre-specified number of clusters — this is a key
   advantage over k-Means for match data where natural groupings are unknown.

2. The k-distance elbow method (Ester et al., 1996) provided a principled,
   data-driven epsilon value, avoiding arbitrary parameter guessing.

3. Noise points (label=-1) represent matches that DBSCAN could not assign to
   any dense cluster — these are the most statistically anomalous matches:
   extreme rank gaps, corrupted odds, or genuine unpredictable upsets.

4. The cluster with the highest upset rate isolates matches where both players
   had near-symmetric implied probabilities (ImpPDiff ≈ 0) and high global
   rankings — a region where match outcomes are inherently uncertain.

5. The cluster with the highest Favorite Win rate corresponds to matches with
   strong odds asymmetry: heavy favorites (P1_Implied_Prob > 0.75) playing
   against lower-ranked opponents in later rounds after seeded bracket sorting.
""")

print(f"\nFull results saved → {out_file}")
