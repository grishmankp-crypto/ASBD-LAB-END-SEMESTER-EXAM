# Analytics & Systems of Big Data (ASBD) — End-Semester Laboratory Project
**Indian Institute of Information Technology, Design and Manufacturing, Kancheepuram (IIITDM Kancheepuram)**

* **Student Name:** PARATE GRISHMANK HARIHAR
* **Roll Number:** CS23I1026
* **Course:** Analytics & Systems of Big Data (ASBD)
* **Dataset:** ATP Tennis Match Dynamics & Betting Data (2012–2017)
* **Repository:** [https://github.com/grishmankp-crypto/ASBD-LAB-END-SEMESTER-EXAM](https://github.com/grishmankp-crypto/ASBD-LAB-END-SEMESTER-EXAM)

---

## 1. Project Overview & Architecture
This laboratory project implements an exhaustive, end-to-end data mining and predictive analytics framework on 14,697 professional ATP tennis matches spanning 2012–2017. The analysis covers:
1. **Data Preprocessing:** Temporal conversion, noise sanitization, task-relevant feature selection, and rank/odds discretization.
2. **Exploratory Data Analytics (10 Unique IEEE Visualizations):** Multi-attribute statistical and econometric figures rendered to IEEE Transactions standards.
3. **Association Rule Mining:**
   * **Apriori Algorithm:** Breadth-first frequent itemset search with $O(1)$ tidset set intersections.
   * **CHARM Algorithm:** Closed Frequent Itemset (CFI) mining via dual itemset-tidset search trees, demonstrating 9.27% lossless compaction.
4. **Predictive Modeling:**
   * **CART Decision Tree Classification:** 5-fold cross-validated depth tuning, confusion matrix, ROC-AUC (0.6829), and explicit text If-Then decision rules.
   * **DBSCAN Density-Based Clustering:** $k$-distance elbow calibration ($\varepsilon = 0.4864$, $\text{MinPts}=5$), 4 distinct density manifolds, and match-level anomaly/upset isolation.
5. **Big Data Scalability Benchmarks:**
   * Comparative execution analysis between single-threaded Scikit-learn CPU execution and distributed PySpark ML (`FPGrowth`, `DecisionTreeClassifier`) modeled under Amdahl's Law.

---

## 2. Directory Structure & Key Files

```text
D:\ASBD-LAB\
├── ATP Dataset_2012-01 to 2017-07_Int_V4.csv   # Raw ATP match dataset (14,735 rows)
├── atp_cleaned_transformed.csv                 # Cleaned dataset with engineered differentials (14,697 rows)
├── atp_transactions_discretized.csv            # Discretized itemset database for ARM / CHARM
│
├── run_step1_step2.py                          # Data cleaning, selection, transformation & baseline plots
├── generate_ieee_plots.py                      # 10 IEEE-standard EDA visualizations
├── refine_step3_plots.py                       # High-resolution Step 3 visualization refiner
├── step3_apriori_charm.py                      # Scratch Apriori and CHARM algorithms
├── step3_decision_tree.py                      # CART Decision Tree classifier with If-Then rule extraction
├── step3_dbscan.py                             # DBSCAN clustering and k-distance calibration
├── step3_pyspark_benchmark.py                  # PySpark pipeline and scalability benchmarks
│
├── step3a_arm_results.txt                      # 1,208 frequent itemsets, 1,096 CFIs, top rules
├── step3b_dt_results.txt                       # Decision Tree metrics, confusion matrix, If-Then rules
├── step3c_dbscan_results.txt                   # DBSCAN cluster profiles and anomaly breakdown
├── step3d_pyspark_results.txt                  # Benchmark execution timings across 1x, 5x, 10x scales
│
├── plot1_correlation_heatmap.pdf               # Fig. 1: Pearson r correlation matrix
├── plot2_violin_odds_by_series.pdf             # Fig. 2: Betting odds violin density by series
├── plot3_boxplot_rank_diff_by_round.pdf        # Fig. 3: Ranking parity convergence across rounds
├── plot4_pairplot_scatter_matrix.pdf           # Fig. 4: Multivariate scatter matrix with KDE
├── plot5_stacked_bar_surface_by_series.pdf     # Fig. 5: Surface allocation across series
├── plot6_hexbin_density_rankings.pdf           # Fig. 6: Log10 2D hexbin with upset overlay
├── plot7_timeseries_match_volume_margin.pdf    # Fig. 7: Monthly match volume and vig dynamics
├── plot8_grouped_bar_favorite_win_rate_ci.pdf  # Fig. 8: Favorite win rate by surface/court + 95% CI
├── plot9_pareto_chart_top_winners.pdf          # Fig. 9: Pareto 80/20 analysis of match wins
├── plot10_radar_chart_tournament_profiles.pdf  # Fig. 10: Multi-attribute tournament radar profiles
├── plot11_12_decision_tree_roc_cm.pdf          # Fig. 11-12: ROC-AUC curve & Confusion Matrix
├── plot13_decision_tree_feature_importance.pdf # Fig. 13: Gini feature importance ranking
├── plot14_decision_tree_structure.pdf          # Fig. 14: Non-overlapping CART decision tree
├── plot15_dbscan_kdistance.pdf                 # Fig. 15: K-distance graph for epsilon selection
├── plot16_dbscan_clusters_pca.pdf              # Fig. 16: 2D PCA projection of DBSCAN clusters
├── plot17_dbscan_cluster_profiles.pdf          # Fig. 17: Cluster outcome signatures & noise breakdown
├── plot18_19_pyspark_benchmark.pdf             # Fig. 18-19: PySpark execution timing & speedup
│
├── report.tex                                  # Complete IEEE-style LaTeX academic project report
└── README.md                                   # Project documentation and reproduction instructions
```

---

## 3. How to Reproduce

### Prerequisites
* Python 3.10+
* Packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pymupdf`

```bash
pip install numpy pandas scipy scikit-learn matplotlib pymupdf
```

### Execution Steps
1. **Preprocessing & Visualizations:**
   ```bash
   python generate_ieee_plots.py
   ```
2. **Association Rule Mining (Apriori & CHARM):**
   ```bash
   python step3_apriori_charm.py
   ```
3. **Decision Tree Classification:**
   ```bash
   python step3_decision_tree.py
   ```
4. **DBSCAN Density Clustering:**
   ```bash
   python step3_dbscan.py
   ```
5. **Refine Plots to Research Standards:**
   ```bash
   python refine_step3_plots.py
   ```
6. **Big Data Benchmark:**
   ```bash
   python step3_pyspark_benchmark.py
   ```

---

## 4. Key Results Summary

* **Dataset Size:** 14,697 clean matches, 25 raw features + 12 engineered features.
* **Favorite Win Rate:** 67.24% baseline across 5.5 seasons.
* **Apriori vs. CHARM:** 1,208 frequent itemsets compacted to 1,096 Closed Frequent Itemsets (9.27% reduction without information loss).
* **Decision Tree (CART):** Accuracy 61.92%, ROC-AUC 0.6829, Precision 79.60%. Dominant feature: `Odds_Ratio` (64.11% Gini importance).
* **DBSCAN Clustering:** $\varepsilon = 0.4864$, $\text{MinPts} = 5$. Identified 4 clusters (Silhouette = 0.1563) and 300 noise outliers representing high-severity upsets and market anomalies.
* **Scalability Analysis:** Validated Amdahl's Law threshold where distributed PySpark parallelism overcomes JVM initialization overhead once data scales past $10^5$ transactions.
