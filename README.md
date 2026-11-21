# Analytics & Big Data Lab Exam Report on ATP Dataset
**Indian Institute of Information Technology, Design and Manufacturing, Kancheepuram (IIITDM Kancheepuram)**

* **Student Name:** PARATE GRISHMANK HARIHAR
* **Roll Number:** CS23I1026
* **Course:** Analytics & Systems of Big Data (ASBD)
* **Department:** Department of Computer Science and Engineering
* **Dataset:** ATP Professional Tennis Match Dynamics & Betting Data (2012 to 2017)
* **GitHub Repository:** [https://github.com/grishmankp-crypto/ASBD-LAB-END-SEMESTER-EXAM](https://github.com/grishmankp-crypto/ASBD-LAB-END-SEMESTER-EXAM)

---

## 1. Project Overview
This laboratory examination project presents an end-to-end data mining, statistical analysis, predictive modeling, and big data scalability framework on 14,697 professional Association of Tennis Professionals (ATP) matches from January 2012 to July 2017. 

The analytical pipeline is structured into four core phases:
1. **Data Preprocessing & Discretization:** Converting corrupted Excel integer timestamps into calendar dates, auditing match outcomes, removing unranked qualifiers, eliminating redundant columns, engineering leak-free differential features, and partitioning continuous variables into categorical transaction tiers.
2. **Exploratory Data Analytics (10 Unique Visualizations):** Formulating ten publication-standard statistical and econometric charts grounded in established methodology, each accompanied by detailed quantitative interpretations.
3. **Association Rule Mining (Apriori vs. CHARM):** Native Python implementations of breadth-first Apriori and depth-first CHARM algorithms using vertical tidset intersections, demonstrating a 9.27% lossless compaction into Closed Frequent Itemsets (CFIs).
4. **Predictive Modeling & Clustering:**
   * **CART Decision Tree Classification:** 5-fold stratified cross-validation for depth tuning, holdout test evaluation (61.92% accuracy, 79.60% precision on favorites, 0.6829 ROC-AUC), Gini feature importance analysis, and human-interpretable If-Then production rules.
   * **DBSCAN Density-Based Clustering:** Empirical k-distance elbow calibration (k=5, epsilon=0.4864), identifying four dense operational manifolds and isolating 300 anomalous matches and Cinderella tournament upsets.
5. **Big Data Scalability Benchmarking:** Comparative performance benchmarking between single-threaded Scikit-learn CPU execution and distributed Apache PySpark ML (`FPGrowth`, `DecisionTreeClassifier`) evaluated across bootstrapped dataset sizes and modeled via Amdahl's Law.

---

## 2. Step-by-Step Methodology & Pipeline Details

### Step 1: Data Preprocessing & Feature Engineering
* **Timestamp Reconstruction:** In the raw dataset (`ATP Dataset_2012-01 to 2017-07_Int_V4.csv`, 14,735 rows), the `Date` column was stored as raw Excel serial date integers (e.g., `42737`). Converted to calendar dates using the epoch `T_0 = 1899-12-30`:
  ```text
  Match_Date = 1899-12-30 + timedelta(days=Date)
  ```
  This restores chronological continuity from January 1, 2012 to July 16, 2017 without missing dates.
* **Outcome Integrity:** Verified player name correspondence across `Winner`, `Player1`, and `Player2`. All records exhibited exact matches (7,348 Player 1 wins vs. 7,349 Player 2 wins), confirming a balanced 50/50 baseline.
* **Missing Value Treatment:** Identified and dropped 38 records (0.26%) lacking player rankings (qualifiers/wildcards) to ensure distance metrics and log-transforms remain mathematically valid. The cleaned dataset retains 14,697 rows (99.74% retention).
* **Elimination of Redundant Encodings:** Dropped arbitrary float columns (`Tournament_Int`, `Series_Int`, `Court_Int`, `Surface_Int`, `Round_Int`, `Winner_Int`) in favor of authentic categorical descriptors and explicit one-hot encodings.
* **Engineered Differentials:** Formulated target and continuous features:
  * `Player1_Won`: Binary indicator (`1` if Winner is Player 1, else `0`)
  * `Favorite_Won`: Binary indicator (`1` if higher-ranked player won, else `0`; baseline win rate: 67.24%)
  * `Is_Upset`: `1 - Favorite_Won`
  * `Rank_Diff`: `Rank_1 - Rank_2`
  * `Abs_Rank_Diff`: `|Rank_1 - Rank_2|`
  * `Log_Rank1`, `Log_Rank2`: `ln(1 + Rank_i)`
  * `Odds_Ratio`: `Odds_1 / Odds_2`
  * `Implied_Prob_Diff` (Delta P): `(1 / Odds_1) - (1 / Odds_2)`
  * `Bookmaker_Margin`: `(1 / Odds_1 + 1 / Odds_2) - 1.0` (mean commission: 5.94% ± 0.40%)
* **Interval Discretization:**
  * **Rank Tiers:** `Top10` (r <= 10), `Top30` (11 <= r <= 30), `Top50` (31 <= r <= 50), `Top100` (51 <= r <= 100), `Beyond100` (r > 100)
  * **Rank Disparity:** `Parity` (|Delta r| <= 10), `Moderate` (10 < |Delta r| <= 40), `Large` (40 < |Delta r| <= 100), `Massive` (|Delta r| > 100)
  * **Odds Classes:** `Heavy_Favorite` (O < 1.30), `Moderate_Favorite` (1.30 <= O < 1.70), `Contested` (1.70 <= O <= 2.20), `Underdog` (O > 2.20)

---

### Step 2: Exploratory Data Analytics (10 Unique Visualizations)
All 10 plots are exported as vector PDFs with publication styling:
1. **`plot1_correlation_heatmap.pdf`:** Pearson correlation matrix across 11 features. Inverse relation between odds and implied probabilities (r = -0.68), strong correlation of Delta P with victory (r = 0.44), bookmaker margin uncorrelated with outcome (r = -0.01).
2. **`plot2_violin_odds_by_series.pdf`:** Kernel density violin plots across series. Grand Slams display long right-skewed odds tails (> 8.00) from 128-player opening draws; Masters 1000 exhibits compact unimodal density (1.60 to 2.40).
3. **`plot3_boxplot_rank_diff_by_round.pdf`:** Monotonic convergence of median ranking disparity from 48.0 (Round 1) down to 7.0 (Finals), validating the Competitive Parity Hypothesis.
4. **`plot4_pairplot_scatter_matrix.pdf`:** 3x3 bivariate scatter matrix with diagonal distributions, identifying linear separability when P1 > 0.65.
5. **`plot5_stacked_bar_surface_by_series.pdf`:** 100% stacked bar chart showing hard court predominance (51.7% to 61.3%), clay share (31% to 36%), and seasonal grass allocation.
6. **`plot6_hexbin_density_rankings.pdf`:** 2D log-hexagonal binning with upset overlay (|Delta R| > 80), revealing upset concentration where top-20 athletes meet opponents ranked 80 to 150.
7. **`plot7_timeseries_match_volume_margin.pdf`:** Dual-axis longitudinal tracking of 66 monthly intervals (2012 to 2017) showing seasonal major tournament peaks and stable bookmaker margin (5.6% to 6.3%).
8. **`plot8_grouped_bar_favorite_win_rate_ci.pdf`:** Higher-ranked win rates with 95% confidence intervals. Outdoor Clay exhibits highest favorite predictability (69.8% ± 1.1%), whereas Indoor Hard exhibits highest upset volatility (64.5% ± 1.5%).
9. **`plot9_pareto_chart_top_winners.pdf`:** Pareto distribution showing elite win concentration. The "Big Four" (Djokovic, Federer, Nadal, Murray) each exceed 250 match wins.
10. **`plot10_radar_chart_tournament_profiles.pdf`:** 5-dimensional normalized radar chart mapping Best-of-5 share, upset rate, parity, surface diversity, and indoor arena exposure across tiers.

---

### Step 3A: Association Rule Mining (Apriori vs. CHARM)
* **Apriori Algorithm:** Breadth-first candidate generation with vertical bitset/tidset intersection support counting (`tid(X union Y) = tid(X) intersect tid(Y)`). At `minsup = 0.05` and `minconf = 0.60`, discovered 1,208 frequent itemsets and 1,508 rules in 0.55 seconds.
* **CHARM Algorithm:** Depth-first Itemset-Tidset tree (IT-tree) search with four pruning properties (subsumption, equivalence, superset, disjoint). Mined 1,096 Closed Frequent Itemsets (CFIs) and 1,306 rules in 0.58 seconds.
* **Compaction Analysis:** CHARM achieves a **9.27% reduction in itemsets** and **13.40% reduction in rules** with mathematically **zero information loss** (exact recovery).
* **Key Finding:** Market odds tiers dominate static ATP rankings in association rules, with heavy favorites yielding rule confidences > 84% and lifts > 5.40.

---

### Step 3B: Predictive Modeling (CART Decision Tree)
* **Target:** `Favorite_Won` in {0, 1} (prevents leakage, baseline 67.24%).
* **Cross-Validation:** 5-fold stratified cross-validation over depths d in [3, 10]. Depth d=6 selected as optimal (mean AUC = 0.6801 ± 0.0116).
* **Holdout Evaluation (25% test set, N_test = 3,674):**
  * Accuracy: **61.92%** (2,275 / 3,674)
  * Precision (Favorite Won): **79.60%**
  * Recall (Favorite Won): **58.30%**
  * Specificity (Upset Detection): **69.35%**
  * ROC-AUC Score: **0.6829**
  * Optimal Operating Threshold (Youden's J): TPR = 0.58, FPR = 0.31
  * Training Wall-Clock Time: **0.0609 seconds**
* **Feature Importance:** `Odds_Ratio` dominates with **64.11%** Gini importance, followed by `Implied_Prob_Diff` (9.74%), `Log_Rank1` (7.71%), and `Log_Rank2` (5.20%).
* **Extracted Rules:** Clean tree visualization rendered to depth 3 (`plot14_decision_tree_structure.pdf`) with extracted human-readable If-Then rules.

---

### Step 3C: Density-Based Spatial Clustering (DBSCAN)
* **Feature Space:** Standardized `[Log_Rank1, Log_Rank2, Implied_Prob_Diff, Bookmaker_Margin]`.
* **Parameter Calibration:** Evaluated sorted 5th-nearest neighbor distance curve (k=5). The 95th-percentile transition established `epsilon = 0.4864, MinPts = 5`.
* **Cluster Profiles:**
  * **Cluster 0 (n = 14,376, 97.83%):** Standard tour core (balanced disparity, 5.95% margin, 67.11% favorite win rate).
  * **Clusters 2 & 3 (n = 14 combined):** Deterministic mismatches (100.0% favorite wins, Delta P > 0.95, margins < 2.8%).
  * **Cluster 1 (n = 5):** High-commission contested matchups (60.0% win rate, 6.41% margin).
  * **Noise Points (n = 300, 2.04%):** Isolates 69 extreme ranking gaps (|Delta r| > 500), 3 odds anomalies, and 31 high-severity upsets (e.g., R. Ramanathan def. Y. Bhambri at odds 2.75; G. Garcia-Lopez def. G. Muller at French Open).
* **Validation Scores:** Silhouette = 0.1563, Davies-Bouldin = 0.5474.

---

### Step 3D: Big Data Scalability Analysis
* **Software Stacks:** Single-threaded Scikit-learn CPU engine vs. Distributed Apache PySpark 3.5 (`VectorAssembler`, `DecisionTreeClassifier`, `FPGrowth`).
* **Bootstrap Scaling Benchmark:**
  * Scale 1x (N = 14,695): Scikit-learn **0.055 s** vs. PySpark 12.014 s (Speedup: 0.005x)
  * Scale 5x (N = 73,475): Scikit-learn 0.525 s vs. PySpark 12.135 s (Speedup: 0.043x)
  * Scale 10x (N = 146,950): Scikit-learn 1.115 s vs. PySpark 12.286 s (Speedup: 0.091x)
  * Scale 100x (N = 1,469,500, projected): Scikit-learn 11.85 s vs. PySpark 14.12 s (Speedup: 0.839x)
  * Scale 1000x (N = 14,695,000, projected): Scikit-learn 142.30 s vs. PySpark **32.45 s** (Speedup: **4.385x**)
* **Amdahl's Law Interpretation:** At small scales, PySpark incurs a fixed ~12-second overhead for JVM initialization, thread allocation, and RDD lineage construction. As data scales toward millions of rows, constant overhead becomes negligible, and distributed partitioned memory surpasses single-core CPU limitations.

---

## 3. Directory Structure

```text
ASBD-LAB-END-SEMESTER-EXAM/
├── ATP Dataset_2012-01 to 2017-07_Int_V4.csv   # Raw ATP match dataset (14,735 rows)
├── atp_cleaned_transformed.csv                 # Cleaned dataset with engineered differentials (14,697 rows)
├── atp_transactions_discretized.csv            # Discretized itemset transactions for ARM / CHARM
│
├── run_step1_step2.py                          # Step 1: Preprocessing, date conversion, feature engineering
├── generate_plots.py                           # Step 2: 10 publication-standard EDA visualizations
├── step3_apriori_charm.py                      # Step 3A: Apriori and CHARM algorithms
├── step3_decision_tree.py                      # Step 3B: CART Decision Tree with 5-fold CV & holdout test
├── step3_dbscan.py                             # Step 3C: DBSCAN clustering & k-distance calibration
├── step3_pyspark_benchmark.py                  # Step 3D: PySpark ML pipeline & scalability benchmark
├── refine_step3_plots.py                       # Step 3: High-resolution visualization refiner
│
├── step3a_arm_results.txt                      # Detailed ARM log: 1,208 FIs, 1,096 CFIs, top rules
├── step3b_dt_results.txt                       # Decision Tree evaluation log: metrics, CV scores, rules
├── step3c_dbscan_results.txt                   # DBSCAN clustering log: profiles and anomaly breakdown
├── step3d_pyspark_results.txt                  # Scalability benchmark log: timings across scales
│
├── plot1_correlation_heatmap.pdf               # Fig. 1: Pearson correlation matrix
├── plot2_violin_odds_by_series.pdf             # Fig. 2: Betting odds violin density by tournament series
├── plot3_boxplot_rank_diff_by_round.pdf        # Fig. 3: Ranking parity convergence across rounds
├── plot4_pairplot_scatter_matrix.pdf           # Fig. 4: Multivariate scatter matrix with class densities
├── plot5_stacked_bar_surface_by_series.pdf     # Fig. 5: Court surface allocation across series
├── plot6_hexbin_density_rankings.pdf           # Fig. 6: 2D hexbin density with major upset overlay
├── plot7_timeseries_match_volume_margin.pdf    # Fig. 7: Monthly match volume and bookmaker margin
├── plot8_grouped_bar_favorite_win_rate_ci.pdf  # Fig. 8: Favorite win rate by surface/court + 95% CI
├── plot9_pareto_chart_top_winners.pdf          # Fig. 9: Pareto analysis of player victories
├── plot10_radar_chart_tournament_profiles.pdf  # Fig. 10: Multi-attribute tournament tier radar profiles
├── plot11_12_decision_tree_roc_cm.pdf          # Fig. 11-12: Decision tree ROC curve & confusion matrix
├── plot13_decision_tree_feature_importance.pdf # Fig. 13: Top 12 Gini feature importances
├── plot14_decision_tree_structure.pdf          # Fig. 14: Non-overlapping CART decision tree (depth 3)
├── plot15_dbscan_kdistance.pdf                 # Fig. 15: Sorted 5-NN distance graph for epsilon calibration
├── plot16_dbscan_clusters_pca.pdf              # Fig. 16: 2D PCA projection of DBSCAN density clusters
├── plot17_dbscan_cluster_profiles.pdf          # Fig. 17: Cluster outcome profiles & noise decomposition
├── plot18_19_pyspark_benchmark.pdf             # Fig. 18-19: PySpark execution times & speedup curves
│
├── report.tex                                  # Complete academic project report in LaTeX format
├── overleaf_project.zip                        # 1-Click upload bundle for Overleaf (report.tex + 17 PDFs)
└── README.md                                   # Comprehensive project documentation
```

---

## 4. How to Reproduce All Results

### Prerequisites
* Python 3.10+ (tested on Python 3.12)
* Packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pymupdf`

Install required packages:
```bash
pip install numpy pandas scipy scikit-learn matplotlib pymupdf
```

*(Optional for PySpark distributed benchmark: Java JDK 17 and `pyspark==3.5.1`)*

### Execution Commands (In Order)

1. **Step 1 — Preprocessing & Feature Engineering:**
   ```bash
   python run_step1_step2.py
   ```
   *Generates `atp_cleaned_transformed.csv` and `atp_transactions_discretized.csv`.*

2. **Step 2 — Generate 10 Visualizations:**
   ```bash
   python generate_plots.py
   ```
   *Produces `plot1` through `plot10` vector PDFs.*

3. **Step 3A — Association Rule Mining (Apriori & CHARM):**
   ```bash
   python step3_apriori_charm.py
   ```
   *Generates `step3a_arm_results.txt`.*

4. **Step 3B — CART Decision Tree Classification:**
   ```bash
   python step3_decision_tree.py
   ```
   *Generates `step3b_dt_results.txt`.*

5. **Step 3C — DBSCAN Clustering & Anomaly Isolation:**
   ```bash
   python step3_dbscan.py
   ```
   *Generates `step3c_dbscan_results.txt`.*

6. **Step 3D — PySpark Scalability Benchmark:**
   ```bash
   python step3_pyspark_benchmark.py
   ```
   *Generates `step3d_pyspark_results.txt`.*

7. **Generate Refined Step 3 Visualizations:**
   ```bash
   python refine_step3_plots.py
   ```
   *Produces `plot11_12` through `plot18_19` vector PDFs.*

---

1. Go to [Overleaf.com](https://www.overleaf.com).
2. Click **New Project** -> **Upload Project**.
3. Select `overleaf_project.zip`.
4. Overleaf automatically unpacks the report and all figures. Click **Recompile** to render the complete academic report.
