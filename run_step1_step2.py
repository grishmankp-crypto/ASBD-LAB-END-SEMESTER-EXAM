import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import scipy.stats as stats

# Set styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = r"D:\ASBD-LAB"
RAW_FILE = os.path.join(BASE_DIR, "ATP Dataset_2012-01 to 2017-07_Int_V4.csv")
CLEAN_FILE = os.path.join(BASE_DIR, "atp_cleaned_transformed.csv")
TRANS_FILE = os.path.join(BASE_DIR, "atp_transactions_discretized.csv")

print("=" * 70)
print("STEP 1: DATA PREPROCESSING (CLEANING, SELECTION, TRANSFORMATION)")
print("=" * 70)

# Load Raw Dataset
df_raw = pd.read_csv(RAW_FILE)
print(f"Raw Dataset Loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# 1. Cleaning
# A. Convert Excel serial dates to standard YYYY-MM-DD
# Excel epoch base 1899-12-30
df = df_raw.copy()
df['Match_Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df['Date'], unit='D')
df['Year'] = df['Match_Date'].dt.year
df['Month'] = df['Match_Date'].dt.month
df['Year_Month'] = df['Match_Date'].dt.to_period('M')

# B. String cleaning
for col in ['Tournament', 'Series', 'Court', 'Surface', 'Round', 'Winner', 'Player1', 'Player2']:
    df[col] = df[col].astype(str).str.strip()

# C. Handle Missing Values / Noise in Rankings
initial_len = len(df)
df = df.dropna(subset=['Player1_Rank', 'Player2_Rank', 'Player1_Odds', 'Player2_Odds', 'Player1_Implied_Prob', 'Player2_Implied_Prob'])
cleaned_len = len(df)
print(f"Removed {initial_len - cleaned_len} records with missing rankings/odds ({cleaned_len} rows retained, {cleaned_len/initial_len*100:.2f}%)")

# Ensure numeric datatypes
df['Player1_Rank'] = df['Player1_Rank'].astype(int)
df['Player2_Rank'] = df['Player2_Rank'].astype(int)
df['Player1_Odds'] = df['Player1_Odds'].astype(float)
df['Player2_Odds'] = df['Player2_Odds'].astype(float)
df['Player1_Implied_Prob'] = df['Player1_Implied_Prob'].astype(float)
df['Player2_Implied_Prob'] = df['Player2_Implied_Prob'].astype(float)

# 2. Data Selection & Feature Engineering
# Construct ground-truth target variables
df['Player1_Won'] = (df['Winner'] == df['Player1']).astype(int)
df['Higher_Ranked_Player'] = np.where(df['Player1_Rank'] < df['Player2_Rank'], 1, 2)
# An upset occurs if the player with the higher numeric rank (worse player) wins
df['Favorite_Won'] = np.where(df['Player1_Rank'] < df['Player2_Rank'], df['Player1_Won'], 1 - df['Player1_Won'])
df['Is_Upset'] = 1 - df['Favorite_Won']

# Engineered predictive features
df['Rank_Diff'] = df['Player1_Rank'] - df['Player2_Rank']
df['Abs_Rank_Diff'] = (df['Player1_Rank'] - df['Player2_Rank']).abs()
df['Log_Rank1'] = np.log1p(df['Player1_Rank'])
df['Log_Rank2'] = np.log1p(df['Player2_Rank'])
df['Odds_Ratio'] = df['Player1_Odds'] / df['Player2_Odds']
df['Implied_Prob_Diff'] = df['Player1_Implied_Prob'] - df['Player2_Implied_Prob']
df['Bookmaker_Margin'] = (df['Player1_Implied_Prob'] + df['Player2_Implied_Prob']) - 1.0

# 3. Data Transformation & Discretization (for ARM & CHARM)
def rank_bin(rank):
    if rank <= 10:
        return 'Top10'
    elif rank <= 30:
        return 'Top30'
    elif rank <= 50:
        return 'Top50'
    elif rank <= 100:
        return 'Top100'
    else:
        return 'Beyond100'

def diff_bin(diff):
    if diff <= 10:
        return 'Parity_Diff<=10'
    elif diff <= 40:
        return 'Moderate_Diff_11-40'
    elif diff <= 100:
        return 'Large_Diff_41-100'
    else:
        return 'Massive_Diff>100'

def odds_tier(odds):
    if odds < 1.30:
        return 'Heavy_Favorite'
    elif odds < 1.70:
        return 'Moderate_Favorite'
    elif odds <= 2.20:
        return 'Contested_Match'
    else:
        return 'Underdog'

df['P1_Rank_Tier'] = df['Player1_Rank'].apply(rank_bin)
df['P2_Rank_Tier'] = df['Player2_Rank'].apply(rank_bin)
df['Disparity_Tier'] = df['Abs_Rank_Diff'].apply(diff_bin)
df['P1_Odds_Tier'] = df['Player1_Odds'].apply(odds_tier)
df['P2_Odds_Tier'] = df['Player2_Odds'].apply(odds_tier)
df['Outcome_Label'] = df['Player1_Won'].apply(lambda x: 'Winner=Player1' if x == 1 else 'Winner=Player2')
df['Upset_Label'] = df['Is_Upset'].apply(lambda x: 'Match=Upset' if x == 1 else 'Match=Expected_Win')

# Save Cleaned and Transformed Dataset
df.to_csv(CLEAN_FILE, index=False)
print(f"Cleaned and transformed dataset saved to: {CLEAN_FILE}")

# Prepare transactional dataset for ARM (Apriori and CHARM)
trans_cols = ['Series', 'Court', 'Surface', 'Round', 'Disparity_Tier', 'P1_Odds_Tier', 'P2_Odds_Tier', 'Outcome_Label', 'Upset_Label']
df_transactions = df[trans_cols].copy()
# Format each column as Item=Value
for col in trans_cols:
    df_transactions[col] = col + "=" + df_transactions[col].astype(str)
df_transactions.to_csv(TRANS_FILE, index=False)
print(f"Discretized transactional dataset saved to: {TRANS_FILE}")

print("\nData Preprocessing Summary:")
print(f"  - Total Matches Analyzed: {len(df)}")
print(f"  - Overall Favorite Win Rate: {df['Favorite_Won'].mean()*100:.2f}%")
print(f"  - Overall Upset Rate: {df['Is_Upset'].mean()*100:.2f}%")
print(f"  - Average Bookmaker Over-round Margin: {df['Bookmaker_Margin'].mean()*100:.2f}%")

print("\n" + "=" * 70)
print("STEP 2: GENERATING 10 UNIQUE VISUALIZATIONS (PDF FORMAT)")
print("=" * 70)

# -------------------------------------------------------------
# PLOT 1: Correlation Heatmap
# Reference: Friendly, M. (2002). Corrgrams: Exploratory displays for correlation matrices.
# -------------------------------------------------------------
print("Generating Plot 1: Correlation Heatmap...")
plt.figure(figsize=(9, 7))
num_cols = ['Player1_Rank', 'Player2_Rank', 'Rank_Diff', 'Abs_Rank_Diff', 'Player1_Odds', 'Player2_Odds', 
            'Player1_Implied_Prob', 'Player2_Implied_Prob', 'Bookmaker_Margin', 'Player1_Won', 'Is_Upset']
corr_matrix = df[num_cols].corr()

plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Pearson Correlation Coefficient (r)', fraction=0.046, pad=0.04)
plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha='right', fontsize=9)
plt.yticks(range(len(num_cols)), num_cols, fontsize=9)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        val = corr_matrix.iloc[i, j]
        plt.text(j, i, f"{val:.2f}", ha='center', va='center', 
                 color='white' if abs(val) > 0.5 else 'black', fontsize=7.5, weight='bold')

plt.title("Plot 1: Inter-Attribute Pearson Correlation Matrix\n(ATP Match Dynamics & Betting Probabilities)", fontsize=11, weight='bold', pad=12)
plt.tight_layout()
p1_path = os.path.join(BASE_DIR, "plot1_correlation_heatmap.pdf")
plt.savefig(p1_path, format='pdf', dpi=300)
plt.close()
print(f"  -> Saved: {p1_path}")

# -------------------------------------------------------------
# PLOT 2: Violin Plot with Embedded Quartile Boxplots
# Reference: Hintze, J. L., & Nelson, R. D. (1998). Violin plots.
# -------------------------------------------------------------
print("Generating Plot 2: Violin Plot across Tournament Series...")
plt.figure(figsize=(10, 6))
series_order = ['GrandSlam', 'Masters1000', 'ATP500', 'ATP250']
data_violin = [df[df['Series'] == s]['Player1_Odds'].clip(upper=10).values for s in series_order]

parts = plt.violinplot(data_violin, positions=range(len(series_order)), showmeans=True, showmedians=True, widths=0.7)
for pc in parts['bodies']:
    pc.set_facecolor('#1f77b4')
    pc.set_edgecolor('#08306b')
    pc.set_alpha(0.7)

parts['cmeans'].set_color('red')
parts['cmedians'].set_color('black')
parts['cbars'].set_color('#333333')

plt.xticks(range(len(series_order)), series_order, fontsize=10, weight='bold')
plt.ylabel("Decimal Betting Odds (Player 1, Capped at 10.0)", fontsize=10)
plt.title("Plot 2: Violin Plot of Decimal Betting Odds Across Tournament Series Tiers\n(Multimodal Density Distributions & Interquartile Dispersion)", fontsize=11, weight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.5)

legend_elements = [
    Patch(facecolor='red', edgecolor='red', label='Mean Odds'),
    Patch(facecolor='black', edgecolor='black', label='Median Odds'),
    Patch(facecolor='#1f77b4', edgecolor='#08306b', label='KDE Density Envelope')
]
plt.legend(handles=legend_elements, loc='upper right', frameon=True)
plt.tight_layout()
p2_path = os.path.join(BASE_DIR, "plot2_violin_odds_by_series.pdf")
plt.savefig(p2_path, format='pdf', dpi=300)
plt.close()
print(f"  -> Saved: {p2_path}")

# -------------------------------------------------------------
# PLOT 3: Box Plot with Outlier Whiskers
# Reference: Tukey, J. W. (1977). Exploratory Data Analysis.
# -------------------------------------------------------------
print("Generating Plot 3: Box Plot of Absolute Rank Difference across Tournament Rounds...")
plt.figure(figsize=(11, 6))
rounds_order = ['1stRound', '2ndRound', '3rdRound', '4thRound', 'Quarterfinals', 'Semifinals', 'TheFinal']
rounds_data = [df[df['Round'] == r]['Abs_Rank_Diff'].dropna().values for r in rounds_order]

box = plt.boxplot(rounds_data, tick_labels=rounds_order, patch_artist=True, 
                  boxprops=dict(facecolor='#aec7e8', color='#1f77b4'),
                  whiskerprops=dict(color='#1f77b4', linewidth=1.2),
                  capprops=dict(color='#1f77b4', linewidth=1.2),
                  medianprops=dict(color='#d62728', linewidth=2),
                  flierprops=dict(marker='o', markerfacecolor='#ff7f0e', markersize=3, alpha=0.5))

plt.ylabel("Absolute Ranking Disparity (|Rank1 - Rank2|)", fontsize=10)
plt.xlabel("Tournament Round Progression", fontsize=10)
plt.title("Plot 3: Box Plot of Ranking Parity Across Match Rounds\n(Demonstrating Convergence Towards Competitive Parity)", fontsize=11, weight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
p3_path = os.path.join(BASE_DIR, "plot3_boxplot_rank_diff_by_round.pdf")
plt.savefig(p3_path, format='pdf', dpi=300)
plt.close()
print(f"  -> Saved: {p3_path}")

# -------------------------------------------------------------
# PLOT 4: Multivariate Pairwise Scatter Matrix
# Reference: Cleveland, W. S. (1993). Visualizing Data.
# -------------------------------------------------------------
print("Generating Plot 4: Multivariate Scatter Matrix...")
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
scatter_vars = ['Player1_Implied_Prob', 'Player2_Implied_Prob', 'Abs_Rank_Diff']
labels = ['P1 Implied Prob', 'P2 Implied Prob', 'Abs Rank Diff']

sample_subset = df.sample(n=min(2500, len(df)), random_state=42)

for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        if i == j:
            # Diagonal: KDE / Histogram
            var_data_p1 = sample_subset[sample_subset['Player1_Won'] == 1][scatter_vars[i]]
            var_data_p2 = sample_subset[sample_subset['Player1_Won'] == 0][scatter_vars[i]]
            ax.hist([var_data_p1, var_data_p2], bins=20, stacked=True, color=['#2ca02c', '#1f77b4'], alpha=0.7)
            ax.set_title(labels[i], fontsize=9, weight='bold')
        else:
            # Scatter
            p1_win = sample_subset[sample_subset['Player1_Won'] == 1]
            p2_win = sample_subset[sample_subset['Player1_Won'] == 0]
            ax.scatter(p1_win[scatter_vars[j]], p1_win[scatter_vars[i]], c='#2ca02c', s=8, alpha=0.3, label='P1 Won')
            ax.scatter(p2_win[scatter_vars[j]], p2_win[scatter_vars[i]], c='#1f77b4', s=8, alpha=0.3, label='P2 Won')
        if i == 2:
            ax.set_xlabel(labels[j], fontsize=8)
        if j == 0:
            ax.set_ylabel(labels[i], fontsize=8)
        ax.tick_params(labelsize=7)

fig.suptitle("Plot 4: Multivariate Pairwise Scatter Matrix with Marginal Distributions\n(Separation of Match Outcomes in Probability & Disparity Space)", fontsize=11, weight='bold', y=0.99)
plt.tight_layout()
p4_path = os.path.join(BASE_DIR, "plot4_pairplot_scatter_matrix.pdf")
plt.savefig(p4_path, format='pdf', dpi=300)
plt.close()
print(f"  -> Saved: {p4_path}")

# -------------------------------------------------------------
# PLOT 5: Stacked Percentage Bar Chart
# Reference: Few, S. (2009). Now You See It: Simple Visualization Techniques.
# -------------------------------------------------------------
print("Generating Plot 5: Stacked Percentage Bar Chart...")
plt.figure(figsize=(9, 6))
ct = pd.crosstab(df['Series'], df['Surface'], normalize='index') * 100
series_idx = ['GrandSlam', 'Masters1000', 'ATP500', 'ATP250']
ct = ct.reindex(series_idx)[['Hard', 'Clay', 'Grass']]

bottom = np.zeros(len(ct))
colors = ['#4a7bb0', '#d95f02', '#2ca02c']
for idx, col in enumerate(ct.columns):
    values = ct[col].values
    plt.bar(ct.index, values, bottom=bottom, label=col, color=colors[idx], width=0.55, edgecolor='#333333')
    for k in range(len(values)):
        if values[k] > 5:
            plt.text(k, bottom[k] + values[k]/2, f"{values[k]:.1f}%", ha='center', va='center', color='white', weight='bold', fontsize=9)
    bottom += values

plt.ylabel("Court Surface Proportion (%)", fontsize=10)
plt.xlabel("Tournament Series Classification", fontsize=10)
plt.title("Plot 5: Stacked Percentage Bar Chart of Surface Allocations Across Series Tiers", fontsize=11, weight='bold')
plt.legend(title='Surface', loc='upper right', frameon=True)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
p5_path = os.path.join(BASE_DIR, "plot5_stacked_bar_surface_by_series.pdf")
plt.savefig(p5_path, format='pdf', dpi=300)
plt.close()
print(f"  -> Saved: {p5_path}")

# -------------------------------------------------------------
# PLOT 6: 2D Hexagonal Binning Density Plot with Outliers
# Reference: Carr, D. B. et al. (1987). Scatterplot matrix techniques for large N.
# -------------------------------------------------------------
print("Generating Plot 6: 2D Hexbin Density Plot...")
plt.figure(figsize=(9, 7))
hb = plt.hexbin(df['Log_Rank1'], df['Log_Rank2'], gridsize=35, cmap='viridis', mincnt=1, bins='log')
cb = plt.colorbar(hb, label='Log10(Match Frequency)', pad=0.03)

# Highlight extreme upsets where low-ranked underdog beat top 10 player
upsets = df[(df['Is_Upset'] == 1) & (df['Abs_Rank_Diff'] > 80)]
plt.scatter(upsets['Log_Rank1'], upsets['Log_Rank2'], color='#e41a1c', s=12, alpha=0.6, label='Major Upsets (|Rank Diff| > 80)')

plt.xlabel("Log(Player 1 Rank)", fontsize=10)
plt.ylabel("Log(Player 2 Rank)", fontsize=10)
plt.title("Plot 6: 2D Hexagonal Binning Density of Matchups with Major Upset Overlay\n(Log-Transformed Global Ranking Space)", fontsize=11, weight='bold')
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
p6_path = os.path.join(BASE_DIR, "plot6_hexbin_density_rankings.pdf")
plt.savefig(p6_path, format='pdf', dpi=300)
plt.close()
print(f"  -> Saved: {p6_path}")

# -------------------------------------------------------------
# PLOT 7: Longitudinal Time-Series Line Plot with Rolling Average
# Reference: Tufte, E. R. (2001). The Visual Display of Quantitative Information.
# -------------------------------------------------------------
print("Generating Plot 7: Time-Series Trend Plot...")
monthly = df.groupby('Year_Month').agg(
    Match_Count=('ATP', 'count'),
    Avg_Margin=('Bookmaker_Margin', 'mean'),
    Upset_Rate=('Is_Upset', 'mean')
).reset_index()
monthly['Date_Plot'] = monthly['Year_Month'].dt.to_timestamp()

fig, ax1 = plt.subplots(figsize=(11, 6))

color1 = '#1f77b4'
ax1.plot(monthly['Date_Plot'], monthly['Match_Count'], marker='o', markersize=3, color=color1, alpha=0.5, label='Monthly Matches')
ax1.plot(monthly['Date_Plot'], monthly['Match_Count'].rolling(6).mean(), color='#08306b', linewidth=2.2, label='6-Month Rolling Mean (Matches)')
ax1.set_xlabel("Timeline (2012 - 2017)", fontsize=10)
ax1.set_ylabel("Monthly Match Volume", color='#08306b', fontsize=10)
ax1.tick_params(axis='y', labelcolor='#08306b')
ax1.grid(True, linestyle='--', alpha=0.4)

ax2 = ax1.twinx()
color2 = '#d95f02'
ax2.plot(monthly['Date_Plot'], monthly['Avg_Margin']*100, color=color2, linestyle='--', linewidth=1.8, label='Avg Bookmaker Margin (%)')
ax2.set_ylabel("Bookmaker Margin (%)", color=color2, fontsize=10)
ax2.tick_params(axis='y', labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)

plt.title("Plot 7: Longitudinal Time-Series Analysis of Match Volume and Bookmaker Over-round Margin", fontsize=11, weight='bold')
plt.tight_layout()
p7_path = os.path.join(BASE_DIR, "plot7_timeseries_match_volume_margin.pdf")
plt.savefig(p7_path, format='pdf', dpi=300)
plt.close()
print(f"  -> Saved: {p7_path}")

# -------------------------------------------------------------
# PLOT 8: Clustered Bar Chart with 95% Confidence Intervals
# Reference: Wickham, H. (2016). ggplot2: Elegant Graphics for Data Analysis.
# -------------------------------------------------------------
print("Generating Plot 8: Clustered Bar Chart with 95% CI...")
plt.figure(figsize=(9, 6))
grouped_stats = df.groupby(['Surface', 'Court'])['Favorite_Won'].agg(
    Mean='mean',
    Count='count',
    Std='std'
).reset_index()
# 95% CI: 1.96 * std / sqrt(n)
grouped_stats['CI'] = 1.96 * (grouped_stats['Std'] / np.sqrt(grouped_stats['Count'])) * 100
grouped_stats['Mean_Pct'] = grouped_stats['Mean'] * 100

surfaces = ['Hard', 'Clay', 'Grass']
x = np.arange(len(surfaces))
width = 0.35

outdoor_vals = []
outdoor_err = []
indoor_vals = []
indoor_err = []

for s in surfaces:
    row_out = grouped_stats[(grouped_stats['Surface'] == s) & (grouped_stats['Court'] == 'Outdoor')]
    if not row_out.empty:
        outdoor_vals.append(row_out['Mean_Pct'].values[0])
        outdoor_err.append(row_out['CI'].values[0])
    else:
        outdoor_vals.append(0)
        outdoor_err.append(0)

    row_in = grouped_stats[(grouped_stats['Surface'] == s) & (grouped_stats['Court'] == 'Indoor')]
    if not row_in.empty:
        indoor_vals.append(row_in['Mean_Pct'].values[0])
        indoor_err.append(row_in['CI'].values[0])
    else:
        indoor_vals.append(0)
        indoor_err.append(0)

plt.bar(x - width/2, outdoor_vals, width, yerr=outdoor_err, capsize=5, label='Outdoor', color='#2b83ba', edgecolor='#333333')
plt.bar(x + width/2, indoor_vals, width, yerr=indoor_err, capsize=5, label='Indoor', color='#fdae61', edgecolor='#333333')

plt.xticks(x, surfaces, fontsize=10, weight='bold')
plt.ylabel("Favorite Win Rate (%) ± 95% CI", fontsize=10)
plt.xlabel("Court Surface", fontsize=10)
plt.ylim(50, 85)
plt.title("Plot 8: Clustered Bar Chart of Favorite Win Rates across Surface & Court Environment", fontsize=11, weight='bold')
plt.legend(title='Court Setting', frameon=True)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
p8_path = os.path.join(BASE_DIR, "plot8_grouped_bar_favorite_win_rate_ci.pdf")
plt.savefig(p8_path, format='pdf', dpi=300)
plt.close()
print(f"  -> Saved: {p8_path}")

# -------------------------------------------------------------
# PLOT 9: Pareto Chart of Match Wins
# Reference: Juran, J. M. (1951). Quality Control Handbook.
# -------------------------------------------------------------
print("Generating Plot 9: Pareto Chart...")
plt.figure(figsize=(12, 6))
top_winners = df['Winner'].value_counts().head(20)
cum_pct = top_winners.cumsum() / top_winners.sum() * 100

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(top_winners.index, top_winners.values, color='#3182bd', edgecolor='#1c4a6e', width=0.6)
ax1.set_ylabel("Total Matches Won (2012-2017)", color='#1c4a6e', fontsize=10, weight='bold')
ax1.set_xticklabels(top_winners.index, rotation=45, ha='right', fontsize=9)
ax1.set_xlabel("ATP Professional Players", fontsize=10)

ax2 = ax1.twinx()
ax2.plot(top_winners.index, cum_pct, color='#de2d26', marker='D', markersize=4, linewidth=2, label='Cumulative % of Top 20 Wins')
ax2.axhline(80, color='gray', linestyle=':', linewidth=1.5, label='80% Pareto Reference Line')
ax2.set_ylabel("Cumulative Percentage (%)", color='#de2d26', fontsize=10, weight='bold')
ax2.set_ylim(0, 105)

ax2.legend(loc='lower right', frameon=True)
plt.title("Plot 9: Pareto Analysis of Match Victories Among Top 20 ATP Competitors\n(Demonstrating Skewed Distribution of Elite Athletic Success)", fontsize=11, weight='bold')
plt.tight_layout()
p9_path = os.path.join(BASE_DIR, "plot9_pareto_chart_top_winners.pdf")
plt.savefig(p9_path, format='pdf', dpi=300)
plt.close()
print(f"  -> Saved: {p9_path}")

# -------------------------------------------------------------
# PLOT 10: Polar / Radar Profile Chart
# Reference: Chambers, J. M. et al. (1983). Graphical Methods for Data Analysis.
# -------------------------------------------------------------
print("Generating Plot 10: Radar / Polar Chart...")
radar_categories = ['Avg Implied Fav Prob', 'Upset Rate', 'Avg Normalized Rank Gap', 'Outdoor Ratio', 'Best-of-5 Ratio']
N = len(radar_categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1] # complete the loop

series_radar = ['GrandSlam', 'Masters1000', 'ATP500', 'ATP250']
radar_df = pd.DataFrame(index=series_radar, columns=radar_categories)

for s in series_radar:
    sub = df[df['Series'] == s]
    radar_df.loc[s, 'Avg Implied Fav Prob'] = np.maximum(sub['Player1_Implied_Prob'], sub['Player2_Implied_Prob']).mean()
    radar_df.loc[s, 'Upset Rate'] = sub['Is_Upset'].mean()
    radar_df.loc[s, 'Avg Normalized Rank Gap'] = (sub['Abs_Rank_Diff'].mean()) / 120.0 # normalized
    radar_df.loc[s, 'Outdoor Ratio'] = (sub['Court'] == 'Outdoor').mean()
    radar_df.loc[s, 'Best-of-5 Ratio'] = (sub['Best_of'] == 5).mean()

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors_radar = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

for idx, s in enumerate(series_radar):
    values = radar_df.loc[s].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=s, color=colors_radar[idx])
    ax.fill(angles, values, color=colors_radar[idx], alpha=0.15)

plt.xticks(angles[:-1], radar_categories, fontsize=9, weight='bold')
ax.set_rlabel_position(30)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
plt.ylim(0, 1.05)
plt.title("Plot 10: Multi-Attribute Radar Characterization Across ATP Tournament Tiers", fontsize=11, weight='bold', pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True)
plt.tight_layout()
p10_path = os.path.join(BASE_DIR, "plot10_radar_chart_tournament_profiles.pdf")
plt.savefig(p10_path, format='pdf', dpi=300)
plt.close()
print(f"  -> Saved: {p10_path}")

print("\n" + "=" * 70)
print("ALL 10 PLOTS GENERATED AND SAVED TO DISK IN PDF FORMAT SUCCESSFULLY!")
print("=" * 70)
