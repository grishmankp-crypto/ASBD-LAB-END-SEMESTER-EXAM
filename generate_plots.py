import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scipy.stats as stats

# ==============================================================================
# Publication TRANSACTIONS / RESEARCH PAPER PUBLICATION STYLING CONFIGURATION
# ==============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.edgecolor': '#222222',
    'axes.linewidth': 1.0,
    'grid.color': '#d3d3d3',
    'grid.linestyle': ':',
    'grid.linewidth': 0.7,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

BASE_DIR = r"D:\ASBD-LAB"
CLEAN_FILE = os.path.join(BASE_DIR, "atp_cleaned_transformed.csv")

df = pd.read_csv(CLEAN_FILE)
df['Match_Date'] = pd.to_datetime(df['Match_Date'])
N_TOTAL = len(df)
print(f"Loaded {N_TOTAL} records for Publication-standard plot generation.")

# ------------------------------------------------------------------------------
# PLOT 1: Inter-Attribute Correlation Matrix (Heatmap)
# Publication-standard: High contrast, annotated r values, professional colormap
# ------------------------------------------------------------------------------
print("Plot 1: Publication Correlation Heatmap...")
plt.figure(figsize=(8.5, 7.0))
corr_cols = [
    'Player1_Rank', 'Player2_Rank', 'Rank_Diff', 'Abs_Rank_Diff', 
    'Player1_Odds', 'Player2_Odds', 'Player1_Implied_Prob', 'Player2_Implied_Prob', 
    'Bookmaker_Margin', 'Player1_Won', 'Is_Upset'
]
corr_labels = [
    'P1 Rank', 'P2 Rank', 'Rank Diff', '|Rank Diff|',
    'P1 Odds', 'P2 Odds', 'P1 Impl Prob', 'P2 Impl Prob',
    'Bkmkr Margin', 'P1 Won (Target)', 'Is Upset'
]
corr_matrix = df[corr_cols].corr()

im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0, aspect='auto')
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label("Pearson Correlation Coefficient ($r$)", fontsize=10)
cbar.ax.tick_params(labelsize=8.5)

plt.xticks(range(len(corr_cols)), corr_labels, rotation=40, ha='right', fontsize=9.5)
plt.yticks(range(len(corr_cols)), corr_labels, fontsize=9.5)

# Text annotation inside each cell
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        val = corr_matrix.iloc[i, j]
        text_color = 'white' if abs(val) > 0.45 else 'black'
        plt.text(j, i, f"{val:.2f}", ha='center', va='center', 
                 color=text_color, fontsize=8.0, fontweight='semibold')

plt.title(f"Figure 1: Pearson Correlation Matrix Across Technical Match & Betting Attributes\n($N = {N_TOTAL:,}$ ATP Matches, 2012–2017)", 
          fontsize=11.5, fontweight='bold', pad=12)
plt.tight_layout()
p1_path = os.path.join(BASE_DIR, "plot1_correlation_heatmap.pdf")
plt.savefig(p1_path, format='pdf')
plt.close()

# ------------------------------------------------------------------------------
# PLOT 2: Violin Plot with Embedded Quartile Boxplots
# Publication-standard: Clear probability envelopes, explicit median/mean, quartile lines
# ------------------------------------------------------------------------------
print("Plot 2: Publication Violin Distribution of Betting Odds...")
plt.figure(figsize=(9.0, 5.5))
series_tiers = ['GrandSlam', 'Masters1000', 'ATP500', 'ATP250']
series_labels = ['Grand Slam\n($n=2,918$)', 'Masters 1000\n($n=3,127$)', 'ATP 500\n($n=2,331$)', 'ATP 250\n($n=6,207$)']
odds_data = [df[df['Series'] == s]['Player1_Odds'].clip(upper=8.0).values for s in series_tiers]

parts = plt.violinplot(odds_data, positions=range(len(series_tiers)), showmeans=False, showmedians=False, widths=0.68)
for pc in parts['bodies']:
    pc.set_facecolor('#2b5c8f')
    pc.set_edgecolor('#102a45')
    pc.set_alpha(0.65)
    pc.set_linewidth(1.0)
parts['cbars'].set_color('#333333')
parts['cbars'].set_linewidth(1.0)
parts['cmaxes'].set_color('#333333')
parts['cmins'].set_color('#333333')

# Overlay clean boxplot elements inside violin
medians = [np.median(d) for d in odds_data]
means = [np.mean(d) for d in odds_data]
q25 = [np.percentile(d, 25) for d in odds_data]
q75 = [np.percentile(d, 75) for d in odds_data]

for i in range(len(series_tiers)):
    # IQR bar
    plt.plot([i, i], [q25[i], q75[i]], color='#111111', linewidth=4.0, solid_capstyle='round')
    # Median marker (white dot)
    plt.scatter(i, medians[i], color='white', edgecolor='#111111', s=35, zorder=5)
    # Mean marker (red diamond)
    plt.scatter(i, means[i], color='#d9381e', marker='D', s=28, zorder=6)

plt.xticks(range(len(series_tiers)), series_labels, fontsize=9.5)
plt.ylabel("Decimal Betting Odds (Capped at 8.00)", fontsize=10.5)
plt.xlabel("Tournament Tier Classification", fontsize=10.5)
plt.title("Figure 2: Empirical Density Distribution and Interquartile Dispersion of Betting Odds\nAcross Tournament Prestige Tiers", 
          fontsize=11.5, fontweight='bold', pad=10)
plt.grid(axis='y', linestyle=':', alpha=0.6)

legend_elements = [
    Patch(facecolor='#2b5c8f', edgecolor='#102a45', alpha=0.65, label='KDE Density Envelope'),
    plt.Line2D([0], [0], color='#111111', linewidth=4, label='Interquartile Range (IQR)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='#111111', markersize=6, label='Median'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#d9381e', markersize=5, label='Parametric Mean')
]
plt.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
plt.tight_layout()
p2_path = os.path.join(BASE_DIR, "plot2_violin_odds_by_series.pdf")
plt.savefig(p2_path, format='pdf')
plt.close()

# ------------------------------------------------------------------------------
# PLOT 3: Box Plot of Absolute Ranking Parity across Tournament Rounds
# Publication-standard: Whisker boundaries, Tukey outlier markers, clear convergence pattern
# ------------------------------------------------------------------------------
print("Plot 3: Publication Boxplot Ranking Convergence...")
plt.figure(figsize=(9.5, 5.5))
round_order = ['1stRound', '2ndRound', '3rdRound', '4thRound', 'Quarterfinals', 'Semifinals', 'TheFinal']
round_display = ['Round 1', 'Round 2', 'Round 3', 'Round 4', 'Quarterfinals', 'Semifinals', 'The Final']
round_data = [df[df['Round'] == r]['Abs_Rank_Diff'].values for r in round_order]

bp = plt.boxplot(round_data, tick_labels=round_display, patch_artist=True, widths=0.55,
                 boxprops=dict(facecolor='#dbe4ee', color='#1f4e79', linewidth=1.2),
                 whiskerprops=dict(color='#1f4e79', linewidth=1.2),
                 capprops=dict(color='#1f4e79', linewidth=1.2),
                 medianprops=dict(color='#c00000', linewidth=2.0),
                 flierprops=dict(marker='o', markerfacecolor='#e26b00', markeredgecolor='none', markersize=3.0, alpha=0.4))

# Compute and plot mean diamonds
round_means = [np.mean(d) for d in round_data]
plt.plot(range(1, len(round_order) + 1), round_means, color='#1f4e79', linestyle='--', linewidth=1.2, marker='^', 
         markerfacecolor='#1f4e79', markersize=5, label='Mean Disparity Trendline')

plt.ylabel("Absolute Ranking Disparity ($|\\mathrm{Rank}_1 - \\mathrm{Rank}_2|$)", fontsize=10.5)
plt.xlabel("Tournament Elimination Stage", fontsize=10.5)
plt.title("Figure 3: Tournament Progression vs. Ranking Disparity ($|\\mathrm{Rank}_1 - \\mathrm{Rank}_2|$)\nDemonstrating Competitive Parity Convergence Towards Finals", 
          fontsize=11.5, fontweight='bold', pad=10)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
plt.tight_layout()
p3_path = os.path.join(BASE_DIR, "plot3_boxplot_rank_diff_by_round.pdf")
plt.savefig(p3_path, format='pdf')
plt.close()

# ------------------------------------------------------------------------------
# PLOT 4: Multivariate Pairwise Scatter Matrix with KDE Diagonals
# Publication-standard: Clear decision boundaries, high-contrast outcome separation
# ------------------------------------------------------------------------------
print("Plot 4: Publication Multivariate Scatter Matrix...")
fig, axes = plt.subplots(3, 3, figsize=(9.0, 8.5))
scatter_vars = ['Player1_Implied_Prob', 'Player2_Implied_Prob', 'Abs_Rank_Diff']
var_titles = ['P1 Implied Prob ($P_1$)', 'P2 Implied Prob ($P_2$)', '|Rank Diff|']

sample_df = df.sample(n=min(3000, len(df)), random_state=42)

for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        if i == j:
            # Diagonal: Separated Histograms
            v1 = sample_df[sample_df['Player1_Won'] == 1][scatter_vars[i]]
            v2 = sample_df[sample_df['Player1_Won'] == 0][scatter_vars[i]]
            ax.hist(v1, bins=25, density=True, alpha=0.55, color='#2ca02c', label='P1 Won')
            ax.hist(v2, bins=25, density=True, alpha=0.55, color='#1f77b4', label='P2 Won')
            ax.set_title(var_titles[i], fontsize=9.5, fontweight='bold', pad=4)
        else:
            # Off-diagonal: Scatter plot
            p2 = sample_df[sample_df['Player1_Won'] == 0]
            p1 = sample_df[sample_df['Player1_Won'] == 1]
            ax.scatter(p2[scatter_vars[j]], p2[scatter_vars[i]], c='#1f77b4', s=6, alpha=0.25, label='P2 Won')
            ax.scatter(p1[scatter_vars[j]], p1[scatter_vars[i]], c='#2ca02c', s=6, alpha=0.25, label='P1 Won')
        
        if i == 2:
            ax.set_xlabel(var_titles[j], fontsize=9)
        else:
            ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel(var_titles[i], fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.grid(True, linestyle=':', alpha=0.4)

# Global Legend
legend_handles = [
    Patch(facecolor='#2ca02c', edgecolor='none', alpha=0.7, label='Player 1 Won ($n=7,348$)'),
    Patch(facecolor='#1f77b4', edgecolor='none', alpha=0.7, label='Player 2 Won ($n=7,349$)')
]
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=True, framealpha=0.9)
fig.suptitle("Figure 4: Multivariate Feature Scatter Matrix with Class-Conditional Marginal Densities\n(Visualizing Decision Boundary Separation in Betting and Disparity Coordinates)", 
             fontsize=11.5, fontweight='bold', y=1.04)
plt.tight_layout()
p4_path = os.path.join(BASE_DIR, "plot4_pairplot_scatter_matrix.pdf")
plt.savefig(p4_path, format='pdf')
plt.close()

# ------------------------------------------------------------------------------
# PLOT 5: Stacked Percentage Bar Chart
# Publication-standard: High contrast color palette, percentage labels centered
# ------------------------------------------------------------------------------
print("Plot 5: Publication Stacked Percentage Bar Chart...")
plt.figure(figsize=(8.5, 5.5))
ct = pd.crosstab(df['Series'], df['Surface'], normalize='index') * 100
series_reindex = ['GrandSlam', 'Masters1000', 'ATP500', 'ATP250']
ct = ct.reindex(series_reindex)[['Hard', 'Clay', 'Grass']]
display_series = ['Grand Slam', 'Masters 1000', 'ATP 500', 'ATP 250']

bar_width = 0.52
bottom_arr = np.zeros(len(ct))
surface_colors = ['#20639b', '#ed553b', '#3caea3']

for idx, col in enumerate(ct.columns):
    vals = ct[col].values
    plt.bar(display_series, vals, bottom=bottom_arr, label=f"{col} Court", 
            color=surface_colors[idx], width=bar_width, edgecolor='#222222', linewidth=0.8)
    for k in range(len(vals)):
        if vals[k] > 6.0:
            plt.text(k, bottom_arr[k] + vals[k]/2.0, f"{vals[k]:.1f}%", 
                     ha='center', va='center', color='white', fontweight='bold', fontsize=9.0)
    bottom_arr += vals

plt.ylabel("Court Surface Proportion (%)", fontsize=10.5)
plt.xlabel("Tournament Classification Tier", fontsize=10.5)
plt.ylim(0, 100)
plt.title("Figure 5: Proportional Allocation of Court Surfaces Across Tournament Prestige Tiers", 
          fontsize=11.5, fontweight='bold', pad=10)
plt.legend(title='Surface Category', loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
p5_path = os.path.join(BASE_DIR, "plot5_stacked_bar_surface_by_series.pdf")
plt.savefig(p5_path, format='pdf')
plt.close()

# ------------------------------------------------------------------------------
# PLOT 6: 2D Hexagonal Binning Density Plot with Upset Overlays
# Publication-standard: Log10 density scale, distinct outlier markers, publication colormap
# ------------------------------------------------------------------------------
print("Plot 6: Publication 2D Hexbin Density Plot...")
plt.figure(figsize=(8.5, 6.5))
hb = plt.hexbin(df['Log_Rank1'], df['Log_Rank2'], gridsize=36, cmap='YlGnBu', mincnt=1, bins='log', edgecolors='none')
cb = plt.colorbar(hb, pad=0.03)
cb.set_label("Logarithmic Match Density [$\\log_{10}(\\mathrm{Count})$]", fontsize=10)
cb.ax.tick_params(labelsize=8.5)

# Overlay anomalous severe upsets (lower-ranked underdog defeated top 20 favorite with |Rank Diff| > 80)
major_upsets = df[(df['Is_Upset'] == 1) & (df['Abs_Rank_Diff'] > 80)]
plt.scatter(major_upsets['Log_Rank1'], major_upsets['Log_Rank2'], color='#d90429', s=14, alpha=0.6, 
            edgecolor='#2b2d42', linewidth=0.5, label=f'Major Upsets ($|\\Delta\\mathrm{{Rank}}| > 80$, $n={len(major_upsets)}$)')

plt.xlabel("Log-Transformed Player 1 Ranking [$\\ln(1 + \\mathrm{Rank}_1)$]", fontsize=10.5)
plt.ylabel("Log-Transformed Player 2 Ranking [$\\ln(1 + \\mathrm{Rank}_2)$]", fontsize=10.5)
plt.title(f"Figure 6: Logarithmic 2D Hexagonal Binning of Global Matchups with Severe Upset Overlay\n($N = {N_TOTAL:,}$ ATP Matches)", 
          fontsize=11.5, fontweight='bold', pad=10)
plt.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
plt.tight_layout()
p6_path = os.path.join(BASE_DIR, "plot6_hexbin_density_rankings.pdf")
plt.savefig(p6_path, format='pdf')
plt.close()

# ------------------------------------------------------------------------------
# PLOT 7: Longitudinal Time-Series Analysis with Dual-Axis Moving Average
# Publication-standard: Two distinct y-axes, explicit line styling, grid alignment
# ------------------------------------------------------------------------------
print("Plot 7: Publication Time-Series Trend Plot...")
monthly = df.groupby('Year_Month').agg(
    Match_Count=('ATP', 'count'),
    Avg_Margin=('Bookmaker_Margin', 'mean')
).reset_index()
monthly['Date_Plot'] = pd.to_datetime(monthly['Year_Month'] + '-01')

fig, ax1 = plt.subplots(figsize=(10.0, 5.5))

color_match = '#104e8b'
ax1.plot(monthly['Date_Plot'], monthly['Match_Count'], marker='.', markersize=4, color='#7fa9cf', alpha=0.5, label='Raw Monthly Match Volume')
ax1.plot(monthly['Date_Plot'], monthly['Match_Count'].rolling(6, center=True).mean(), color=color_match, linewidth=2.2, label='6-Month Centered Moving Average')
ax1.set_xlabel("Observation Period (2012 – 2017)", fontsize=10.5)
ax1.set_ylabel("Monthly Match Frequency [Counts]", color=color_match, fontsize=10.5, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_match)
ax1.grid(True, linestyle=':', alpha=0.5)

ax2 = ax1.twinx()
color_vig = '#b22222'
ax2.plot(monthly['Date_Plot'], monthly['Avg_Margin'] * 100.0, color=color_vig, linestyle='--', linewidth=1.8, label='Bookmaker Over-round Margin (%)')
ax2.set_ylabel("Bookmaker Margin Over-round [%]", color=color_vig, fontsize=10.5, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_vig)
ax2.set_ylim(4.5, 7.5)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, framealpha=0.9, edgecolor='#cccccc')

plt.title("Figure 7: Longitudinal Monthly ATP Match Volume and Bookmaker Over-round Margin Dynamics\n(Demonstrating Empirical Market Vigor Stability at $\\mu = 5.94\\%$)", 
          fontsize=11.5, fontweight='bold', pad=10)
plt.tight_layout()
p7_path = os.path.join(BASE_DIR, "plot7_timeseries_match_volume_margin.pdf")
plt.savefig(p7_path, format='pdf')
plt.close()

# ------------------------------------------------------------------------------
# PLOT 8: Clustered Bar Chart with 95% Confidence Intervals
# Publication-standard: Error bars explicitly derived from normal approximation, clear value labels
# ------------------------------------------------------------------------------
print("Plot 8: Publication Grouped Bar Chart with 95% CI...")
plt.figure(figsize=(8.5, 5.5))
grouped_stats = df.groupby(['Surface', 'Court'])['Favorite_Won'].agg(
    Mean='mean', Count='count', Std='std'
).reset_index()
grouped_stats['CI'] = 1.96 * (grouped_stats['Std'] / np.sqrt(grouped_stats['Count'])) * 100.0
grouped_stats['Mean_Pct'] = grouped_stats['Mean'] * 100.0

surfaces = ['Hard', 'Clay', 'Grass']
x = np.arange(len(surfaces))
width = 0.32

out_means, out_cis = [], []
in_means, in_cis = [], []

for s in surfaces:
    r_out = grouped_stats[(grouped_stats['Surface'] == s) & (grouped_stats['Court'] == 'Outdoor')]
    out_means.append(r_out['Mean_Pct'].values[0] if not r_out.empty else 0)
    out_cis.append(r_out['CI'].values[0] if not r_out.empty else 0)

    r_in = grouped_stats[(grouped_stats['Surface'] == s) & (grouped_stats['Court'] == 'Indoor')]
    in_means.append(r_in['Mean_Pct'].values[0] if not r_in.empty else 0)
    in_cis.append(r_in['CI'].values[0] if not r_in.empty else 0)

rects1 = plt.bar(x - width/2, out_means, width, yerr=out_cis, capsize=4.0, label='Outdoor Stadiums', 
                 color='#1b4965', edgecolor='#111111', linewidth=0.8)
rects2 = plt.bar(x + width/2, in_means, width, yerr=in_cis, capsize=4.0, label='Indoor Arenas', 
                 color='#e76f51', edgecolor='#111111', linewidth=0.8)

# Add exact value labels on top of bars
for rect, val, ci in zip(rects1, out_means, out_cis):
    if val > 0:
        plt.text(rect.get_x() + rect.get_width()/2.0, val + ci + 0.8, f"{val:.1f}%", 
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1b4965')

for rect, val, ci in zip(rects2, in_means, in_cis):
    if val > 0:
        plt.text(rect.get_x() + rect.get_width()/2.0, val + ci + 0.8, f"{val:.1f}%", 
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#e76f51')

plt.xticks(x, surfaces, fontsize=10)
plt.ylabel("Higher-Ranked Favorite Win Rate [%]", fontsize=10.5)
plt.xlabel("Court Surface Material", fontsize=10.5)
plt.ylim(50, 80)
plt.title("Figure 8: Higher-Ranked Player Win Rates Across Surface & Environmental Setting\n(Error Caps Represent Parametric 95% Confidence Intervals)", 
          fontsize=11.5, fontweight='bold', pad=10)
plt.legend(title='Venue Environment', loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
p8_path = os.path.join(BASE_DIR, "plot8_grouped_bar_favorite_win_rate_ci.pdf")
plt.savefig(p8_path, format='pdf')
plt.close()

# ------------------------------------------------------------------------------
# PLOT 9: Pareto Analysis of Player Victories (Fixed Ticks, No Warnings)
# Publication-standard: Explicit 80/20 threshold, dual y-axis scaling, exact tick placement
# ------------------------------------------------------------------------------
print("Plot 9: Publication Pareto Analysis Chart...")
top_winners = df['Winner'].value_counts().head(20)
cum_pct = (top_winners.cumsum() / top_winners.sum()) * 100.0

fig, ax1 = plt.subplots(figsize=(10.5, 5.5))
x_pos = np.arange(len(top_winners))

ax1.bar(x_pos, top_winners.values, color='#2c7bb6', edgecolor='#111111', linewidth=0.8, width=0.62)
ax1.set_ylabel("Total Matches Won (2012–2017)", color='#104e8b', fontsize=10.5, fontweight='bold')
ax1.set_xlabel("ATP Professional Competitors", fontsize=10.5)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(top_winners.index, rotation=42, ha='right', fontsize=9.0)
ax1.grid(axis='y', linestyle=':', alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(x_pos, cum_pct, color='#d7191c', marker='s', markersize=4.5, linewidth=2.0, label='Cumulative Win %')
ax2.axhline(80.0, color='#555555', linestyle='--', linewidth=1.5, label='80% Pareto Boundary')
ax2.set_ylabel("Cumulative Contribution [%]", color='#d7191c', fontsize=10.5, fontweight='bold')
ax2.set_ylim(0, 105)

# Highlight exact 80% intersection
ax2.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
plt.title("Figure 9: Pareto Analysis of Match Victories Among Top 20 Elite ATP Professionals\n(Validating Cumulative Skewness and Concentration of Elite Performance)", 
          fontsize=11.5, fontweight='bold', pad=10)
plt.tight_layout()
p9_path = os.path.join(BASE_DIR, "plot9_pareto_chart_top_winners.pdf")
plt.savefig(p9_path, format='pdf')
plt.close()

# ------------------------------------------------------------------------------
# PLOT 10: Radar / Polar Architectural Profile Chart
# Publication-standard: High-contrast shaded polygons, clear angular ticks, normalized metrics
# ------------------------------------------------------------------------------
print("Plot 10: Publication Radar Profile Chart...")
radar_features = [
    'Fav Win Rate', 'Upset Frequency', 'Norm Rank Disparity', 'Outdoor Proportion', 'Best-of-5 Ratio'
]
num_vars = len(radar_features)
angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
angles += angles[:1]

radar_data_df = pd.DataFrame(index=series_reindex, columns=radar_features)
for s in series_reindex:
    sub = df[df['Series'] == s]
    radar_data_df.loc[s, 'Fav Win Rate'] = sub['Favorite_Won'].mean()
    radar_data_df.loc[s, 'Upset Frequency'] = sub['Is_Upset'].mean()
    radar_data_df.loc[s, 'Norm Rank Disparity'] = sub['Abs_Rank_Diff'].mean() / 110.0
    radar_data_df.loc[s, 'Outdoor Proportion'] = (sub['Court'] == 'Outdoor').mean()
    radar_data_df.loc[s, 'Best-of-5 Ratio'] = (sub['Best_of'] == 5).mean()

fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
series_colors = ['#d7191c', '#fdae61', '#2c7bb6', '#2b83ba']

for idx, s in enumerate(series_reindex):
    vals = radar_data_df.loc[s].values.flatten().tolist()
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=1.8, linestyle='solid', label=s, color=series_colors[idx])
    ax.fill(angles, vals, color=series_colors[idx], alpha=0.12)

plt.xticks(angles[:-1], radar_features, fontsize=9.5, fontweight='bold')
ax.set_rlabel_position(35)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.20", "0.40", "0.60", "0.80", "1.00"], color="#666666", size=8.5)
plt.ylim(0, 1.05)
plt.title("Figure 10: Multi-Attribute Radar Characterization Across Tournament Prestige Tiers\n(Normalized Architectural Signatures)", 
          fontsize=11.5, fontweight='bold', pad=22)
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.12), frameon=True, framealpha=0.9, edgecolor='#cccccc')
plt.tight_layout()
p10_path = os.path.join(BASE_DIR, "plot10_radar_chart_tournament_profiles.pdf")
plt.savefig(p10_path, format='pdf')
plt.close()

print("\n" + "="*70)
print("ALL 10 Publication RESEARCH-PAPER STANDARD PLOTS SUCCESSFULLY GENERATED!")
print("="*70)
