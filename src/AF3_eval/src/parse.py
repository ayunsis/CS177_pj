import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

core2016_df= pd.read_csv('src\AF3_eval\outputs\output_core2016.csv')
sfcnn_df = pd.read_csv('src\AF3_eval\outputs\output_sfcnn.csv')


core2016_df['gap'] = (core2016_df['score'] - core2016_df['affinity']).abs()
sfcnn_df['gap'] = (sfcnn_df['score'] - sfcnn_df['affinity']).abs()
core2016_df['gap'] = core2016_df['gap'].round(2)
sfcnn_df['gap'] = sfcnn_df['gap'].round(2)
sfcnn_core_gap = (core2016_df['gap'] - sfcnn_df['gap']).abs()



sfcnn_core_gap = sfcnn_core_gap.round(2)

# Set a consistent color palette
core_color = '#00BFFF'   # blue
sfcnn_color = '#FFA500'  # yellow-orange
diff_color = '#FF6347'   # red
mean_linewidth = 2.0

plt.figure(figsize=(12, 7))
plt.plot(core2016_df['gap'], label='Core2016 Gap', marker='o', markersize=3, linewidth=1.5, alpha=0.85, color=core_color)
plt.plot(sfcnn_df['gap'], label='SFCNN Gap', marker='s', markersize=3, linewidth=1.5, alpha=0.85, color=sfcnn_color)
plt.title('Comparison of Absolute Prediction Gaps\n(Core2016 vs SFCNN)', fontsize=17)
plt.xlabel('Entry Index', fontsize=13)
plt.ylabel('Absolute Gap (|score - affinity|)', fontsize=13)
plt.axhline(y=sfcnn_df['gap'].mean(), color=sfcnn_color, linestyle='--', linewidth=mean_linewidth, label=f'SFCNN Mean Gap ({sfcnn_df["gap"].mean():.2f})')
plt.axhline(y=core2016_df['gap'].mean(), color=core_color, linestyle='--', linewidth=mean_linewidth, label=f'Core2016 Mean Gap ({core2016_df["gap"].mean():.2f})')
plt.legend(fontsize=12, loc='upper right')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/img/gap_comparison.png', dpi=300)

plt.figure(figsize=(12, 7))
plt.plot(core2016_df['gap'], label='Core2016 Gap', marker='o', markersize=3, linewidth=1.5, alpha=0.85, color=core_color)
plt.plot(sfcnn_core_gap, label='|Core2016 Gap - SFCNN Gap|', marker='^', markersize=3, linewidth=1.5, alpha=0.85, color=diff_color)
plt.title('Difference Between Core2016 and SFCNN Gaps Across Entries', fontsize=17)
plt.xlabel('Entry Index', fontsize=13)
plt.ylabel('Gap Difference', fontsize=13)
plt.axhline(y=sfcnn_core_gap.mean(), color=diff_color, linestyle='--', linewidth=mean_linewidth, label=f'Mean Gap Difference ({sfcnn_core_gap.mean():.2f})')
plt.axhline(y=core2016_df['gap'].mean(), color=core_color, linestyle='--', linewidth=mean_linewidth, label=f'Core2016 Mean Gap ({core2016_df["gap"].mean():.2f})')
plt.legend(fontsize=12, loc='upper right')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/img/sfcnn_diff_comparison.png', dpi=300)



top20_idx = sfcnn_core_gap.nlargest(20).index
heatmap_data = pd.DataFrame({
    'core2016_gap': core2016_df.loc[top20_idx, 'gap'].values,
    'sfcnn_gap': sfcnn_df.loc[top20_idx, 'gap'].values,
    'gap_diff': sfcnn_core_gap.loc[top20_idx].values,
}, index=core2016_df.loc[top20_idx, 'pdbid'])
heatmap_data = heatmap_data.sort_values(by='core2016_gap', ascending=False)

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap='plasma', fmt=".2f", alpha=0.85, linewidths=0.5, linecolor='white', cbar_kws={'label': 'Gap Value'})
plt.title('Top 20 Entries with Largest Core2016 Gaps\nComparison with SFCNN and Gap Difference', fontsize=16)
plt.xlabel('Metric', fontsize=13)
plt.ylabel('PDB ID', fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/img/top20_heatmap.png', dpi=300)


plt.figure(figsize=(12, 7))
sns.histplot(core2016_df['gap'], bins=30, color=core_color, label='Core2016 Gap', kde=True, stat='density', alpha=0.6, edgecolor='black')
sns.histplot(sfcnn_df['gap'], bins=30, color=sfcnn_color, label='SFCNN Gap', kde=True, stat='density', alpha=0.6, edgecolor='black')
plt.title('Distribution of Absolute Gaps\n(Core2016 vs SFCNN)', fontsize=16)
plt.xlabel('Absolute Gap', fontsize=13)
plt.ylabel('Density', fontsize=13)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/img/gap_histogram.png', dpi=300)


plt.figure(figsize=(10, 7))
gap_df = pd.DataFrame({
    'Core2016 Gap': core2016_df['gap'],
    'SFCNN Gap': sfcnn_df['gap']
})
sns.boxplot(data=gap_df, palette=[core_color, sfcnn_color])
plt.title('Boxplot of Absolute Gaps', fontsize=15)
plt.ylabel('Absolute Gap', fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.8)
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/img/gap_boxplot.png', dpi=300)


plt.figure(figsize=(10, 7))
sns.regplot(x=core2016_df['gap'], y=sfcnn_df['gap'], scatter_kws={'alpha':0.7, 'color':core_color}, line_kws={'color':sfcnn_color, 'linewidth':2})
corr = np.corrcoef(core2016_df['gap'], sfcnn_df['gap'])[0, 1]
plt.title(f'Correlation of Gaps Between Core2016 and SFCNN\nPearson r = {corr:.2f}', fontsize=15)
plt.xlabel('Core2016 Gap', fontsize=13)
plt.ylabel('SFCNN Gap', fontsize=13)
plt.grid(True, linestyle=':', alpha=0.8)
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/img/gap_correlation.png', dpi=300)

plt.figure(figsize=(10, 7))
sns.violinplot(data=gap_df, palette=[core_color, sfcnn_color], alpha=0.5)
plt.title('Violin Plot of Absolute Gaps', fontsize=15)
plt.ylabel('Absolute Gap', fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.8)
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/img/gap_violinplot.png', dpi=300)


