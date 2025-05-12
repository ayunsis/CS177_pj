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



plt.figure(figsize=(10, 6))
plt.plot(core2016_df['gap'], label='core2016 gap', marker='o', markersize=2, linewidth=1.2, alpha=0.8)
plt.plot(sfcnn_df['gap'], label='sfcnn gap', marker='o', markersize=2, linewidth=1.2, alpha=0.8)
plt.title('Groundtruth/Sfcnn Gap Comparison Across Entries')
plt.xlabel('Entry Index')
plt.ylabel('Gap')
plt.axhline(y=sfcnn_df['gap'].mean(), color='purple', linestyle='--', linewidth=1.2, label='sfcnn mean gap')
plt.axhline(y=core2016_df['gap'].mean(), color='blue', linestyle='--', linewidth=1.2, label='core2016 mean gap')
plt.legend()
plt.xticks()
plt.grid()
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/img/gap_comparison.png', dpi=300)

plt.figure(figsize=(10, 6))
plt.plot(core2016_df['gap'], label='core2016 gap', marker='o', markersize=2, linewidth=1.2, alpha=0.8)
plt.plot(sfcnn_core_gap, label='sfcnn/groundtruth gap', marker='o', markersize=2, linewidth=1.2, alpha=0.8)
plt.title('Sfcnn difference Comparison Across Entries')
plt.xlabel('Entry Index')
plt.ylabel('Gap')
plt.axhline(y=sfcnn_core_gap.mean(), color='purple', linestyle='--', linewidth=1.2, label='sfcnn/groundtruth mean gap')
plt.axhline(y=core2016_df['gap'].mean(), color='blue', linestyle='--', linewidth=1.2, label='core2016 mean gap')
plt.legend()
plt.xticks()
plt.grid()
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/img/sfcnn_diff_comparison.png', dpi=300)



top20_idx = core2016_df['gap'].nlargest(20).index
heatmap_data = pd.DataFrame({
    'core2016_gap': core2016_df.loc[top20_idx, 'gap'].values,
    'sfcnn_gap': sfcnn_df.loc[top20_idx, 'gap'].values,
    'gap_diff': sfcnn_core_gap.loc[top20_idx].values,
}, index=core2016_df.loc[top20_idx, 'pdbid'])
heatmap_data = heatmap_data.sort_values(by='core2016_gap', ascending=False)

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='plasma', fmt=".2f", alpha=0.8)
plt.title('Top 20 Groundtruth/Sfcnn difference (core2016 vs sfcnn)')
plt.xlabel('Metrics')
plt.xticks()
plt.ylabel('PDB ID')
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/img/top20_heatmap.png', dpi=300)


plt.figure(figsize=(10, 6))
sns.histplot(core2016_df['gap'], bins=30, color='blue', label='core2016 gap', kde=True, stat='density', alpha=0.5, edgecolor='black')
sns.histplot(sfcnn_df['gap'], bins=30, color='purple', label='sfcnn gap', kde=True, stat='density', alpha=0.5, edgecolor='black')
plt.title('Histogram of Gaps')
plt.grid()
plt.xlabel('Gap')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/img/gap_histogram.png', dpi=300)


plt.figure(figsize=(10, 6))
gap_df = pd.DataFrame({
    'core2016 gap': core2016_df['gap'],
    'sfcnn gap': sfcnn_df['gap']
})
sns.boxplot(data=gap_df, palette='plasma')
plt.title('Boxplot of Gaps')
plt.ylabel('Gap')
plt.grid()
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/img/gap_boxplot.png', dpi=300)


plt.figure(figsize=(10, 6))
sns.regplot(x=core2016_df['gap'], y=sfcnn_df['gap'], scatter_kws={'alpha':0.6})
corr = np.corrcoef(core2016_df['gap'], sfcnn_df['gap'])[0, 1]
plt.title(f'Correlation of Gaps (r={corr:.2f})')
plt.xlabel('core2016 gap')
plt.ylabel('sfcnn gap')
plt.grid()
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/img/gap_correlation.png', dpi=300)

plt.figure(figsize=(10, 6))
sns.violinplot(data=gap_df, palette='plasma', alpha=0.6)
plt.title('Violin Plot of Gaps')
plt.ylabel('Gap')
plt.grid()
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/img/gap_violinplot.png', dpi=300)

means = (core2016_df['gap'] + sfcnn_df['gap']) / 2
diffs = core2016_df['gap'] - sfcnn_df['gap']
mean_diff = diffs.mean()
std_diff = diffs.std()

plt.figure(figsize=(10, 6))
plt.scatter(means, diffs, alpha=0.6)
plt.axhline(mean_diff, color='red', linestyle='--', label='Mean difference')
plt.axhline(mean_diff + 1.96*std_diff, color='purple', linestyle='--', label='±1.96 SD')
plt.axhline(mean_diff - 1.96*std_diff, color='purple', linestyle='--')
plt.title('Bland-Altman Plot')
plt.xlabel('Mean Gap')
plt.ylabel('Difference in Gap')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/img/bland_altman.png', dpi=300)
