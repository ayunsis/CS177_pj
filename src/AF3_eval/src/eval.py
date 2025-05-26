import predict
import torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import linear_model
from torch.utils.data import Dataset, DataLoader
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

class HDF5GridDataset(Dataset):
    def __init__(self, h5_path, data_key, label_path=None, label_key=None, indices=None, normalize_y=15.0):
        self.h5_path = h5_path
        self.data_key = data_key
        self.label_path = label_path
        self.label_key = label_key
        self.indices = indices
        self.normalize_y = normalize_y
        self.length = None
        self.h5_file = None
        self.label_file = None
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f[self.data_key]) if indices is None else len(indices)

    def __len__(self):
        return self.length

    def _ensure_open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        if self.label_path and self.label_file is None:
            self.label_file = h5py.File(self.label_path, 'r')

    def __getitem__(self, idx):
        self._ensure_open()
        real_idx = self.indices[idx] if self.indices is not None else idx
        grid = torch.tensor(self.h5_file[self.data_key][real_idx], dtype=torch.float32)
        if self.label_path is not None and self.label_key is not None:
            label = torch.tensor(self.label_file[self.label_key][real_idx], dtype=torch.float32).unsqueeze(0)
            label = label / self.normalize_y
            return grid, label
        else:
            return grid
        

MODEL_PATH = 'src/sfcnn/src/train_results/cnnmodel/best_overall_weights.pt'
CURRENT_BEST = 'src/AF3_eval/model/pearson-0.7686.pt'
CORE_GRIDS = r'data/chai_hdf5/core_grids.h5'
CORE_2016_LABEL = r'data/chai_hdf5/core_2016_label.h5'
CORE_sfcnn_LABEL = r'data/chai_hdf5/core_sfcnn_label.h5'

with h5py.File(CORE_GRIDS, 'r') as f:
    test_len = len(f['core_grids'])
test_idx = np.arange(test_len)


model = predict.build_model(CURRENT_BEST, dropout=0.15)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
core_csv = pd.read_csv('data/core_affinity_final.csv')

def evaluate(label_path, label_key, output_csv, desc):
    test_dataset = HDF5GridDataset(
        CORE_GRIDS, 'core_grids', label_path, label_key, test_idx
    )
    test_loader = DataLoader(test_dataset, batch_size=32)

    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc=desc):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy() * 15)
            targets.append(yb.cpu().numpy() * 15)
    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten() 

    df = pd.DataFrame({'pdbid':core_csv['pdbid'], 'score': preds, 'affinity': targets})
    df.to_csv(output_csv, sep=',', index=False)

    regr = linear_model.LinearRegression()
    x = df['score'].values.reshape(-1,1)
    y = df['affinity'].values.reshape(-1,1)
    regr.fit(x, y)
    y_ = regr.predict(x)
    pearson = scipy.stats.pearsonr(df['affinity'].values, df['score'].values)[0]
    rmse = np.sqrt(np.mean((df['score']-df['affinity'])**2))
    mae = np.mean(np.abs(df['score']-df['affinity']))
    sd = np.sqrt(np.sum((y-y_)**2)/(len(df) - 1.0))
    
    return pearson, rmse, mae, sd, df


casf2016_metrics = evaluate(CORE_2016_LABEL, 'core_label', 'src/AF3_eval/outputs/output_core2016.csv', 'CASF2016')
sfcnn_metrics = evaluate(CORE_sfcnn_LABEL, 'core_label', 'src/AF3_eval/outputs/output_sfcnn.csv', 'SFCNN')

# Extract metrics
casf2016_pearson, casf2016_rmse, casf2016_mae, casf2016_sd, casf2016_df = casf2016_metrics
sfcnn_pearson, sfcnn_rmse, sfcnn_mae, sfcnn_sd, sfcnn_df = sfcnn_metrics

# Create comparison plot
metrics = ['Pearson', 'RMSE', 'MAE', 'SD']
casf2016_results = [casf2016_pearson, casf2016_rmse, casf2016_mae, casf2016_sd]
sfcnn_results = [sfcnn_pearson, sfcnn_rmse, sfcnn_mae, sfcnn_sd]

compare_df = pd.DataFrame({'CASF2016': casf2016_results, 'AF3': sfcnn_results}, index=metrics)
print('\nMetrics Comparison:')
print(compare_df)

# Bar plot for visual comparison
plt.figure(figsize=(8, 5))
compare_df_reset = compare_df.reset_index().melt(id_vars='index', var_name='Dataset', value_name='Value')
sns.barplot(data=compare_df_reset, x='index', y='Value', hue='Dataset', palette='viridis')
plt.title('Metrics: SFCNN CASF2016 and SFCNN AF3 vs. Ground Truth')
plt.ylabel('Metric Value')
plt.xlabel('Metric')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/metrics_comparison.png', dpi=300)
plt.show()

# Scatter plots for both datasets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# CASF2016 scatter plot
sns.scatterplot(x=casf2016_df['affinity'], y=casf2016_df['score'], alpha=0.7, s=60, edgecolor='k', ax=ax1)
sns.regplot(x=casf2016_df['affinity'], y=casf2016_df['score'], scatter=False, color='red', ax=ax1)
ax1.plot([casf2016_df['affinity'].min(), casf2016_df['affinity'].max()],
         [casf2016_df['affinity'].min(), casf2016_df['affinity'].max()],
         'k--', lw=2, label='Ideal')
ax1.set_xlabel('Ground Truth Affinity')
ax1.set_ylabel('Predicted Score')
ax1.set_title(f'CASF2016\nPearson={casf2016_pearson:.3f}, RMSE={casf2016_rmse:.3f}')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.8)

# SFCNN scatter plot
sns.scatterplot(x=sfcnn_df['affinity'], y=sfcnn_df['score'], alpha=0.7, s=60, edgecolor='k', ax=ax2)
sns.regplot(x=sfcnn_df['affinity'], y=sfcnn_df['score'], scatter=False, color='red', ax=ax2)
ax2.plot([sfcnn_df['affinity'].min(), sfcnn_df['affinity'].max()],
         [sfcnn_df['affinity'].min(), sfcnn_df['affinity'].max()],
         'k--', lw=2, label='Ideal')
ax2.set_xlabel('Ground Truth Affinity')
ax2.set_ylabel('Predicted Score')
ax2.set_title(f'SFCNN\nPearson={sfcnn_pearson:.3f}, RMSE={sfcnn_rmse:.3f}')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.8)

plt.tight_layout()
plt.savefig('src/AF3_eval/outputs/scatter_comparison.png', dpi=300)
plt.show()

