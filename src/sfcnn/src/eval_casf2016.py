import predict
import torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import linear_model
from train import HDF5GridDataset
import matplotlib.pyplot as plt
import scipy

MODEL_PATH = 'src/sfcnn/src/train_results/cnnmodel/best_overall_weights.pt'


CORE_GRIDS = r'data/test_hdf5/core_grids.h5'
CORE_LABEL = r'data/test_hdf5/core_label.h5'
with h5py.File(CORE_GRIDS, 'r') as f:
    test_len = len(f['core_grids'])
test_idx = np.arange(test_len)

test_dataset = HDF5GridDataset(
    CORE_GRIDS, 'core_grids', CORE_LABEL, 'core_label', test_idx
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

model = predict.build_model(MODEL_PATH, dropout=0.15)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
core_csv = pd.read_csv('data/core_affinity_2016.csv')

preds = []
targets = []
with torch.no_grad():
    for xb, yb in tqdm(test_loader, desc="CASF2016 Evaluation"):
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        preds.append(pred.cpu().numpy()* 15)
        targets.append(yb.cpu().numpy()* 15)
preds = np.concatenate(preds ).flatten()
targets = np.concatenate(targets).flatten() 

df = pd.DataFrame({'score': preds, 'affinity': targets})
out_df = pd.DataFrame({'pdbid': core_csv['pdbid'],
                       'affinity': preds})

pdbids_to_drop = ['2xb8', '2ymd', '3n76', '3n7a', '3n86', '4ciw']
indices_to_drop = core_csv[core_csv['pdbid'].isin(pdbids_to_drop)].index.tolist()
out_df = out_df.drop(indices_to_drop).reset_index(drop=True)
core_csv = core_csv.drop(indices_to_drop).reset_index(drop=True)

core_csv.to_csv('data/core_affinity_final.csv', sep=',', index=False)
out_df.to_csv('data/sfcnn_out.csv', sep=',', index=False)
df.to_csv('src/sfcnn/outputs/output.csv', sep=',', index=False)

regr = linear_model.LinearRegression()
x = df['score'].values.reshape(-1,1)
y = df['affinity'].values.reshape(-1,1)
regr.fit(x, y)
y_ = regr.predict(x)
pearson = scipy.stats.pearsonr(df['affinity'].values, df['score'].values)[0]
rmse = np.sqrt(np.mean((df['score']-df['affinity'])**2))
mae = np.mean(np.abs(df['score']-df['affinity']))
sd = np.sqrt(np.sum((y-y_)**2)/(len(df) - 1.0))

print('CASF2016 Pearson:', pearson)
print('CASF2016 RMSE:', rmse)
print('CASF2016 MAE:', mae)
print('CASF2016 SD:', sd)


import seaborn as sns

tf_pearson = 0.792789
tf_rmse = 1.3262567334485227
tf_mae = 1.027727719298246
tf_sd = 1.32525068


pt_pearson = pearson
pt_rmse = rmse
pt_mae = mae
pt_sd = sd

metrics = ['Pearson', 'RMSE', 'MAE', 'SD']
tf_results = [tf_pearson, tf_rmse, tf_mae, tf_sd]
pt_results = [pt_pearson, pt_rmse, pt_mae, pt_sd]
compare_df = pd.DataFrame({'TensorFlow': tf_results, 'PyTorch': pt_results}, index=metrics)
print('\nReproduction Metrics Comparison:')
print(compare_df)

# Bar plot for visual comparison
plt.figure(figsize=(8, 5))
compare_df_reset = compare_df.reset_index().melt(id_vars='index', var_name='Framework', value_name='Value')
sns.barplot(data=compare_df_reset, x='index', y='Value', hue='Framework', palette='viridis')
plt.title('Reproduction Metrics: TensorFlow vs PyTorch')
plt.ylabel('Metric Value')
plt.xlabel('Metric')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('src/sfcnn/outputs/reproduction_metrics_comparison_seaborn.png', dpi=300)
plt.show()

plt.figure(figsize=(7, 7))
sns.scatterplot(x=df['affinity'], y=df['score'], alpha=0.7, s=60, edgecolor='k')
sns.regplot(x=df['affinity'], y=df['score'], scatter=False, color='red', label='Fit')
plt.plot([df['affinity'].min(), df['affinity'].max()],
         [df['affinity'].min(), df['affinity'].max()],
         'k--', lw=2, label='Ideal')
plt.xlabel('Ground Truth Affinity')
plt.ylabel('Predicted Score')
plt.title(f'CASF2016: PyTorch Reproduction\nPearson={pearson:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.8)
plt.tight_layout()
plt.savefig('src/sfcnn/outputs/reproduction_scatter_seaborn.png', dpi=300)
plt.show()

residuals = df['score'] - df['affinity']
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color='orange', edgecolor='black')
plt.title('Residuals (Predicted - Ground Truth)')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.8)
plt.tight_layout()
plt.savefig('src/sfcnn/outputs/reproduction_residuals_seaborn.png', dpi=300)
plt.show()

abs_error = np.abs(df['score'] - df['affinity'])
plt.figure(figsize=(8, 5))
sns.histplot(abs_error, bins=30, kde=True, color='purple', edgecolor='black')
plt.title('Absolute Error Distribution')
plt.xlabel('Absolute Error')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.8)
plt.tight_layout()
plt.savefig('src/sfcnn/outputs/reproduction_abs_error_seaborn.png', dpi=300)
plt.show()