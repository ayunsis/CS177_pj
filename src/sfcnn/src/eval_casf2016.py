import predict
import torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import linear_model
from train import HDF5GridDataset
import scipy

MODEL_PATH = 'src/sfcnn/src/train_results/valuable_models/pearson-0.728.pt'


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
out_df = pd.DataFrame({'score': preds})
out_df = out_df.drop([84, 95, 67, 68, 69, 25]).reset_index(drop=True)
out_df.to_csv('data/sfcnn_out.csv', sep='\t', index=False)
df.to_csv('src/sfcnn/outputs/output.csv', sep='\t', index=False)

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