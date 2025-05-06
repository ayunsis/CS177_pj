import predict
import torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import linear_model
from torch.utils.data import Dataset, DataLoader
import scipy

class HDF5GridDataset(Dataset):
    def __init__(self, h5_path, data_key, label_path=None, label_key=None, indices=None, normalize_y=15.0):
        self.h5_path = h5_path
        self.data_key = data_key
        self.label_path = label_path
        self.label_key = label_key
        self.indices = indices
        self.normalize_y = normalize_y
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f[self.data_key]) if indices is None else len(indices)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        with h5py.File(self.h5_path, 'r') as f:
            grid = torch.tensor(f[self.data_key][real_idx], dtype=torch.float32)

        if self.label_path is not None and self.label_key is not None:
            with h5py.File(self.label_path, 'r') as f:
                label = torch.tensor(f[self.label_key][real_idx], dtype=torch.float32).unsqueeze(0)
                label = label / self.normalize_y
            return grid, label
        else:
            return grid
        

MODEL_PATH = 'src/AF3_eval/model/pearson-0.728.pt'
CORE_GRIDS = r'data/chai_hdf5/core_grids.h5'
CORE_2016_LABEL = r'data/chai_hdf5/core_2016_label.h5'
CORE_sfcnn_LABEL = r'data/chai_hdf5/core_sfcnn_label.h5'

with h5py.File(CORE_GRIDS, 'r') as f:
    test_len = len(f['core_grids'])
test_idx = np.arange(test_len)


model = predict.build_model(MODEL_PATH, dropout=0.15)
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

    print(f'{desc} Pearson:', pearson)
    print(f'{desc} RMSE:', rmse)
    print(f'{desc} MAE:', mae)
    print(f'{desc} SD:', sd)


evaluate(CORE_2016_LABEL, 'core_label', 'src/AF3_eval/outputs/output_core2016.csv', 'CASF2016')
evaluate(CORE_sfcnn_LABEL, 'core_label', 'src/AF3_eval/outputs/output_sfcnn.csv', 'SFCNN')

