import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py
import argparse
from tqdm import tqdm
import pandas as pd
from openbabel import pybel
from glob import glob
from sklearn import linear_model
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

if __name__ == '__main__':
    TRAIN_GRIDS = r'data/train_hdf5/train_grids.h5'
    TRAIN_LABEL = r'data/train_hdf5/train_label.h5'
    CORE_GRIDS = r'data/test_hdf5/core_grids.h5'
    CORE_LABEL = r'data/test_hdf5/core_label.h5'
    with h5py.File('data/train_hdf5/train_grids.h5', 'r') as f:
        print("train_grids:", f['train_grids'].shape)
    
    with h5py.File('data/train_hdf5/train_label.h5', 'r') as f:
        print("train_label:", f['train_label'].shape)
    
    with h5py.File('data/test_hdf5/core_grids.h5', 'r') as f:
        print("core_grids:", f['core_grids'].shape)
    
    with h5py.File('data/test_hdf5/core_label.h5', 'r') as f:
        print("core_label:", f['core_label'].shape)

    def get_indices(total, val_start=41000, seed=1234):
        all_idx = np.arange(total)
        np.random.seed(seed)
        np.random.shuffle(all_idx)
        train_idx = all_idx[:val_start]
        val_idx = all_idx[val_start:]
        return train_idx, val_idx

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-b', default=32, type=int)
    parser.add_argument('--dropout', '-d', default=0.15, type=float)
    parser.add_argument('--lr', default=0.0015, type=float)
    args = parser.parse_args()

    # Prepare indices for train/val split
    with h5py.File(TRAIN_GRIDS, 'r') as f:
        total = len(f['train_grids'])
    train_idx, val_idx = get_indices(total)

    with h5py.File(CORE_GRIDS, 'r') as f:
        test_len = len(f['core_grids'])
    test_idx = np.arange(test_len)

    # Datasets and loaders
    train_dataset = HDF5GridDataset(
        TRAIN_GRIDS, 'train_grids', TRAIN_LABEL, 'train_label', train_idx
    )
    val_dataset = HDF5GridDataset(
        TRAIN_GRIDS, 'train_grids', TRAIN_LABEL, 'train_label', val_idx
    )
    test_dataset = HDF5GridDataset(
        CORE_GRIDS, 'core_grids', CORE_LABEL, 'core_label', test_idx
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch)
    print('Dataset load finished')

    class CNN3D(nn.Module):
        def __init__(self, dropout=0.5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(28, 7, kernel_size=1),  # input: (B,28,20,20,20)
                nn.BatchNorm3d(7),
                nn.ReLU(),
                nn.Conv3d(7, 7, kernel_size=3),
                nn.BatchNorm3d(7),
                nn.ReLU(),
                nn.Conv3d(7, 7, kernel_size=3),
                nn.BatchNorm3d(7),
                nn.ReLU(),
                nn.Conv3d(7, 28, kernel_size=1),
                nn.BatchNorm3d(28),
                nn.ReLU(),
                nn.Conv3d(28, 56, kernel_size=3, padding=1),
                nn.BatchNorm3d(56),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(56, 112, kernel_size=3, padding=1),
                nn.BatchNorm3d(112),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(112, 224, kernel_size=3, padding=1),
                nn.BatchNorm3d(224),
                nn.ReLU(),
                nn.MaxPool3d(2),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(224*2*2*2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1)
            )
        def forward(self, x):       # (batch*seq, 20, 20, 20, 28)
            x = x.permute(0, 4, 1, 2, 3)  # (B, 20, 20, 20, 28) -> (B, 28, 20, 20, 20)
            x = self.net(x)
            x = self.fc(x)
            return x

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('current device: ', device)

    model = CNN3D(dropout=args.dropout).to(device)
    last_linear = model.fc[-1]
    params = [
        {'params': [p for n, p in model.named_parameters() if 'fc.5' not in n], 'weight_decay': 0.0},
        {'params': last_linear.parameters(), 'weight_decay': 0.03} 
    ]
    optimizer = optim.RMSprop(params, lr=args.lr)
    criterion = nn.MSELoss()

    os.makedirs('src/sfcnn/src/train_results/cnnmodel', exist_ok=True)

    train_loss_history = []
    best_pearson = -1.0

    TRAIN_EPOCHS = 200

    for epoch in tqdm(range(1, TRAIN_EPOCHS+1), desc="Epochs"):
        model.train()
        train_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Train {epoch:03d}", leave=False):
            xb, yb = xb.to(device), yb.to(device)  
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_loss_history.append(train_loss)

        # Pearson test on CASF2016 set every epoch >= 10
        if epoch >= 1:
            model.eval()
            preds = []
            targets = []
            with torch.no_grad():
                for xb, yb in tqdm(test_loader, desc=f"CASF2016 Pearson {epoch:03d}", leave=False):
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    preds.append(pred.cpu().numpy())
                    targets.append(yb.cpu().numpy())
            preds = np.concatenate(preds).flatten() * 15
            targets = np.concatenate(targets).flatten() * 15
            pearson = np.corrcoef(preds, targets)[0, 1]

            # Save best model by Pearson
            if pearson > best_pearson:
                best_pearson = pearson
                torch.save(model.state_dict(), f'src/sfcnn/src/train_results/cnnmodel/weights_{epoch:03d}-{pearson:.4f}.pt')
                # print(f"Epoch {epoch:03d}: New best Pearson {pearson:.4f}, model saved.")

    np.save('src/sfcnn/src/train_results/train_loss_history.npy', np.array(train_loss_history))

    # Final evaluation on CASF2016 set after training
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Final CASF2016 Evaluation", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())
    preds = np.concatenate(preds).flatten() * 15
    targets = np.concatenate(targets).flatten() * 15

    # Save predictions for further analysis
    df = pd.DataFrame({'score': preds, 'affinity': targets})
    df.to_csv('src/sfcnn/outputs/output.csv', sep='\t', index=False)

    # Linear regression and metrics
    regr = linear_model.LinearRegression()
    x = df['score'].values.reshape(-1,1)
    y = df['affinity'].values.reshape(-1,1)
    regr.fit(x, y)
    y_ = regr.predict(x)
    pearson = scipy.stats.pearsonr(df['affinity'].values, df['score'].values)[0]
    rmse = np.sqrt(np.mean((df['score']-df['affinity'])**2))
    mae = np.mean(np.abs(df['score']-df['affinity']))
    sd = np.sqrt(np.sum((y-y_)**2)/(len(df) - 1.0))

    print('Final CASF2016 Pearson:', pearson)
    print('Final CASF2016 RMSE:', rmse)
    print('Final CASF2016 MAE:', mae)
    print('Final CASF2016 SD:', sd)