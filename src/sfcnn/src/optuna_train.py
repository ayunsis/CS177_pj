import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import h5py
from sklearn import linear_model
import optuna
from tqdm import tqdm

# --- Dataset class (copied from train.py) ---
class HDF5GridDataset(torch.utils.data.Dataset):
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

# --- Model class (copied from train.py) ---
class CNN3D(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(28, 7, kernel_size=1),
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
    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = self.net(x)
        x = self.fc(x)
        return x

# --- Data preparation (copied from train.py) ---
TRAIN_GRIDS = r'data/train_hdf5/train_grids.h5'
TRAIN_LABEL = r'data/train_hdf5/train_label.h5'
CORE_GRIDS = r'data/test_hdf5/core_grids.h5'
CORE_LABEL = r'data/test_hdf5/core_label.h5'

def get_indices(total, val_start=41000, seed=1234):
    all_idx = np.arange(total)
    np.random.seed(seed)
    np.random.shuffle(all_idx)
    train_idx = all_idx[:val_start]
    val_idx = all_idx[val_start:]
    return train_idx, val_idx

with h5py.File(TRAIN_GRIDS, 'r') as f:
    total = len(f['train_grids'])
train_idx, val_idx = get_indices(total)

with h5py.File(CORE_GRIDS, 'r') as f:
    test_len = len(f['core_grids'])
test_idx = np.arange(test_len)

train_dataset = HDF5GridDataset(
    TRAIN_GRIDS, 'train_grids', TRAIN_LABEL, 'train_label', train_idx
)
val_dataset = HDF5GridDataset(
    TRAIN_GRIDS, 'train_grids', TRAIN_LABEL, 'train_label', val_idx
)
test_dataset = HDF5GridDataset(
    CORE_GRIDS, 'core_grids', CORE_LABEL, 'core_label', test_idx
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def objective(trial):
    batch = trial.suggest_categorical('batch', [16, 32, 64])
    dropout = trial.suggest_float('dropout', 0.05, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-4)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch, pin_memory=True, num_workers=8)

    model = CNN3D(dropout=dropout).to(device)
    last_linear = model.fc[-1]
    params = [
        {'params': [p for n, p in model.named_parameters() if 'fc.5' not in n], 'weight_decay': 0.0},
        {'params': last_linear.parameters(), 'weight_decay': 0.01}
    ]
    optimizer = optim.RMSprop(params, lr=lr)
    criterion = nn.MSELoss()

    best_test_pearson = -1.0
    EPOCHS = 50  # Use fewer epochs for tuning

    for epoch in range(1, EPOCHS+1):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for xb, yb in tqdm(test_loader, desc=f"Epoch {epoch} Testing", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                preds.append(pred.cpu().numpy())
                targets.append(yb.cpu().numpy())
        preds = np.concatenate(preds).flatten()
        targets = np.concatenate(targets).flatten()
        pearson = np.corrcoef(preds, targets)[0, 1]
        if pearson > best_test_pearson:
            best_test_pearson = pearson

    return best_test_pearson

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print("Best trial:")
    print(study.best_trial)
    print("Best parameters:", study.best_trial.params)