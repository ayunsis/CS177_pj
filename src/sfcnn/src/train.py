import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py
import argparse
from tqdm import tqdm  # <-- Add tqdm import

with h5py.File('data/train_hdf5/train_grids.h5', 'r') as f:
    print("train_grids:", f['train_grids'].shape)

with h5py.File('data/train_hdf5/train_label.h5', 'r') as f:
    print("train_label:", f['train_label'].shape)

with h5py.File('data/test_hdf5/core_grids.h5', 'r') as f:
    print("core_grids:", f['core_grids'].shape)

with h5py.File('data/test_hdf5/core_label.h5', 'r') as f:
    print("core_label:", f['core_label'].shape)
# Efficient HDF5 Dataset class
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

TRAIN_GRIDS = r'data/train_hdf5/train_grids.h5'
TRAIN_LABEL = r'data/train_hdf5/train_label.h5'
CORE_GRIDS = r'data/test_hdf5/core_grids.h5'
CORE_LABEL = r'data/test_hdf5/core_label.h5'

def get_indices(total, val_start=4100, seed=1234):
    all_idx = np.arange(total)
    np.random.seed(seed)
    np.random.shuffle(all_idx)
    train_idx = all_idx[:val_start]
    val_idx = all_idx[val_start:]
    return train_idx, val_idx

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--batch', '-b', default=64, type=int)
parser.add_argument('--dropout', '-d', default=0.5, type=float)
parser.add_argument('--lr', default=0.004, type=float)
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
# Model definition
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
        x = x.permute(0, 4, 1, 2, 3)  # (64,10,20,20,20,28) -> (10,64,20,20,20)
        x = self.net(x)
        x = self.fc(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('current device: ', device)
model = CNN3D(dropout=args.dropout).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=0.01)
criterion = nn.MSELoss()

# Training loop
best_val_loss = float('inf')
os.makedirs('src/sfcnn/src/train_results/cnnmodel', exist_ok=True)

train_loss_history = []
val_loss_history = []
val_mae_history = []

TRAIN_EPOCHS = 200

for epoch in tqdm(range(1, TRAIN_EPOCHS+1), desc="Epochs"):
    train_losses = []
    val_losses = []
    val_maes = []
    model.train()
    train_loss = 0
    for xb, yb in tqdm(train_loader, desc=f"Train {epoch:03d}", leave=False):
        xb, yb = xb.to(device), yb.to(device)  # xb: [batch, 10, 20, 20, 20, 28], yb: [batch, 1]
        b, s, d1, d2, d3, c = xb.shape

        xb = xb.reshape(b * s, d1, d2, d3, c)  # [batch*10, 20, 20, 20, 28]
        yb = yb.repeat(1, s).reshape(b * s, 1)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    train_loss_history.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    val_mae = 0
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc=f"Val {epoch:03d}", leave=False):
            xb, yb = xb.to(device), yb.to(device)  # xb: [batch, 10, 20, 20, 20, 28], yb: [batch, 1]
            b, s, d1, d2, d3, c = xb.shape

            xb = xb.reshape(b * s, d1, d2, d3, c)  # [batch*10, 20, 20, 20, 28]
            yb = yb.repeat(1, s).reshape(b * s, 1)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item() * xb.size(0)
            val_mae += torch.abs(pred - yb).sum().item()
    val_loss /= len(val_loader.dataset)
    val_mae /= len(val_loader.dataset)
    val_losses.append(val_loss)
    val_maes.append(val_mae)
    val_loss_history.append(val_loss)
    val_mae_history.append(val_mae)

    # print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mae={val_mae:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'src/sfcnn/src/train_results/cnnmodel/weights_{epoch:03d}-{val_loss:.4f}.pt')


np.save('src/sfcnn/src/train_results/train_loss_history.npy', np.array(train_loss_history))
np.save('src/sfcnn/src/train_results/val_loss_history.npy', np.array(val_loss_history))
np.save('src/sfcnn/src/train_results/val_mae_history.npy', np.array(val_mae_history))

model.eval()
test_loss = 0
test_mae = 0
with torch.no_grad():
    for xb, yb in tqdm(test_loader, desc="Testing", leave=False):
        xb, yb = xb.to(device), yb.to(device)  # xb: [batch, 10, 20, 20, 20, 28], yb: [batch, 1]
        b, s, d1, d2, d3, c = xb.shape
    
        xb = xb.reshape(b * s, d1, d2, d3, c)  # [batch*10, 20, 20, 20, 28]
        yb = yb.repeat(1, s).reshape(b * s, 1)
        pred = model(xb)
        test_loss += criterion(pred, yb).item() * xb.size(0)
        test_mae += torch.abs(pred - yb).sum().item()
test_loss /= len(test_loader.dataset)
test_mae /= len(test_loader.dataset)
print(f"Test loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")