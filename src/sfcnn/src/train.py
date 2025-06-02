import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py
import argparse
from tqdm import tqdm
from sklearn import linear_model
from sklearn.model_selection import KFold
import warnings
import shutil
warnings.filterwarnings("ignore")

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-b', default=32, type=int) 
    parser.add_argument('--dropout', '-d', default=0.15, type=float)
    parser.add_argument('--lr', default=0.00068, type=float)  
    parser.add_argument('--k_folds', '-k', default=7, type=int, help='Number of folds for cross validation')
    args = parser.parse_args()

    with h5py.File(TRAIN_GRIDS, 'r') as f:
        total = len(f['train_grids'])
    
    all_train_idx = np.arange(total)
    
    with h5py.File(CORE_GRIDS, 'r') as f:
        test_len = len(f['core_grids'])
    test_idx = np.arange(test_len)

    test_dataset = HDF5GridDataset(
        CORE_GRIDS, 'core_grids', CORE_LABEL, 'core_label', test_idx
    )
    test_loader = DataLoader(test_dataset, 
                             batch_size=args.batch, 
                             pin_memory=True, 
                             num_workers=2)

    print('Dataset load finished')

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('current device: ', device)

    if os.path.exists('src/sfcnn/src/train_results/cnnmodel'):
        shutil.rmtree('src/sfcnn/src/train_results/cnnmodel')
    os.makedirs('src/sfcnn/src/train_results/cnnmodel', exist_ok=True)
    if os.path.exists('src/sfcnn/src/train_results/cv_results'):
        shutil.rmtree('src/sfcnn/src/train_results/cv_results')
    os.makedirs('src/sfcnn/src/train_results/cv_results', exist_ok=True)

    print("Using normal K-Fold cross validation...")
    
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=1234)
    fold_results = []
    best_overall_loss = float('inf')
    best_fold = -1
    TRAIN_EPOCHS = 150
    SAVE_EPOCHS = 0
    from torch.cuda.amp import autocast, GradScaler
    for fold, (train_ids, val_ids) in enumerate(kfold.split(all_train_idx)):        
        print(f'\nFOLD {fold + 1}/{args.k_folds}')
        print('-' * 50)
        
        print(f"Train set: {len(train_ids)} samples")
        print(f"Val set: {len(val_ids)} samples")
        
        fold_dir = f'src/sfcnn/src/train_results/cv_results/fold_{fold+1}'
        fold_model_dir = f'{fold_dir}/models'
        fold_metrics_dir = f'{fold_dir}/metrics'
        
        os.makedirs(fold_model_dir, exist_ok=True)
        os.makedirs(fold_metrics_dir, exist_ok=True)
        
        train_fold_idx = all_train_idx[train_ids]
        val_fold_idx = all_train_idx[val_ids]
        np.save(f'{fold_dir}/train_indices.npy', train_fold_idx)
        np.save(f'{fold_dir}/val_indices.npy', val_fold_idx)
        
        train_dataset = HDF5GridDataset(
            TRAIN_GRIDS, 'train_grids', TRAIN_LABEL, 'train_label', train_fold_idx
        )
        val_dataset = HDF5GridDataset(
            TRAIN_GRIDS, 'train_grids', TRAIN_LABEL, 'train_label', val_fold_idx
        )
        
        train_loader = DataLoader(train_dataset, 
                                  batch_size=args.batch, 
                                  shuffle=True, 
                                  pin_memory=True, 
                                  num_workers=2,
                                  persistent_workers=True)
        val_loader = DataLoader(val_dataset, 
                                batch_size=args.batch, 
                                pin_memory=True, 
                                num_workers=2,
                                persistent_workers=True)
        
        model = CNN3D(dropout=args.dropout).to(device)
        
        last_linear = model.fc[-1]
        params = [
            {'params': [p for n, p in model.named_parameters() if 'fc.5' not in n], 'weight_decay': 0},
            {'params': last_linear.parameters(), 'weight_decay': 0.01} 
        ]
        
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(params, lr=args.lr)
        
        scaler = GradScaler()

        train_loss_history = []
        train_metrics_history = []
        val_metrics_history = []
        test_metrics_history = []
        best_fold_loss = float('inf')

        for epoch in tqdm(range(1, TRAIN_EPOCHS+1), desc=f"Fold {fold+1} Progress", unit="epoch"):
            model.train()
            train_loss = 0
            train_preds = []
            train_targets = []
            
            train_pbar = tqdm(train_loader, desc=f"F{fold+1}E{epoch:03d} Train", leave=False, unit="batch")
            for xb, yb in train_pbar:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                with autocast():
                    pred = model(xb)
                    loss = criterion(pred, yb)
                
                scaler.scale(loss).backward()
                
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item() * xb.size(0)
                train_preds.append(pred.detach().cpu().numpy())
                train_targets.append(yb.detach().cpu().numpy())
                
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            train_pbar.close()
            train_loss /= len(train_loader.dataset)
            train_loss_history.append(train_loss)

            if len(train_preds) > 0:
                train_preds = np.concatenate(train_preds).flatten()
                train_targets = np.concatenate(train_targets).flatten()
                train_pearson = np.corrcoef(train_preds, train_targets)[0, 1]
                train_rmse = np.sqrt(np.mean((train_preds - train_targets) ** 2))
                train_mae = np.mean(np.abs(train_preds - train_targets))
                train_regr = linear_model.LinearRegression()
                x_train = train_preds.reshape(-1, 1)
                y_train = train_targets.reshape(-1, 1)
                train_regr.fit(x_train, y_train)
                y_train_ = train_regr.predict(x_train)
                train_sd = np.sqrt(np.sum((y_train - y_train_) ** 2) / (len(y_train) - 1.0))
                train_metrics_history.append([epoch, train_pearson, train_rmse, train_mae, train_sd])
            else:
                train_metrics_history.append([epoch, 0.0, 0.0, 0.0, 0.0])
            
            if epoch >= SAVE_EPOCHS:
                model.eval()
                preds = []
                targets = []
                val_loss = 0
                
                val_pbar = tqdm(val_loader, desc=f"F{fold+1}E{epoch:03d} Val", leave=False, unit="batch")
                with torch.no_grad():
                    for xb, yb in val_pbar:
                        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                        with autocast():
                            pred = model(xb)
                            loss = criterion(pred, yb)
                        val_loss += loss.item() * xb.size(0)
                        preds.append(pred.cpu().numpy())
                        targets.append(yb.cpu().numpy())
                val_pbar.close()
                
                val_loss /= len(val_loader.dataset)
                preds = np.concatenate(preds).flatten()
                targets = np.concatenate(targets).flatten()
                pearson = np.corrcoef(preds, targets)[0, 1]
                rmse = np.sqrt(np.mean((preds - targets) ** 2))
                mae = np.mean(np.abs(preds - targets))
                
                regr = linear_model.LinearRegression()
                x = preds.reshape(-1, 1)
                y = targets.reshape(-1, 1)
                regr.fit(x, y)
                y_ = regr.predict(x)
                sd = np.sqrt(np.sum((y - y_) ** 2) / (len(y) - 1.0))

                val_metrics_history.append([epoch, pearson, rmse, mae, sd])

                
                test_preds = []
                test_targets = []
                test_loss = 0
                
                test_pbar = tqdm(test_loader, desc=f"F{fold+1}E{epoch:03d} Test", leave=False, unit="batch")
                with torch.no_grad():
                    for xb, yb in test_pbar:
                        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                        with autocast():
                            pred = model(xb)
                            loss = criterion(pred, yb)
                        test_loss += loss.item() * xb.size(0)
                        test_preds.append(pred.cpu().numpy())
                        test_targets.append(yb.cpu().numpy())
                test_pbar.close()
                
                test_loss /= len(test_loader.dataset)
                test_preds = np.concatenate(test_preds).flatten()
                test_targets = np.concatenate(test_targets).flatten()
                test_pearson = np.corrcoef(test_preds, test_targets)[0, 1]
                test_rmse = np.sqrt(np.mean((test_preds - test_targets) ** 2))
                test_mae = np.mean(np.abs(test_preds - test_targets))
                test_regr = linear_model.LinearRegression()
                test_x = test_preds.reshape(-1, 1)
                test_y = test_targets.reshape(-1, 1)
                test_regr.fit(test_x, test_y)
                test_y_ = test_regr.predict(test_x)
                test_sd = np.sqrt(np.sum((test_y - test_y_) ** 2) / (len(test_y) - 1.0))

                test_metrics_history.append([epoch, test_pearson, test_rmse, test_mae, test_sd])

                if val_loss < best_fold_loss:
                    best_fold_loss = val_loss
                    torch.save(model.state_dict(), f'{fold_model_dir}/epoch{epoch:03d}_{val_loss:.4f}.pt')
                    model_info = {
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'val_pearson': pearson,
                        'train_pearson': train_pearson,
                        'test_pearson': test_pearson,
                        'fold': fold + 1,
                        'model_config': {
                            'dropout': args.dropout,
                            'batch_size': args.batch,
                            'learning_rate': args.lr
                        }
                    }
                    np.save(f'{fold_model_dir}/best_model_info.npy', model_info)
                
                if val_loss < best_overall_loss:
                    best_overall_loss = val_loss
                    best_fold = fold + 1
                    torch.save(model.state_dict(), f'src/sfcnn/src/train_results/cnnmodel/best_overall_weights.pt')
                    torch.save(model.state_dict(), f'{fold_model_dir}/overall_best_weights.pt')


        np.save(f'{fold_metrics_dir}/train_metrics_history.npy', np.array(train_metrics_history))
        np.save(f'{fold_metrics_dir}/val_metrics_history.npy', np.array(val_metrics_history))
        np.save(f'{fold_metrics_dir}/test_metrics_history.npy', np.array(test_metrics_history))
        np.save(f'{fold_metrics_dir}/train_loss_history.npy', np.array(train_loss_history))
        
        torch.save(model.state_dict(), f'{fold_model_dir}/final_weights_epoch{TRAIN_EPOCHS}.pt')
        

        final_test_metrics = test_metrics_history[-1] if test_metrics_history else [0, 0, 0, 0, 0]
        fold_summary = {
            'fold': fold + 1,
            'best_val_loss': best_fold_loss,
            'final_val_loss': val_metrics_history[-1][2] if val_metrics_history else 0,  # RMSE as loss proxy
            'final_test_loss': final_test_metrics[2],  # RMSE as loss proxy
            'final_train_loss': train_loss,
            'final_test_pearson': final_test_metrics[1],
            'final_val_pearson': val_metrics_history[-1][1] if val_metrics_history else 0,
            'final_train_pearson': train_metrics_history[-1][1] if train_metrics_history else 0,
            'test_metrics': {
                'test_pearson': final_test_metrics[1],
                'test_rmse': final_test_metrics[2],
                'test_mae': final_test_metrics[3],
                'test_sd': final_test_metrics[4]
            },
            'total_epochs': TRAIN_EPOCHS,
            'train_samples': len(train_fold_idx),
            'val_samples': len(val_fold_idx),
            'hyperparameters': {
                'batch_size': args.batch,
                'dropout': args.dropout,
                'learning_rate': args.lr,
                'epochs': TRAIN_EPOCHS
            }
        }
        np.save(f'{fold_dir}/fold_summary.npy', fold_summary)
        
        fold_results.append(fold_summary)
        
        print(f'Fold {fold + 1} completed. Best val loss: {best_fold_loss:.4f}, Test loss: {test_loss:.4f}')
        print(f'Results saved to: {fold_dir}')

    print(f'\nCross-Validation Results Summary:')
    print('-' * 50)
    val_losses = [result['best_val_loss'] for result in fold_results]
    test_losses = [result['final_test_loss'] for result in fold_results]
    print(f'Mean Val Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}')
    print(f'Mean Test Loss: {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}')
    print(f'Best Overall Val Loss: {best_overall_loss:.4f} (Fold {best_fold})')
    
    cv_summary = {
        'fold_results': fold_results,
        'mean_val_loss': np.mean(val_losses),
        'std_val_loss': np.std(val_losses),
        'mean_test_loss': np.mean(test_losses),
        'std_test_loss': np.std(test_losses),
        'best_overall_val_loss': best_overall_loss,
        'best_fold': best_fold,
        'total_train_samples': len(all_train_idx),
        'total_test_samples': len(test_idx),        
        'cv_config': {
            'k_folds': args.k_folds,
            'random_state': 1234,
            'shuffle': True,
            'cv_type': 'normal_kfold'
        },
        'global_hyperparameters': {
            'batch_size': args.batch,
            'dropout': args.dropout,
            'learning_rate': args.lr,
            'epochs': TRAIN_EPOCHS
        }
    }
    np.save('src/sfcnn/src/train_results/cv_results/cv_summary.npy', cv_summary)
    
    np.save('src/sfcnn/src/train_results/cv_summary.npy', cv_summary)
    
    print(f'\nAll results saved to: src/sfcnn/src/train_results/cv_results/')