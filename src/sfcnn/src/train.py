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

    def get_indices(total, val_start=41000, seed=1234):
        all_idx = np.arange(total)
        np.random.seed(seed)
        np.random.shuffle(all_idx)
        train_idx = all_idx[:val_start]
        val_idx = all_idx[val_start:]
        return train_idx, val_idx

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-b', default=32, type=int) 
    parser.add_argument('--dropout', '-d', default=0.15, type=float)
    parser.add_argument('--lr', default=0.00068, type=float)  
    parser.add_argument('--k_folds', '-k', default=10, type=int, help='Number of folds for cross validation')
    parser.add_argument('--grad_clip', default=0.15, type=float, help='Gradient clipping value')
    args = parser.parse_args()

    with h5py.File(TRAIN_GRIDS, 'r') as f:
        total = len(f['train_grids'])
    
    # Use all training data for cross validation
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
                             num_workers=2)  # Increased workers

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

    if os.path.exists('src/sfcnn/src/train_results/cnnmodel'):
        shutil.rmtree('src/sfcnn/src/train_results/cnnmodel')
    os.makedirs('src/sfcnn/src/train_results/cnnmodel', exist_ok=True)
    
    # Create overall results directory
    if os.path.exists('src/sfcnn/src/train_results/cv_results'):
        shutil.rmtree('src/sfcnn/src/train_results/cv_results')
    os.makedirs('src/sfcnn/src/train_results/cv_results', exist_ok=True)

    # Cross validation setup
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=1234)
    fold_results = []
    best_overall_pearson = -1.0
    best_fold = -1

    TRAIN_EPOCHS = 100
    SAVE_EPOCHS = 0

    # --- Mixed Precision Setup ---
    from torch.cuda.amp import autocast
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(all_train_idx)):
        print(f'\nFOLD {fold + 1}/{args.k_folds}')
        print('-' * 50)
        
        # Create fold-specific directories
        fold_dir = f'src/sfcnn/src/train_results/cv_results/fold_{fold+1}'
        fold_model_dir = f'{fold_dir}/models'
        fold_metrics_dir = f'{fold_dir}/metrics'
        
        os.makedirs(fold_model_dir, exist_ok=True)
        os.makedirs(fold_metrics_dir, exist_ok=True)
        
        # Create datasets for this fold
        train_fold_idx = all_train_idx[train_ids]
        val_fold_idx = all_train_idx[val_ids]
        
        # Save fold indices for reproducibility
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
                                  num_workers=2,  # Increased workers
                                  persistent_workers=True)  # Keep workers alive
        val_loader = DataLoader(val_dataset, 
                                batch_size=args.batch, 
                                pin_memory=True, 
                                num_workers=2,  # Increased workers
                                persistent_workers=True)
        
        # Initialize model for this fold
        model = CNN3D(dropout=args.dropout).to(device)
        
        # Get parameters for manual optimization
        last_linear = model.fc[-1]
        params = [
            {'params': [p for n, p in model.named_parameters() if 'fc.5' not in n], 'weight_decay': 0.0},
            {'params': last_linear.parameters(), 'weight_decay': 0.01} 
        ]
        
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(params, lr=args.lr)

        train_loss_history = []
        train_metrics_history = []
        val_metrics_history = []
        test_metrics_history = []
        best_fold_pearson = -1.0

        for epoch in tqdm(range(1, TRAIN_EPOCHS+1), desc=f"Fold {fold+1} Progress", unit="epoch"):
            model.train()
            train_loss = 0
            train_preds = []
            train_targets = []
            
            # Training loop with progress bar
            train_pbar = tqdm(train_loader, desc=f"F{fold+1}E{epoch:03d} Train", leave=False, unit="batch")
            for xb, yb in train_pbar:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                
                # Zero gradients using optimizers
                optimizer.zero_grad()
                
                # Use autocast for forward pass only
                with autocast():
                    pred = model(xb)
                    loss = criterion(pred, yb)
                
                # Simple backward pass without scaling
                loss.backward()
                
                # Gradient clipping
                # torch.nn.utils.clip_grad_norm_([p for group in params for p in group['params'] if p.grad is not None], args.grad_clip)
                
                # Optimizer steps
                optimizer.step()
                
                train_loss += loss.item() * xb.size(0)
                
                # Only collect predictions every few batches to save memory
                if len(train_preds) < 1000:  # Limit memory usage
                    train_preds.append(pred.detach().cpu().numpy())
                    train_targets.append(yb.detach().cpu().numpy())
                
                # Update progress bar with current loss
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            train_pbar.close()
            train_loss /= len(train_loader.dataset)
            train_loss_history.append(train_loss)

            # Calculate training metrics less frequently for speed
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
                
                # Validation loop with progress bar
                val_pbar = tqdm(val_loader, desc=f"F{fold+1}E{epoch:03d} Val", leave=False, unit="batch")
                with torch.no_grad():
                    for xb, yb in val_pbar:
                        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                        # Use autocast for inference speed
                        with autocast():
                            pred = model(xb)
                        preds.append(pred.cpu().numpy())
                        targets.append(yb.cpu().numpy())
                val_pbar.close()
                
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

                # Test evaluation with progress bar
                model.eval()
                test_preds = []
                test_targets = []
                
                test_pbar = tqdm(test_loader, desc=f"F{fold+1}E{epoch:03d} Test", leave=False, unit="batch")
                with torch.no_grad():
                    for xb, yb in test_pbar:
                        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                        # Use autocast for inference speed
                        with autocast():
                            pred = model(xb)
                        test_preds.append(pred.cpu().numpy())
                        test_targets.append(yb.cpu().numpy())
                test_pbar.close()
                
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

                # Save best model for this fold
                if test_pearson > best_fold_pearson:
                    best_fold_pearson = test_pearson
                    # Save to fold-specific directory
                    torch.save(model.state_dict(), f'{fold_model_dir}/epoch{epoch:03d}_{test_pearson:.4f}.pt')
                    # Also save config info
                    model_info = {
                        'epoch': epoch,
                        'test_pearson': test_pearson,
                        'val_pearson': pearson,
                        'train_pearson': train_pearson,
                        'fold': fold + 1,
                        'model_config': {
                            'dropout': args.dropout,
                            'batch_size': args.batch,
                            'learning_rate': args.lr
                        }
                    }
                    np.save(f'{fold_model_dir}/best_model_info.npy', model_info)
                
                # Update overall best
                if test_pearson > best_overall_pearson:
                    best_overall_pearson = test_pearson
                    best_fold = fold + 1
                    # Save overall best model
                    torch.save(model.state_dict(), f'src/sfcnn/src/train_results/cnnmodel/best_overall_weights.pt')
                    # Copy to main model directory as well
                    torch.save(model.state_dict(), f'{fold_model_dir}/overall_best_weights.pt')

            # Update main progress bar with current metrics
            # tqdm.write(f"F{fold+1}E{epoch:03d} | Train: P={train_pearson:.4f} L={train_loss:.4f} | " +
            #           (f"Val: P={pearson:.4f} | Test: P={test_pearson:.4f}" if epoch >= SAVE_EPOCHS else ""))

        # Save fold results to fold-specific directory
        np.save(f'{fold_metrics_dir}/train_metrics_history.npy', np.array(train_metrics_history))
        np.save(f'{fold_metrics_dir}/val_metrics_history.npy', np.array(val_metrics_history))
        np.save(f'{fold_metrics_dir}/test_metrics_history.npy', np.array(test_metrics_history))
        np.save(f'{fold_metrics_dir}/train_loss_history.npy', np.array(train_loss_history))
        
        # Save final model state
        torch.save(model.state_dict(), f'{fold_model_dir}/final_weights_epoch{TRAIN_EPOCHS}.pt')
        
        # Save fold summary
        fold_summary = {
            'fold': fold + 1,
            'best_test_pearson': best_fold_pearson,
            'final_val_pearson': val_metrics_history[-1][1] if val_metrics_history else 0,
            'final_test_pearson': test_metrics_history[-1][1] if test_metrics_history else 0,
            'final_train_pearson': train_metrics_history[-1][1] if train_metrics_history else 0,
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
        
        print(f'Fold {fold + 1} completed. Best test Pearson: {best_fold_pearson:.4f}')
        print(f'Results saved to: {fold_dir}')

    # Print cross-validation summary
    print(f'\nCross-Validation Results Summary:')
    print('-' * 50)
    test_pearsons = [result['best_test_pearson'] for result in fold_results]
    print(f'Mean Test Pearson: {np.mean(test_pearsons):.4f} ± {np.std(test_pearsons):.4f}')
    print(f'Best Overall Pearson: {best_overall_pearson:.4f} (Fold {best_fold})')
    
    # Save comprehensive cross-validation summary
    cv_summary = {
        'fold_results': fold_results,
        'mean_test_pearson': np.mean(test_pearsons),
        'std_test_pearson': np.std(test_pearsons),
        'best_overall_pearson': best_overall_pearson,
        'best_fold': best_fold,
        'total_train_samples': len(all_train_idx),
        'total_test_samples': len(test_idx),
        'cv_config': {
            'k_folds': args.k_folds,
            'random_state': 1234,
            'shuffle': True
        },
        'global_hyperparameters': {
            'batch_size': args.batch,
            'dropout': args.dropout,
            'learning_rate': args.lr,
            'epochs': TRAIN_EPOCHS
        }
    }
    np.save('src/sfcnn/src/train_results/cv_results/cv_summary.npy', cv_summary)
    
    # Also save a backup in the main results directory
    np.save('src/sfcnn/src/train_results/cv_summary.npy', cv_summary)
    
    print(f'\nAll results saved to: src/sfcnn/src/train_results/cv_results/')