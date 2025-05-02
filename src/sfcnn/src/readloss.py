import numpy as np


train_metrics = np.load('src/sfcnn/src/train_results/train_metrics_history.npy')
val_metrics = np.load('src/sfcnn/src/train_results/val_metrics_history.npy')


epoch = 71
train_row = train_metrics[epoch - 1]  
val_row = val_metrics[epoch - 1]  

print(f"Epoch {int(train_row[0])}:")
print(f"  Train Pearson: {train_row[1]:.4f}")
print(f"  Train RMSE:    {train_row[2]:.4f}")
print(f"  Train MAE:     {train_row[3]:.4f}")
print(f"  Train SD:      {train_row[4]:.4f}")
print('=' * 30)
print(f"  Val Pearson:   {val_row[1]:.4f}")
print(f"  Val RMSE:      {val_row[2]:.4f}")
print(f"  Val MAE:       {val_row[3]:.4f}")
print(f"  Val SD:        {val_row[4]:.4f}")