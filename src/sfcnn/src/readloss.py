import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_metrics = np.load('src/sfcnn/src/train_results/train_metrics_history.npy')
val_metrics = np.load('src/sfcnn/src/train_results/val_metrics_history.npy')
test_metrics = np.load('src/sfcnn/src/train_results/test_metrics_history.npy')

# Pad the shorter array with zeros
max_len = max(train_metrics.shape[0], val_metrics.shape[0])
tm = np.zeros((max_len, train_metrics.shape[1]))
vm = np.zeros((max_len, val_metrics.shape[1]))
tm = np.zeros((max_len, test_metrics.shape[1]))

tm[:train_metrics.shape[0], :] = train_metrics
vm[:val_metrics.shape[0], :] = val_metrics
tm[:test_metrics.shape[0], :] = test_metrics

train_metrics = tm
val_metrics = vm
test_metrics = tm

epoch = 300
train_row = train_metrics[epoch - 1]  
val_row = val_metrics[epoch - 1]  
test_row = test_metrics[epoch - 1]

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
print('=' * 30)
print(f"  Test Pearson:  {test_row[1]:.4f}")
print(f"  Test RMSE:     {test_row[2]:.4f}")
print(f"  Test MAE:      {test_row[3]:.4f}")
print(f"  Test SD:       {test_row[4]:.4f}")

epochs = train_metrics[:, 0]
metrics_names = ['Pearson', 'RMSE', 'MAE', 'SD']

plt.figure(figsize=(12, 8))
for i, name in enumerate(metrics_names, 1):
    plt.subplot(2, 2, i)
    sns.lineplot(x=epochs, y=train_metrics[:, i], label='Train')
    sns.lineplot(x=epochs, y=val_metrics[:, i], label='Val')
    sns.lineplot(x=epochs, y=test_metrics[:, i], label='Test')
    plt.title(name)
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend()
    plt.grid()
    plt.xlim(0, max(epochs))
    plt.ylim(0, 1 if name == 'Pearson' else None)

plt.tight_layout()
plt.show()
