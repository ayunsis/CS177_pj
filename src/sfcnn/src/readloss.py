import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_metrics = np.load('src/sfcnn/src/train_results/train_metrics_history.npy')
val_metrics = np.load('src/sfcnn/src/train_results/val_metrics_history.npy')
test_metrics = np.load('src/sfcnn/src/train_results/test_metrics_history.npy')

max_len = max(train_metrics.shape[0], val_metrics.shape[0])
tm = np.zeros((max_len, train_metrics.shape[1]))
vm = np.zeros((max_len, val_metrics.shape[1]))
testm = np.zeros((max_len, test_metrics.shape[1]))

tm[:train_metrics.shape[0], :] = train_metrics
vm[:val_metrics.shape[0], :] = val_metrics
testm[:test_metrics.shape[0], :] = test_metrics

train_metrics = tm
val_metrics = vm
test_metrics = testm

epochs = train_metrics[:, 0]
metrics_names = ['Pearson', 'RMSE', 'MAE', 'SD']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green

sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

for i, name in enumerate(metrics_names, 1):
    plt.subplot(2, 2, i)
    plt.plot(epochs, train_metrics[:, i], label='Train', color=colors[0], linewidth=2)
    plt.plot(epochs, val_metrics[:, i], label='Val', color=colors[1], linewidth=2)
    plt.plot(epochs, test_metrics[:, i], label='Test', color=colors[2], linewidth=2)
    plt.title(name)
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend(frameon=True)
    plt.grid(alpha=0.7)
    plt.xlim(0, max(epochs))
    if name == 'Pearson':
        plt.ylim(0, 1)

plt.tight_layout()
plt.show()