import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def load_cv_results():
    """Load cross-validation results and summary"""
    try:
        # Try new structure first
        cv_summary = np.load('src/sfcnn/src/train_results/cv_results/cv_summary.npy', allow_pickle=True).item()
        return cv_summary
    except FileNotFoundError:
        try:
            # Fallback to old location
            cv_summary = np.load('src/sfcnn/src/train_results/cv_summary.npy', allow_pickle=True).item()
            return cv_summary
        except FileNotFoundError:
            print("Cross-validation summary not found. Make sure to run training with cross-validation first.")
            return None

def plot_fold_metrics(fold_num, k_folds=7):
    """Plot metrics for a specific fold"""
    # Try new structure first
    fold_dir = f'src/sfcnn/src/train_results/cv_results/fold_{fold_num}/metrics'
    try:
        train_metrics = np.load(f'{fold_dir}/train_metrics_history.npy')
        val_metrics = np.load(f'{fold_dir}/val_metrics_history.npy')
        test_metrics = np.load(f'{fold_dir}/test_metrics_history.npy')
    except FileNotFoundError:
        # Fallback to old structure
        try:
            train_metrics = np.load(f'src/sfcnn/src/train_results/fold{fold_num}_train_metrics_history.npy')
            val_metrics = np.load(f'src/sfcnn/src/train_results/fold{fold_num}_val_metrics_history.npy')
            test_metrics = np.load(f'src/sfcnn/src/train_results/fold{fold_num}_test_metrics_history.npy')
        except FileNotFoundError:
            print(f"Metrics files for fold {fold_num} not found in either new or old structure.")
            return

    max_len = max(train_metrics.shape[0], val_metrics.shape[0], test_metrics.shape[0])
    tm = np.zeros((max_len, train_metrics.shape[1]))
    vm = np.zeros((max_len, val_metrics.shape[1]))
    testm = np.zeros((max_len, test_metrics.shape[1]))

    tm[:train_metrics.shape[0], :] = train_metrics
    vm[:val_metrics.shape[0], :] = val_metrics
    testm[:test_metrics.shape[0], :] = test_metrics

    epochs = tm[:, 0]
    metrics_names = ['Pearson', 'RMSE', 'MAE', 'SD']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    plt.suptitle(f'Fold {fold_num} Training Metrics', fontsize=16, y=0.98)

    for i, name in enumerate(metrics_names, 1):
        plt.subplot(2, 2, i)
        plt.plot(epochs, tm[:, i], label='Train', color=colors[0], linewidth=2)
        plt.plot(epochs, vm[:, i], label='Val', color=colors[1], linewidth=2)
        plt.plot(epochs, testm[:, i], label='Test', color=colors[2], linewidth=2)
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

def plot_cv_summary(k_folds=7):
    """Plot cross-validation summary across all folds"""
    cv_summary = load_cv_results()
    if cv_summary is None:
        return

    all_train_metrics = []
    all_val_metrics = []
    all_test_metrics = []
    
    for fold in range(1, k_folds + 1):
        # Try new structure first, fallback to old structure
        fold_metrics_dir = f'src/sfcnn/src/train_results/cv_results/fold_{fold}/metrics'
        try:
            train_metrics = np.load(f'{fold_metrics_dir}/train_metrics_history.npy')
            val_metrics = np.load(f'{fold_metrics_dir}/val_metrics_history.npy')
            test_metrics = np.load(f'{fold_metrics_dir}/test_metrics_history.npy')
        except FileNotFoundError:
            try:
                train_metrics = np.load(f'src/sfcnn/src/train_results/fold{fold}_train_metrics_history.npy')
                val_metrics = np.load(f'src/sfcnn/src/train_results/fold{fold}_val_metrics_history.npy')
                test_metrics = np.load(f'src/sfcnn/src/train_results/fold{fold}_test_metrics_history.npy')
            except FileNotFoundError:
                print(f"Warning: Metrics for fold {fold} not found in either structure, skipping...")
                continue
        
        all_train_metrics.append(train_metrics)
        all_val_metrics.append(val_metrics)
        all_test_metrics.append(test_metrics)

    if not all_train_metrics:
        print("No fold metrics found.")
        return

    # Find common epoch range and average metrics across folds
    min_epochs = min(len(metrics) for metrics in all_test_metrics)
    
    train_mean = np.mean([metrics[:min_epochs] for metrics in all_train_metrics], axis=0)
    train_std = np.std([metrics[:min_epochs] for metrics in all_train_metrics], axis=0)
    val_mean = np.mean([metrics[:min_epochs] for metrics in all_val_metrics], axis=0)
    val_std = np.std([metrics[:min_epochs] for metrics in all_val_metrics], axis=0)
    test_mean = np.mean([metrics[:min_epochs] for metrics in all_test_metrics], axis=0)
    test_std = np.std([metrics[:min_epochs] for metrics in all_test_metrics], axis=0)

    epochs = train_mean[:, 0]
    metrics_names = ['Pearson', 'RMSE', 'MAE', 'SD']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{k_folds}-Fold Cross-Validation Results Summary', fontsize=16, y=0.98)

    for i, name in enumerate(metrics_names):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        ax.plot(epochs, train_mean[:, i+1], label='Train', color=colors[0], linewidth=2)
        ax.fill_between(epochs, train_mean[:, i+1] - train_std[:, i+1], 
                       train_mean[:, i+1] + train_std[:, i+1], color=colors[0], alpha=0.2)
        
        ax.plot(epochs, val_mean[:, i+1], label='Val', color=colors[1], linewidth=2)
        ax.fill_between(epochs, val_mean[:, i+1] - val_std[:, i+1], 
                       val_mean[:, i+1] + val_std[:, i+1], color=colors[1], alpha=0.2)
        
        ax.plot(epochs, test_mean[:, i+1], label='Test', color=colors[2], linewidth=2)
        ax.fill_between(epochs, test_mean[:, i+1] - test_std[:, i+1], 
                       test_mean[:, i+1] + test_std[:, i+1], color=colors[2], alpha=0.2)
        
        ax.set_title(f'{name} (Mean ± Std)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(name)
        ax.legend(frameon=True)
        ax.grid(alpha=0.7)
        ax.set_xlim(0, max(epochs))
        if name == 'Pearson':
            ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    print(f"\nCross-Validation Summary:")
    print(f"Number of folds: {k_folds}")
    print(f"Mean Test Pearson: {cv_summary['mean_test_pearson']:.4f} ± {cv_summary['std_test_pearson']:.4f}")
    print(f"Best Overall Pearson: {cv_summary['best_overall_pearson']:.4f} (Fold {cv_summary['best_fold']})")
    
    print(f"\nIndividual Fold Results:")
    for i, result in enumerate(cv_summary['fold_results'], 1):
        print(f"  Fold {i}: Best Test Pearson = {result['best_test_pearson']:.4f}")

    if 'best_fold' in cv_summary:
        best_fold_num = cv_summary['best_fold']
        print(f"\nPlotting metrics for the best fold: Fold {best_fold_num}")
        plot_fold_metrics(best_fold_num, k_folds)

def list_available_folds():
    """List available folds in the results directory"""
    print("Checking for available folds...")
    
    # Check new structure
    cv_dir = 'src/sfcnn/src/train_results/cv_results'
    if os.path.exists(cv_dir):
        fold_dirs = [d for d in os.listdir(cv_dir) if d.startswith('fold_') and os.path.isdir(os.path.join(cv_dir, d))]
        if fold_dirs:
            print(f"Found {len(fold_dirs)} folds in new structure: {', '.join(sorted(fold_dirs))}")
            return len(fold_dirs)
    
    # Check old structure
    results_dir = 'src/sfcnn/src/train_results'
    if os.path.exists(results_dir):
        fold_files = [f for f in os.listdir(results_dir) if f.startswith('fold') and f.endswith('_train_metrics_history.npy')]
        if fold_files:
            fold_nums = [f.split('fold')[1].split('_')[0] for f in fold_files]
            print(f"Found {len(fold_nums)} folds in old structure: {', '.join(sorted(fold_nums))}")
            return len(fold_nums)
    
    print("No fold results found.")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--mode', choices=['fold', 'summary', 'legacy', 'list'], default='summary',
                       help='Visualization mode: fold (specific fold), summary (CV summary), legacy (old format), list (show available folds)')
    parser.add_argument('--fold', type=int, default=1, help='Fold number to visualize (for fold mode)')
    parser.add_argument('--k_folds', type=int, default=7, help='Number of folds used in cross-validation')
    args = parser.parse_args()

    if args.mode == 'list':
        available_folds = list_available_folds()
        if available_folds > 0:
            print(f"\nUse --mode fold --fold N to view specific fold results")
            print(f"Use --mode summary to view cross-validation summary")
    
    elif args.mode == 'legacy':
        # Original code for backward compatibility
        try:
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
        except FileNotFoundError:
            print("Legacy metrics files not found. Use --mode summary or --mode fold instead.")
    
    elif args.mode == 'fold':
        plot_fold_metrics(args.fold, args.k_folds)
    
    elif args.mode == 'summary':
        plot_cv_summary(args.k_folds)