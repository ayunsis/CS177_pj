import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import linear_model
import scipy
import tensorflow as tf
from tf_predict import build_model

MODEL_PATH = 'src/AF3_eval/src/weights_22_112-0.0083.h5'
CORE_GRIDS = r'data/chai_hdf5/core_grids.h5'
CORE_2016_LABEL = r'data/chai_hdf5/core_2016_label.h5'
CORE_sfcnn_LABEL = r'data/chai_hdf5/core_sfcnn_label.h5'

core_csv = pd.read_csv('data/core_affinity_final.csv')

def load_grids_labels(grid_path, grid_key, label_path, label_key, normalize_y=15.0):
    with h5py.File(grid_path, 'r') as fg, h5py.File(label_path, 'r') as fl:
        grids = fg[grid_key][:]
        labels = fl[label_key][:]
    grids = grids.astype(np.float32)
    labels = labels.astype(np.float32) / normalize_y
    return grids, labels

def evaluate(label_path, label_key, output_csv, desc):
    grids, labels = load_grids_labels(CORE_GRIDS, 'core_grids', label_path, label_key)
    model = build_model()
    preds = model.predict(grids, batch_size=32).flatten() * 15
    targets = labels.flatten() * 15

    df = pd.DataFrame({'pdbid': core_csv['pdbid'], 'score': preds, 'affinity': targets})
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

if __name__ == '__main__':
    evaluate(CORE_2016_LABEL, 'core_label', 'src/AF3_eval/outputs/output_core2016_tf.csv', 'CASF2016')
    evaluate(CORE_sfcnn_LABEL, 'core_label', 'src/AF3_eval/outputs/output_sfcnn_tf.csv', 'SFCNN')