import predict
from openbabel import pybel
from glob import glob
import os
import pandas as pd
import numpy as np
from sklearn import linear_model
MODEL_PATH = 'src/sfcnn/src/train_results/cnnmodel/weights_003-0.0148.pt'
core_dirs = glob(os.path.join('data/coreset','*'))
print(len(core_dirs))

core_id = [os.path.split(i)[-1] for i in core_dirs]

model = predict.build_model(MODEL_PATH)

# f = open('src/sfcnn/outputs/output.csv','w')
# f.write('#code\tscore\n')
# for pdbid in core_id:
#     proteinfile = os.path.join('data/coreset',pdbid, pdbid+ '_protein.pdb')
#     ligandfile = os.path.join('data/coreset',pdbid, pdbid+'_ligand.mol2')
#     protein = next(pybel.readfile('pdb',proteinfile))
#     ligand = next(pybel.readfile('mol2',ligandfile))
#     result = predict.predict(protein, ligand, model)
#     f.write(pdbid+'\t%.4f\n' % result)
# f.close()

predict_result = pd.read_csv('src\sfcnn\outputs\output.csv', comment='#', sep='\t', names=['pdbid', 'score'], header=0)
core_affinity = pd.read_csv('data/core_affinity_2016.csv', sep='\t')

df = pd.merge(predict_result, core_affinity, on='pdbid')

regr=linear_model.LinearRegression()
x=df['score'].values.reshape(-1,1)
y=df['affinity'].values.reshape(-1,1)
regr.fit( x  ,  y  )
y_ = regr.predict(x)

print(df[['score', 'affinity']].corr())
print('RMSE: ' + str(np.sqrt(np.mean((df['score']-df['affinity'])**2))))
print('MAE: ' + str(np.mean(np.abs(df['score']-df['affinity']))))
print('SD: ' + str(np.sqrt(sum((y-y_)**2)/284)))