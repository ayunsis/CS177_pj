import predict
from openbabel import pybel
from glob import glob
import os

MODEL_PATH = 'src/sfcnn/src/train_results/cnnmodel/weights_021-0.1539.pt'
core_dirs = glob(os.path.join('data/coreset','*'))
print(len(core_dirs))

core_id = [os.path.split(i)[-1] for i in core_dirs]

model = predict.build_model(MODEL_PATH)
# for name, param in model.named_parameters():
#     print(name, param.data.mean().item(), param.data.std().item())
f = open('src/sfcnn/outputs/output.csv','w')
f.write('#code\tscore\n')
for pdbid in core_id:
    proteinfile = os.path.join('data/coreset',pdbid, pdbid+ '_protein.pdb')
    ligandfile = os.path.join('data/coreset',pdbid, pdbid+'_ligand.mol2')
    protein = next(pybel.readfile('pdb',proteinfile))
    ligand = next(pybel.readfile('mol2',ligandfile))
    result = predict.predict(protein, ligand, model)
    f.write(pdbid+'\t%.4f\n' % result)
f.close()