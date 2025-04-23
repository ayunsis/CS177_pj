import predict
from openbabel import pybel
from glob import glob
import os


model = predict.build_model()
core_dirs = glob(os.path.join('data\coreset','*'))
core_id = [os.path.split(i)[-1] for i in core_dirs]
f = open('src\sfcnn-origin\out/output.csv','w')
f.write('#code\tscore\n')
for pdbid in core_id:
    proteinfile = os.path.join('data\coreset',pdbid, pdbid+ '_protein.pdb')
    ligandfile = os.path.join('data\coreset',pdbid, pdbid+'_ligand.mol2')
    protein = next(pybel.readfile('pdb',proteinfile))
    ligand = next(pybel.readfile('mol2',ligandfile))
    result = predict.predict(protein, ligand, model)
    f.write(pdbid+'\t%.4f\n' % result)
f.close()
