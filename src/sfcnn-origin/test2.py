import predict
from openbabel import pybel

model = predict.build_model()
protein = next(pybel.readfile('pdb','src\sfcnn-origin\input/1a30_protein.pdb'))
ligands = list(pybel.readfile('mol2','src\sfcnn-origin\input/1a30_decoys.mol2'))
for ligand in ligands:
    result = predict.predict(protein, ligand, model)
    print(result[0])
