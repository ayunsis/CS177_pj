import predict
from openbabel import pybel

model = predict.build_model()
protein = next(pybel.readfile('pdb',r'src\sfcnn-origin\input/1a30_protein.pdb'))
ligand = next(pybel.readfile('mol2',r'src\sfcnn-origin\input/1a30_ligand.mol2'))
result = predict.predict(protein, ligand, model)
print(result[0])
