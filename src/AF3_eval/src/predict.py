import argparse
from openbabel import pybel
import numpy as np
import torch
import torch.nn as nn


class Feature_extractor():
    def __init__(self):
        self.atom_codes = {}
        others = ([3,4,5,11,12,13]+list(range(19,32))+list(range(37,51))+list(range(55,84)))
        atom_types = [1,(6,1),(6,2),(6,3),(7,1),(7,2),(7,3),8,15,(16,2),(16,3),34,[9,17,35,53],others]
        for i, j in enumerate(atom_types):
            if type(j) is list:
                for k in j:
                    self.atom_codes[k] = i
            else:
                self.atom_codes[j] = i              
        self.sum_atom_types = len(atom_types)
    def encode(self, atomic_num, molprotein):
        encoding = np.zeros(self.sum_atom_types*2)
        if molprotein == 1:
            encoding[self.atom_codes[atomic_num]] = 1.0
        else:
            encoding[self.sum_atom_types+self.atom_codes[atomic_num]] = 1.0
        return encoding
    def get_features(self, molecule, molprotein):
        coords = []
        features = []
        for atom in molecule:
            coords.append(atom.coords)
            if atom.atomicnum in [6,7,16]:
                atomicnum = (atom.atomicnum,atom.hyb)
                features.append(self.encode(atomicnum,molprotein))
            else:
                features.append(self.encode(atom.atomicnum,molprotein))
        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        return coords, features  
    def grid(self,coords, features):
        assert coords.shape[1] == 3
        assert coords.shape[0] == features.shape[0]  
        grid=np.zeros((1,20,20,20,features.shape[1]),dtype=np.float32)
        x=y=z=np.array(range(-10,10),dtype=np.float32)+0.5
        for i in range(len(coords)):
            coord=coords[i]
            tmpx=abs(coord[0]-x)
            tmpy=abs(coord[1]-y)
            tmpz=abs(coord[2]-z)
            if np.max(tmpx)<=19.5 and np.max(tmpy)<=19.5 and np.max(tmpz) <=19.5:
                grid[0,np.argmin(tmpx),np.argmin(tmpy),np.argmin(tmpz)] += features[i]                
        return grid

def get_grid(protein, ligand):
    Feature = Feature_extractor()
    coords1, features1 = Feature.get_features(protein,1)
    coords2, features2 = Feature.get_features(ligand,0)
    center=(np.max(coords2,axis=0)+np.min(coords2,axis=0))/2
    coords=np.concatenate([coords1,coords2],axis = 0)
    features=np.concatenate([features1,features2],axis = 0)
    assert len(coords) == len(features)
    coords = coords-center
    grid=Feature.grid(coords,features)
    return grid

# --- PyTorch Model definition (same as in train.py) ---
class CNN3D(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(28, 7, kernel_size=1),
            nn.BatchNorm3d(7),
            nn.ReLU(),
            nn.Conv3d(7, 7, kernel_size=3),
            nn.BatchNorm3d(7),
            nn.ReLU(),
            nn.Conv3d(7, 7, kernel_size=3),
            nn.BatchNorm3d(7),
            nn.ReLU(),
            nn.Conv3d(7, 28, kernel_size=1),
            nn.BatchNorm3d(28),
            nn.ReLU(),
            nn.Conv3d(28, 56, kernel_size=3, padding=1),
            nn.BatchNorm3d(56),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(56, 112, kernel_size=3, padding=1),
            nn.BatchNorm3d(112),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(112, 224, kernel_size=3, padding=1),
            nn.BatchNorm3d(224),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224*2*2*2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)  # (B, 20, 20, 20, 28) -> (B, 28, 20, 20, 20)
        x = self.net(x)
        x = self.fc(x)
        return x

def predict(protein, ligand, model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    grid = get_grid(protein, ligand)  # shape: (1, 20, 20, 20, 28)
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(grid_tensor)
        result = output.item() * 15  # match normalization in training
    return result

def build_model(weights_path, dropout=0.5, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = CNN3D(dropout=dropout)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='PyTorch Predict the affinity of protein and ligand!')
    parser.add_argument('--protein', '-p', required=True, help='protein file')
    parser.add_argument('--pff', '-m', default='pdb', help='file format of protein')
    parser.add_argument('--ligand', '-l', required=True, help='ligand file')
    parser.add_argument('--lff', '-n', default='mol2', help='file format of ligand')
    parser.add_argument('--output', '-o', default='output', help='output file')
    parser.add_argument('--weights', '-w', required=True, help='path to PyTorch model weights (.pt)')
    args = parser.parse_args()

    ligands = list(pybel.readfile(args.lff, args.ligand))
    protein = next(pybel.readfile(args.pff, args.protein))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN3D()
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()
    with open(args.output, 'w') as f:
        f.write('Predict the affinity of %s and %s\n' % (args.protein, args.ligand))
        for ligand in ligands:
            result = predict(protein, ligand, model, device)
            f.write('%.4f\n' % result)
    print('Done!')

if __name__ == '__main__':
    main()

# python src\sfcnn\src\predict.py --protein src\sfcnn-origin\input\1a30_protein.pdb --ligand src\sfcnn-origin\input\1a30_ligand.mol2 --weights src\sfcnn\src\train_results\cnnmodel\weights_001-0.2389.pt --output result.txt