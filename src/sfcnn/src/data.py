#!/usr/bin/env python
# coding: utf-8
import os

os.environ["OPENBABEL_WARNINGS"] = "0"

print("Importing libraries...")
from glob import glob
from openbabel import pybel
import numpy as np
import random
import h5py

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
    def rotation_matrix(self, t, roller):
        if roller==0:
            return np.array([[1,0,0],[0,np.cos(t),np.sin(t)],[0,-np.sin(t),np.cos(t)]])
        elif roller==1:
            return np.array([[np.cos(t),0,-np.sin(t)],[0,1,0],[np.sin(t),0,np.cos(t)]])
        elif roller==2:
            return np.array([[np.cos(t),np.sin(t),0],[-np.sin(t),np.cos(t),0],[0,0,1]])
    def grid(self,coords, features, resolution=1.0, max_dist=10.0, rotations=9):
        assert coords.shape[1] == 3
        assert coords.shape[0] == features.shape[0]  
        grids = []
        x=y=z=np.array(range(-10,10),dtype=np.float32)+0.5
        # original
        grid=np.zeros((20,20,20,features.shape[1]),dtype=np.float32)
        for i in range(len(coords)):
            coord=coords[i]
            tmpx=abs(coord[0]-x)
            tmpy=abs(coord[1]-y)
            tmpz=abs(coord[2]-z)
            if np.max(tmpx)<=19.5 and np.max(tmpy)<=19.5 and np.max(tmpz) <=19.5:
                grid[np.argmin(tmpx),np.argmin(tmpy),np.argmin(tmpz)] += features[i]
        grids.append(grid)
        # rotations
        coords_rot = coords.copy()
        for j in range(rotations):
            theta = random.uniform(np.pi/18,np.pi/2)
            roller = random.randrange(3)
            coords_rot = np.dot(coords_rot, self.rotation_matrix(theta,roller))
            grid_rot=np.zeros((20,20,20,features.shape[1]),dtype=np.float32)
            for i in range(len(coords_rot)):
                coord=coords_rot[i]
                tmpx=abs(coord[0]-x)
                tmpy=abs(coord[1]-y)
                tmpz=abs(coord[2]-z)
                if np.max(tmpx)<=19.5 and np.max(tmpy)<=19.5 and np.max(tmpz) <=19.5:
                    grid_rot[np.argmin(tmpx),np.argmin(tmpy),np.argmin(tmpz)] += features[i]
            grids.append(grid_rot)
        return np.stack(grids, axis=0)  

Feature = Feature_extractor()

TRAIN_PATH = r'data/refined-set'
CORE_PATH = r'data/coreset'
print("Preparing training and core directories...")
train_dirs = glob(os.path.join(TRAIN_PATH,'*')) 
core_dirs = glob(os.path.join(CORE_PATH,'*'))
core_dirs.sort()
core_id = [os.path.split(i)[1] for i in core_dirs]
train_new_dirs=[]
for i in train_dirs:
    pdb_id = os.path.split(i)[1]
    if pdb_id not in core_id:
        train_new_dirs.append(i)
np.random.shuffle(train_new_dirs)

INDEX_PATH = r'data/PDBbind_v2019_plain_text_index/index/INDEX_general_PL_data.2019'
print("Loading affinity data...")
affinity ={}
with open(INDEX_PATH,'r') as f:
    for line in f.readlines():
        if line[0] != '#':
            affinity[line.split()[0]] = float(line.split()[3])

train_label = []
core_label = []
for i in train_new_dirs:
    pdb_id=os.path.split(i)[1]
    train_label.extend([affinity[pdb_id]]*10)
    
for i in core_dirs:
    core_id=os.path.split(i)[1]
    if not affinity.get(core_id):
        print(core_id)
    else:
        core_label.append(affinity[core_id])
train_label=np.array(train_label,dtype=np.float32)
core_label=np.array(core_label,dtype=np.float32)

TRAIN_LABELS = r'data/train_hdf5/train_label.h5'
CORE_LABELS = r'data/test_hdf5/core_label.h5'
TRAIN_GRIDS = r'data/train_hdf5/train_grids.h5'
TEST_GRIDS = r'data/test_hdf5/core_grids.h5'

print("Saving label data...")
os.makedirs(os.path.dirname(TRAIN_LABELS), exist_ok=True)
os.makedirs(os.path.dirname(CORE_LABELS), exist_ok=True)
with h5py.File(TRAIN_LABELS, 'w') as f:
    f.create_dataset('train_label', data=train_label)
with h5py.File(CORE_LABELS, 'w') as f:
    f.create_dataset('core_label', data=core_label)

print("Processing core complexes...")
core_complexes = []
for directory in core_dirs:
    pdb_id = os.path.split(directory)[1]
    ligand = next(pybel.readfile('mol2',os.path.join(directory,pdb_id+'_ligand.mol2')))
    pdb = next(pybel.readfile('pdb',os.path.join(directory,pdb_id+'_protein.pdb')))
    core_complexes.append((pdb,ligand))   


os.makedirs(os.path.dirname(TEST_GRIDS), exist_ok=True)
with h5py.File(TEST_GRIDS, 'w') as h5f:
    num_core = len(core_complexes)
    sample_grid = Feature.grid(*Feature.get_features(core_complexes[0][0],1), *Feature.get_features(core_complexes[0][1],0), rotations=0)
    grid_shape = sample_grid.shape[1:]  # (20,20,20,28)
    dset = h5f.create_dataset('core_grids', shape=(num_core,)+grid_shape, dtype=np.float32)
    for idx, mols in enumerate(core_complexes):
        print(f"Processing core complex {idx+1}/{num_core}...")
        coords1, features1 = Feature.get_features(mols[0],1)
        coords2, features2 = Feature.get_features(mols[1],0)
        center=(np.max(coords2,axis=0)+np.min(coords2,axis=0))/2
        coords=np.concatenate([coords1,coords2],axis = 0)
        features=np.concatenate([features1,features2],axis = 0)
        assert len(coords) == len(features)
        coords = coords-center
        grid=Feature.grid(coords,features,rotations=0)[0]  
        dset[idx] = grid

print("Processing training complexes...")
train_complexes = []
for directory in train_new_dirs:
    pdb_id = os.path.split(directory)[1]
    ligand = next(pybel.readfile('mol2',os.path.join(directory,pdb_id+'_ligand.mol2')))
    pdb = next(pybel.readfile('pdb',os.path.join(directory,pdb_id+'_protein.pdb')))
    train_complexes.append((pdb,ligand))   


os.makedirs(os.path.dirname(TRAIN_GRIDS), exist_ok=True)
with h5py.File(TRAIN_GRIDS, 'w') as h5f:
    num_train = len(train_complexes)
    sample_grid = Feature.grid(*Feature.get_features(train_complexes[0][0],1), *Feature.get_features(train_complexes[0][1],0))
    grid_shape = sample_grid.shape[1:]  # (20,20,20,28)
    dset = h5f.create_dataset('train_grids', shape=(num_train*10,)+grid_shape, dtype=np.float32)
    idx = 0
    for mols in train_complexes:
        print(f"Processing training complex {idx//10+1}/{num_train}...")
        coords1, features1 = Feature.get_features(mols[0],1)
        coords2, features2 = Feature.get_features(mols[1],0)
        center=(np.max(coords2,axis=0)+np.min(coords2,axis=0))/2
        coords=np.concatenate([coords1,coords2],axis = 0)
        features=np.concatenate([features1,features2],axis = 0)
        assert len(coords) == len(features)
        coords = coords-center
        grids=Feature.grid(coords,features)  # shape: (10, 20, 20, 20, 28)
        for rot in range(grids.shape[0]):
            dset[idx] = grids[rot]
            idx += 1

print("All done!")