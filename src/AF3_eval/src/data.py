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
        return np.stack(grids, axis=0)  # shape: (rotations+1, 20, 20, 20, features)

Feature = Feature_extractor()

CORE_PATH = r'data/chai_results_pdb'
print("Preparing core directory...")
core_dirs = glob(os.path.join(CORE_PATH, '*'))
core_dirs.sort()

# TODO modify core label to the sfcnn_one and core_2016 label
INDEX_PATH = r'data/PDBbind_v2019_plain_text_index/index/INDEX_general_PL_data.2019'
print("Loading affinity data...")
affinity = {}
with open(INDEX_PATH, 'r') as f:
    for line in f:
        if not line.startswith('#'):
            parts = line.split()
            affinity[parts[0]] = float(parts[3])

# Prepare core label array
core_label = []
for directory in core_dirs:
    cid = os.path.basename(directory)
    if cid in affinity:
        core_label.append(affinity[cid])
core_label = np.array(core_label, dtype=np.float32)

CORE_LABELS = r'data/chai_hdf5/core_label.h5'
TEST_GRIDS  = r'data/chai_hdf5/core_grids.h5'

print("Saving core label data...")
os.makedirs(os.path.dirname(CORE_LABELS), exist_ok=True)
with h5py.File(CORE_LABELS, 'w') as f:
    f.create_dataset('core_label', data=core_label)

print("Processing core complexes...")
core_complexes = []
for directory in core_dirs:
    cid = os.path.basename(directory)
    ligand = next(pybel.readfile('mol2', os.path.join(directory, cid + '_ligand.mol2')))
    pdb    = next(pybel.readfile('pdb',  os.path.join(directory, cid + '_protein.pdb')))
    core_complexes.append((pdb, ligand))

os.makedirs(os.path.dirname(TEST_GRIDS), exist_ok=True)
with h5py.File(TEST_GRIDS, 'w') as h5f:
    num_core = len(core_complexes)
    # sample shape for dataset
    coords_p, feats_p = Feature.get_features(core_complexes[0][0], 1)
    coords_l, feats_l = Feature.get_features(core_complexes[0][1], 0)
    sample_grid = Feature.grid(
        np.concatenate([coords_p, coords_l], axis=0),
        np.concatenate([feats_p, feats_l], axis=0),
        rotations=0
    )
    grid_shape = sample_grid.shape[1:]  # (20,20,20,features)
    dset = h5f.create_dataset('core_grids', shape=(num_core,) + grid_shape, dtype=np.float32)

    for idx, (pdb_mol, lig_mol) in enumerate(core_complexes):
        print(f"Processing core complex {idx+1}/{num_core}...")
        coords1, feats1 = Feature.get_features(pdb_mol, 1)
        coords2, feats2 = Feature.get_features(lig_mol, 0)
        center = (np.max(coords2, axis=0) + np.min(coords2, axis=0)) / 2
        coords  = np.concatenate([coords1, coords2], axis=0) - center
        feats   = np.concatenate([feats1, feats2], axis=0)
        dset[idx] = Feature.grid(coords, feats, rotations=0)[0]  # just the unrotated grid

print("Core processing complete.")