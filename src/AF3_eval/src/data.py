#!/usr/bin/env python
# coding: utf-8
import os

os.environ["OPENBABEL_WARNINGS"] = "0"

from glob import glob
import numpy as np
import random
import h5py
import pandas as pd
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa

class Feature_extractor():
    def __init__(self):
        self.atom_codes = {}
        self.others = ([3,4,5,11,12,13]+list(range(19,32))+list(range(37,51))+list(range(55,84)))
        atom_types = [
            1,
            (6,1),(6,2),(6,3),
            (7,1),(7,2),(7,3),
            8,
            15,
            (16,2),(16,3),  
            34,
            [9,17,35,53],
            self.others
        ]
        for i, j in enumerate(atom_types):
            if type(j) is list:
                for k in j:
                    self.atom_codes[k] = i
            else:
                self.atom_codes[j] = i         
        self.sum_atom_types = len(atom_types)

    def encode(self, atomic_num, molprotein):
        key = atomic_num
        encoding = np.zeros(self.sum_atom_types*2)
        if key not in self.atom_codes and isinstance(key, tuple):
            key=83
        if molprotein == 1:
            encoding[self.atom_codes[key]] = 1.0
        else:
            encoding[self.sum_atom_types+self.atom_codes[key]] = 1.0
        
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
        
        grid=np.zeros((20,20,20,features.shape[1]),dtype=np.float32)
        for i in range(len(coords)):
            coord=coords[i]
            tmpx=abs(coord[0]-x)
            tmpy=abs(coord[1]-y)
            tmpz=abs(coord[2]-z)
            if np.max(tmpx)<=19.5 and np.max(tmpy)<=19.5 and np.max(tmpz) <=19.5:
                grid[np.argmin(tmpx),np.argmin(tmpy),np.argmin(tmpz)] += features[i]
        grids.append(grid)
        
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

CORE_PATH = r'data/chai_results_cif'
core_dirs = glob(os.path.join(CORE_PATH, '*'))
core_dirs.sort()

sfcnn_csv = r'data/sfcnn_out.csv'
core2016_csv = r'data/core_affinity_final.csv'

sfcnn_df = pd.read_csv(sfcnn_csv)
sfcnn_labels = []
for directory in core_dirs:
    cid = os.path.basename(directory)
    row = sfcnn_df[sfcnn_df['pdbid'] == cid]
    if not row.empty:
        sfcnn_labels.append(float(row.iloc[0]['affinity']))
sfcnn_labels = np.array(sfcnn_labels, dtype=np.float32)

core2016_df = pd.read_csv(core2016_csv)
core2016_labels = []
for directory in core_dirs:
    cid = os.path.basename(directory)
    row = core2016_df[core2016_df['pdbid'] == cid]
    if not row.empty:
        core2016_labels.append(float(row.iloc[0]['affinity']))
core2016_labels = np.array(core2016_labels, dtype=np.float32)

CORE_LABELS = r'data/chai_hdf5/core_label.h5'
CORE_2016_LABELS = r'data/chai_hdf5/core_2016_label.h5'
CORE_sfcnn_LABELS = r'data/chai_hdf5/core_sfcnn_label.h5'
TEST_GRIDS  = r'data/chai_hdf5/core_grids.h5'

os.makedirs(os.path.dirname(CORE_LABELS), exist_ok=True)
with h5py.File(CORE_2016_LABELS, 'w') as f:
    f.create_dataset('core_label', data=core2016_labels)
with h5py.File(CORE_sfcnn_LABELS, 'w') as f:
    f.create_dataset('core_label', data=sfcnn_labels)

core_complexes = []
parser = MMCIFParser(QUIET=True)

element_to_num = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'CL': 17, 'K': 19, 'CA': 20, 'ZN': 30, 'MG': 12, 'FE': 26, 'CU': 29, 'MN': 25, 'BR': 35, 'I': 53, 'SE': 34
}
def get_atomic_number(atom):
    symbol = atom.element.strip().upper()
    return element_to_num.get(symbol, 6) 

class AtomWrapper:
    def __init__(self, atom):
        self.coords = atom.coord
        self.atomicnum = get_atomic_number(atom)
        self.hyb = 1  

for directory in core_dirs:
    cid = os.path.basename(directory)
    cif_path = os.path.join(directory, cid + '.cif')
    structure = parser.get_structure(cid, cif_path)
    protein_atoms = []
    ligand_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True):
                    for atom in residue:
                        protein_atoms.append(atom)
                else:
                    for atom in residue:
                        ligand_atoms.append(atom)
    pdb_atoms = [AtomWrapper(atom) for atom in protein_atoms]
    lig_atoms = [AtomWrapper(atom) for atom in ligand_atoms]
    core_complexes.append((pdb_atoms, lig_atoms))

os.makedirs(os.path.dirname(TEST_GRIDS), exist_ok=True)
with h5py.File(TEST_GRIDS, 'w') as h5f:
    num_core = len(core_complexes)

    coords_p, feats_p = Feature.get_features(core_complexes[0][0], 1)
    coords_l, feats_l = Feature.get_features(core_complexes[0][1], 0)
    sample_grid = Feature.grid(
        np.concatenate([coords_p, coords_l], axis=0),
        np.concatenate([feats_p, feats_l], axis=0),
        rotations=0
    )
    grid_shape = sample_grid.shape[1:]
    dset = h5f.create_dataset('core_grids', shape=(num_core,) + grid_shape, dtype=np.float32)

    for idx, (pdb_mol, lig_mol) in enumerate(core_complexes):
        coords1, feats1 = Feature.get_features(pdb_mol, 1)
        coords2, feats2 = Feature.get_features(lig_mol, 0)
        center = (np.max(coords2, axis=0) + np.min(coords2, axis=0)) / 2
        coords  = np.concatenate([coords1, coords2], axis=0) - center
        feats   = np.concatenate([feats1, feats2], axis=0)
        dset[idx] = Feature.grid(coords, feats, rotations=0)[0]