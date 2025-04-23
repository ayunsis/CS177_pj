import os
import json
from rdkit import Chem
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

CORESET_DIR = 'data/coreset'
OUTPUT_JSON = 'coreset_smiles_chains.json'

def get_ligand_smiles(mol2_path):
    mol = Chem.MolFromMol2File(mol2_path)
    if mol:
        return Chem.MolToSmiles(mol)
    return None

def get_protein_chains(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    chains = {}
    seen_sequences = set()
    for model in structure:
        for chain in model:
            seq = ""
            for residue in chain:
                if residue.get_id()[0] == " ":
                    seq += seq1(residue.get_resname())
            if seq and seq not in seen_sequences:
                chains[chain.id] = seq
                seen_sequences.add(seq)
        break  # Only first model
    return chains

def process_coreset(coreset_dir):
    data = {}
    for instance in os.listdir(coreset_dir):
        instance_dir = os.path.join(coreset_dir, instance)
        ligand_path = os.path.join(instance_dir, f"{instance}_ligand.mol2")
        protein_path = os.path.join(instance_dir, f"{instance}_protein.pdb")
        if not (os.path.isfile(ligand_path) and os.path.isfile(protein_path)):
            continue
        smiles = get_ligand_smiles(ligand_path)
        chains = get_protein_chains(protein_path)
        data[instance] = {
            "smiles": smiles,
            "chains": chains
        }
    return data

if __name__ == "__main__":
    result = process_coreset(CORESET_DIR)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Output written to {OUTPUT_JSON}")