import os
import json
import shutil
import zipfile
from Bio.PDB import MMCIFParser, PDBIO
from rdkit import Chem
from rdkit.Chem import AllChem

# Paths (adjust as needed)
index_folder_root = r'data\chai_results_zip\target3'
target_json = r'src\sfcnn\targets\target3.json'
output_root = r'data\chai_results_cif'

# Load index-to-protein mapping
with open(target_json, encoding='utf-8') as f:
    data = json.load(f)
index_to_protein = {str(v['index']): k for k, v in data.items()}

def extract_ligand_pdb_to_mol2(pdb_path, mol2_path, ligand_resname="LIG2", chain_id="B"):
    ligand_lines = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("HETATM"):
                ligand_lines.append(line)
    if not ligand_lines:
        print("Ligand not found.")
        return

    tmp_pdb = os.path.join(os.path.dirname(mol2_path), "ligand_tmp.pdb")
    with open(tmp_pdb, "w") as f:
        for line in ligand_lines:
            f.write(line)
        f.write("END\n")

    mol = Chem.MolFromPDBFile(tmp_pdb, removeHs=False)
    if mol is None:
        print("Failed to parse ligand PDB.")
        return

    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol)

    Chem.MolToMol2File(mol, mol2_path)
    print(f"Ligand written to {mol2_path}")

for fname in os.listdir(index_folder_root):
    if not fname.lower().endswith('.zip'):
        continue
    index = os.path.splitext(fname)[0]
    protein_name = index_to_protein.get(index)
    if not protein_name:
        continue
    zip_path = os.path.join(index_folder_root, fname)
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        cif_files = [f for f in zipf.namelist() if f.lower().endswith('pred.rank_0.cif')]
        if not cif_files:
            print(f"No pred.rank_0.cif file found in {zip_path}")
            continue
        cif_file = cif_files[0]
        dest_folder = os.path.join(output_root, protein_name)
        os.makedirs(dest_folder, exist_ok=True)
        dest_cif_path = os.path.join(dest_folder, f"{protein_name}.cif")
        # Decode and save .cif file
        with zipf.open(cif_file) as source, open(dest_cif_path, 'wb') as target:
            shutil.copyfileobj(source, target)
        print(f"Extracted {cif_file} from {zip_path} -> {dest_cif_path}")

        # # Convert .cif to .pdb
        # pdb_path = os.path.join(dest_folder, f"{protein_name}.pdb")
        # parser = MMCIFParser()
        # structure = parser.get_structure("protein", dest_cif_path)
        # io = PDBIO()
        # io.set_structure(structure)
        # io.save(pdb_path)
        # print(f"PDB written to {pdb_path}")

        # # Convert .pdb to .mol2 (ligand extraction)
        # mol2_path = os.path.join(dest_folder, f"{protein_name}.mol2")
        # extract_ligand_pdb_to_mol2(pdb_path, mol2_path)