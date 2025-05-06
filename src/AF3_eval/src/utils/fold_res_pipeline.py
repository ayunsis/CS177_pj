import os
import json
import shutil
import zipfile
from Bio.PDB import MMCIFParser, PDBIO
from rdkit import Chem
from rdkit.Chem import AllChem

# Paths (adjust as needed)
index_folder_root = r'data/chai_results_zip/target2'
target_json = r'src\sfcnn/targets/target2.json'
output_root = r'data/chai_results_cif'

# Load index-to-protein mapping
with open(target_json, encoding='utf-8') as f:
    data = json.load(f)
index_to_protein = {str(v['index']): k for k, v in data.items()}

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
