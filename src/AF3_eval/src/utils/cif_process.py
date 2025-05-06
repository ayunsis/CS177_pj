import os
import subprocess

def extract_protein_and_ligand_with_openbabel(cif_path, out_dir):
    protein_name = os.path.splitext(os.path.basename(cif_path))[0]
    protein_pdb = os.path.join(out_dir, f'{protein_name}_protein.pdb')
    ligand_mol2 = os.path.join(out_dir, f'{protein_name}_ligand.mol2')

    # Convert CIF to PDB (whole structure)
    subprocess.run(['obabel', cif_path, '-O', protein_pdb], check=True)

    # Try to convert CIF to MOL2 with --separate --largest
    try:
        subprocess.run(['obabel', cif_path, '-O', ligand_mol2, '--separate', '--largest'], check=True)
    except subprocess.CalledProcessError:
        try:
            subprocess.run(['obabel', cif_path, '-O', ligand_mol2], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Open Babel failed for {cif_path}: {e}")
            ligand_mol2 = None

    return protein_pdb, ligand_mol2

def process_all_cifs(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.cif'):
                cif_path = os.path.join(root, file)
                protein_name = os.path.splitext(file)[0]
                out_folder = os.path.join(output_dir, protein_name)
                os.makedirs(out_folder, exist_ok=True)
                print(f"Processing {cif_path} -> {out_folder}")
                extract_protein_and_ligand_with_openbabel(cif_path, out_folder)

if __name__ == "__main__":
    process_all_cifs(
        input_dir='data/chai_results_cif',
        output_dir='data/chai_results_pdb'
    )