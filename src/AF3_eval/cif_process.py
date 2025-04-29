import os
import subprocess
from Bio.PDB import MMCIFParser, PDBIO, Select

class ProteinSelect(Select):
    def accept_residue(self, residue):
        # Only select standard residues (ATOM records)
        return residue.id[0] == " "

class LigandSelect(Select):
    def accept_residue(self, residue):
        # Only select HETATM residues (ligands, ions, etc.)
        return residue.id[0] != " "

def extract_protein_and_ligand(cif_path, out_dir):
    protein_name = os.path.splitext(os.path.basename(cif_path))[0]
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(protein_name, cif_path)

    protein_pdb = os.path.join(out_dir, f'{protein_name}_protein.pdb')
    io = PDBIO()
    io.set_structure(structure)
    io.save(protein_pdb, select=ProteinSelect())

    ligand_pdb = os.path.join(out_dir, f'{protein_name}_ligand.pdb')
    io.save(ligand_pdb, select=LigandSelect())

    ligand_mol2 = None
    if os.path.exists(ligand_pdb) and os.path.getsize(ligand_pdb) > 0:
        ligand_mol2 = os.path.join(out_dir, f'{protein_name}_ligand.mol2')
        subprocess.run(['obabel', ligand_pdb, '-O', ligand_mol2], check=True)
        os.remove(ligand_pdb)
    else:
        print(f"No HETATM residues found in {cif_path}")
        if os.path.exists(ligand_pdb):
            os.remove(ligand_pdb)

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
                extract_protein_and_ligand(cif_path, out_folder)

if __name__ == "__main__":
    process_all_cifs(
        input_dir='data/chai_results_cif',
        output_dir='data/chai_results_pdb'
    )