from Bio.PDB import MMCIFParser, PDBIO

cif_file = "src\sfcnn-origin\input\pred.rank_0.cif"
pdb_file = "output.pdb"

parser = MMCIFParser()
structure = parser.get_structure("protein", cif_file)

io = PDBIO()
io.set_structure(structure)
io.save(pdb_file)

# Save as: extract_ligand_to_mol2.py
from rdkit import Chem
from rdkit.Chem import AllChem

def extract_ligand_pdb_to_mol2(pdb_path, mol2_path, ligand_resname="LIG2", chain_id="B"):
    # Read PDB file and extract ligand atoms
    ligand_lines = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("HETATM"):
                ligand_lines.append(line)
    if not ligand_lines:
        print("Ligand not found.")
        return

    # Write ligand to temporary PDB
    tmp_pdb = "ligand_tmp.pdb"
    with open(tmp_pdb, "w") as f:
        for line in ligand_lines:
            f.write(line)
        f.write("END\n")

    # Load with RDKit
    mol = Chem.MolFromPDBFile(tmp_pdb, removeHs=False)
    if mol is None:
        print("Failed to parse ligand PDB.")
        return

    # Generate 3D coordinates if missing
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol)

    # Write to MOL2
    Chem.MolToMol2File(mol, mol2_path)
    print(f"Ligand written to {mol2_path}")

# Usage
extract_ligand_pdb_to_mol2("output.pdb", "ligand.mol2")