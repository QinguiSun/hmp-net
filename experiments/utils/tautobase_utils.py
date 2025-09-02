import os
import re
import torch
from torch_geometric.data import Dataset, Data
from glob import glob
from tqdm import tqdm

# Mapping from atomic symbol to atomic number
ATOMIC_NUM_MAP = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
    'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
    'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
    'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86
}

def read_xyz_file(filepath):
    """
    Reads an XYZ file and extracts atomic numbers, positions, and energy.
    The energy is assumed to be in the comment line.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    num_atoms = int(lines[0].strip())

    # Extract energy from the comment line (2nd line)
    # Example comment: "energy=-1192.3350663700505"
    comment = lines[1].strip()
    try:
        # More robustly find the float value in the comment
        energy_str = re.findall(r"[-+]?\d*\.\d+|\d+", comment)[-1]
        energy = float(energy_str)
    except (IndexError, ValueError):
        # Fallback if energy is not found or parsing fails
        energy = 0.0
        # print(f"Warning: Could not parse energy from comment in {filepath}. Defaulting to 0.0.")

    atoms = []
    positions = []
    for i in range(2, 2 + num_atoms):
        line = lines[i].strip().split()
        symbol = line[0]
        pos = [float(x) for x in line[1:4]]
        atoms.append(ATOMIC_NUM_MAP[symbol])
        positions.append(pos)

    atomic_numbers = torch.tensor(atoms, dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float)

    return Data(z=atomic_numbers, pos=pos, y=torch.tensor([energy], dtype=torch.float), natoms=torch.tensor([num_atoms], dtype=torch.long))

class MolecularDataset(Dataset):
    """
    A PyTorch Geometric dataset for loading molecular data from .xyz files from a single directory.
    """
    def __init__(self, root, transform=None, pre_transform=None):
        self.file_paths = sorted(glob(os.path.join(root, '*.xyz')))
        super().__init__(root, transform, pre_transform)

    def len(self):
        return len(self.file_paths)

    def get(self, idx):
        filepath = self.file_paths[idx]
        data = read_xyz_file(filepath)
        return data

class TautobaseDataset(Dataset):
    """
    A PyTorch Geometric dataset for loading tautomer pairs from the Tautobase dataset.
    It assumes the directory structure is:
    - root/
      - A_xyz/
        - a_1.xyz, a_2.xyz, ...
      - B_xyz/
        - b_1.xyz, b_2.xyz, ...
    """
    def __init__(self, root, transform=None, pre_transform=None):
        self.paths_a = sorted(glob(os.path.join(root, 'A_xyz', '*.xyz')), key=self._sort_key)
        self.paths_b = sorted(glob(os.path.join(root, 'B_xyz', '*.xyz')), key=self._sort_key)

        if len(self.paths_a) != len(self.paths_b):
            raise ValueError("Mismatch in number of molecules in A_xyz and B_xyz directories.")

        super().__init__(root, transform, pre_transform)

    @staticmethod
    def _sort_key(s):
        # Extracts the number from filenames like 'a_1.xyz' or 'b_10.xyz'
        return int(re.search(r'\d+', os.path.basename(s)).group())

    def len(self):
        return len(self.paths_a)

    def get(self, idx):
        path_a = self.paths_a[idx]
        path_b = self.paths_b[idx]

        data_a = read_xyz_file(path_a)
        data_b = read_xyz_file(path_b)

        # The 'idx' in the data object will be useful for matching pairs later
        data_a.pair_id = idx
        data_b.pair_id = idx

        return data_a, data_b

if __name__ == '__main__':
    # Example usage and testing
    print("--- Testing MolecularDataset (QM9 sample) ---")
    qm9_sample_path = '../../dataset_sample/qm9'
    if os.path.exists(qm9_sample_path):
        qm9_dataset = MolecularDataset(root=qm9_sample_path)
        print(f"Found {len(qm9_dataset)} molecules.")
        if len(qm9_dataset) > 0:
            data_sample = qm9_dataset[0]
            print("Sample data object:", data_sample)
            print("Energy (y):", data_sample.y)
            print("Num atoms:", data_sample.natoms.item())
    else:
        print(f"Path not found: {qm9_sample_path}")

    print("\n--- Testing TautobaseDataset (QTautobase/QM9 sample) ---")
    tautobase_sample_path = '../../dataset_sample/QTautobase/QM9'
    if os.path.exists(tautobase_sample_path):
        tautobase_dataset = TautobaseDataset(root=tautobase_sample_path)
        print(f"Found {len(tautobase_dataset)} tautomer pairs.")
        if len(tautobase_dataset) > 0:
            pair_sample = tautobase_dataset[0]
            data_a, data_b = pair_sample
            print("Sample pair (A):", data_a)
            print("Energy A (y):", data_a.y)
            print("Sample pair (B):", data_b)
            print("Energy B (y):", data_b.y)
    else:
        print(f"Path not found: {tautobase_sample_path}")
