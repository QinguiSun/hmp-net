# models/dataset_input.py
import torch
from torch_geometric.data import Data, Dataset

SYMBOL2Z = {
    "H": 1,
    "C": 6,
    "Na": 11,
    "Cl": 17,
    "Ag": 47,
}

class InputDataDataset(Dataset):
    def __init__(self, path: str, transform=None, pre_transform=None, keep_ref_charges: bool = True):
        super().__init__(None, transform, pre_transform)
        self.path = path
        self.keep_ref_charges = keep_ref_charges
        self._blocks = self._index_blocks()

    def _index_blocks(self):
        with open(self.path, "r") as f:
            lines = f.readlines()

        blocks = []
        start = None
        for i, line in enumerate(lines):
            s = line.strip()
            if s == "begin":
                start = i
            elif s == "end" and start is not None:
                blocks.append((start, i))
                start = None

        self._lines = lines
        return blocks

    def len(self):
        return len(self._blocks)

    def get(self, idx: int):
        start, end = self._blocks[idx]
        lines = self._lines[start:end+1]

        pos_list, z_list, q_atom_list, f_list = [], [], [], []
        energy = None
        q_tot = 0.0

        for line in lines:
            s = line.strip()
            if not s:
                continue
            parts = s.split()

            if parts[0] == "atom":
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                sym = parts[4]
                if sym not in SYMBOL2Z:
                    raise KeyError(f"Unknown element symbol {sym}. Extend SYMBOL2Z.")
                znum = SYMBOL2Z[sym]

                q_atom = float(parts[5])
                fx, fy, fz = float(parts[7]), float(parts[8]), float(parts[9])

                pos_list.append([x, y, z])
                z_list.append(znum)
                q_atom_list.append(q_atom)
                f_list.append([fx, fy, fz])

            elif parts[0] == "energy":
                energy = float(parts[1])

            elif parts[0] == "charge":
                q_tot = float(parts[1])

        if energy is None:
            raise ValueError(f"Missing energy in block {idx}")

        data = Data(
            pos=torch.tensor(pos_list, dtype=torch.float32),
            z=torch.tensor(z_list, dtype=torch.long),
            y=torch.tensor([energy], dtype=torch.float32),
            forces=torch.tensor(f_list, dtype=torch.float32),
            q_tot=torch.tensor([q_tot], dtype=torch.float32),
        )

        if self.keep_ref_charges:
            data.q_atom_ref = torch.tensor(q_atom_list, dtype=torch.float32)

        return data
