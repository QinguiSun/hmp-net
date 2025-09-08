# tautobase_utils.py
import os
import re
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data 
from glob import glob
from tqdm import tqdm
from torch_geometric.utils import coalesce

# 优先使用 PyG 自带的 radius_graph；若不可用则回退到 torch_cluster
try:
    # PyG 2.4+ 提供了 torch_geometric.nn.radius_graph
    from torch_geometric.nn import radius_graph as pyg_radius_graph
    _HAS_PYG_RADIUS = True
except Exception:
    _HAS_PYG_RADIUS = False
try:
    from torch_cluster import radius_graph as tc_radius_graph
    _HAS_TC_RADIUS = True
except Exception:
    _HAS_TC_RADIUS = False

def _radius_graph(pos: torch.Tensor,
                  r_cutoff: float,
                  batch: torch.Tensor = None,
                  max_num_neighbors: int = 64,
                  loop: bool = False) -> torch.Tensor:
    """
    A thin wrapper that calls an available radius_graph implementation.
    - pos: (N, 3) float tensor
    - r_cutoff: float cutoff radius (in the same unit as pos)
    - batch: optional (N,) tensor if you build multi-graph at once
    - max_num_neighbors: cap neighbors per node to avoid dense graphs
    - loop: include self loops or not
    Returns:
      edge_index: (2, E) long tensor
    """
    if _HAS_PYG_RADIUS:
        # torch_geometric.nn.radius_graph API
        return pyg_radius_graph(pos, r_cutoff, batch=batch,
                                max_num_neighbors=max_num_neighbors,
                                loop=loop)
    elif _HAS_TC_RADIUS:
        # torch_cluster.radius_graph API
        return tc_radius_graph(pos, r_cutoff, batch=batch,
                               max_num_neighbors=max_num_neighbors,
                               loop=loop)
    else:
        raise ImportError(
            "No radius_graph implementation found. "
            "Please install torch-geometric (>=2.4) or torch-cluster."
        )

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

# MODIFICATION START: 创建一个辅助函数来处理非标准的科学计数法字符串
def _parse_scientific_notation(s: str) -> float:
    """
    Converts a string with non-standard scientific notation to a float.
    Handles formats like '1.23*^-4', '1.23*10^-4', '1.23D-04'.
    """
    # 规范化字符串，将所有变体替换为标准的 'e'
    normalized_s = s.replace('D', 'e').replace('d', 'e')
    normalized_s = normalized_s.replace('*10^', 'e')
    normalized_s = normalized_s.replace('*^', 'e')
    return float(normalized_s)
# MODIFICATION END

def read_xyz_file(filepath, r_cutoff: float = 5.0, max_num_neighbors: int = 64):
    """
    Reads an XYZ file and extracts atomic numbers, positions, and energy.
    The energy is assumed to be in the comment line.
    Also dynamically builds a radius graph with the given cutoff.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    num_atoms = int(lines[0].strip())

    # Extract energy from the comment line (2nd line)
    # Example comment: "energy=-1192.3350663700505"
    comment = lines[1].strip()
    try:
        # MODIFICATION START: 修正了能量解析的逻辑
        # 错误原因：之前的代码错误地将整个注释行字符串传递给 _parse_scientific_notation，
        # 这会导致 float() 转换失败。
        # 正确逻辑：1. 对整个注释行字符串进行文本替换，使其格式规范化。
        #           2. 使用正则表达式从规范化的字符串中找出所有数字。
        #           3. 将找到的最后一个数字字符串转换为浮点数。

        # 1. 对整个注释行字符串进行文本替换，以处理非标准科学计数法
        normalized_comment = comment.replace('D', 'e').replace('d', 'e')
        normalized_comment = normalized_comment.replace('*10^', 'e')
        normalized_comment = normalized_comment.replace('*^', 'e')

        # 2. 在规范化后的字符串中查找所有数字模式（包括整数、浮点数和标准科学计数法）
        numeric_strings = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", normalized_comment)
        
        if not numeric_strings:
            raise ValueError("No numeric value found in comment line")

        # 3. 最后一个找到的数字通常是能量值
        energy = float(numeric_strings[-1])
        # MODIFICATION END

    except (IndexError, ValueError):
        energy = 0.0
        # 仅在调试时取消注释，以避免大量输出
        print(f"Warning: Could not parse energy from comment in {filepath}. Defaulting to 0.0.")
        
    atoms = []
    positions = []
    for i in range(2, 2 + num_atoms):
        line = lines[i].strip().split()
        symbol = line[0]
        
        # MODIFICATION START: 在解析坐标时使用辅助函数
        # 这样可以正确处理坐标值中出现的非标准科学计数法
        try:
            pos = [_parse_scientific_notation(x) for x in line[1:4]]
        except (ValueError, IndexError) as e:
            print(f"Error parsing coordinates in file: {filepath} on line {i+1}: '{lines[i].strip()}'")
            # 如果某一行格式错误，可以选择跳过该文件或用零填充
            # 这里我们选择用零填充并继续，但打印警告
            pos = [0.0, 0.0, 0.0]
        # MODIFICATION END
        
        atoms.append(ATOMIC_NUM_MAP[symbol])
        positions.append(pos)

    atomic_numbers = torch.tensor(atoms, dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float)

    # --- 动态构图：基于半径的邻接 ---
    # 单分子样本，不需要 batch；loop=False 去除自环
    edge_index = _radius_graph(
        pos=pos, r_cutoff=r_cutoff, batch=None,
        max_num_neighbors=max_num_neighbors, loop=False
    )
    # 无向化：添加反向边并做去重（coalesce）
    rev = edge_index.flip(0)
    edge_index = torch.cat([edge_index, rev], dim=1)
    edge_index = coalesce(edge_index, num_nodes=pos.size(0))

    return Data(
        atoms=atomic_numbers,
        pos=pos,
        edge_index=edge_index,
        y=torch.tensor([energy], dtype=torch.float),
        natoms=torch.tensor([num_atoms], dtype=torch.long),
    )

class MolecularDataset(Dataset):
    """
    A PyTorch Geometric dataset for loading molecular data from .xyz files from a single directory.
    """
    def __init__(self, root, transform=None, 
                r_cutoff: float = 5.0, max_num_neighbors: int = 64):
        self.root = root
        self.transform = transform
        self.file_paths = sorted(glob(os.path.join(root, '*.xyz')))
        self.r_cutoff = r_cutoff
        self.max_num_neighbors = max_num_neighbors 

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        data = read_xyz_file(
            filepath,
            r_cutoff=self.r_cutoff,
            max_num_neighbors=self.max_num_neighbors
        )
        
        if self.transform:
            data = self.transform(data)
            
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
    def __init__(self, root, transform=None, 
                 r_cutoff: float = 5.0, max_num_neighbors: int = 64):
        self.root = root
        self.transform = transform
        self.paths_a = sorted(glob(os.path.join(root, 'A_xyz', '*.xyz')), key=self._sort_key)
        self.paths_b = sorted(glob(os.path.join(root, 'B_xyz', '*.xyz')), key=self._sort_key)

        if len(self.paths_a) != len(self.paths_b):
            raise ValueError("Mismatch in number of molecules in A_xyz and B_xyz directories.")

        self.r_cutoff = r_cutoff
        self.max_num_neighbors = max_num_neighbors

    @staticmethod
    def _sort_key(s):
        # Extracts the number from filenames like 'a_1.xyz' or 'b_10.xyz'
        return int(re.search(r'\d+', os.path.basename(s)).group())

    def __len__(self):
        return len(self.paths_a)

    def __getitem__(self, idx):
        path_a = self.paths_a[idx]
        path_b = self.paths_b[idx]

        data_a = read_xyz_file(
            path_a,
            r_cutoff=self.r_cutoff,
            max_num_neighbors=self.max_num_neighbors
        )
        data_b = read_xyz_file(
            path_b,
            r_cutoff=self.r_cutoff,
            max_num_neighbors=self.max_num_neighbors
        )

        # The 'idx' in the data object will be useful for matching pairs later
        data_a.pair_id = idx
        data_b.pair_id = idx
        
        if self.transform:
            data_a = self.transform(data_a)
            data_b = self.transform(data_b)
            
        return data_a, data_b

if __name__ == '__main__':
    # Example usage and testing
    print("--- Testing MolecularDataset (QM9 sample) ---")
    qm9_sample_path = '../../dataset_sample/qm9'
    if os.path.exists(qm9_sample_path):
        qm9_dataset = MolecularDataset(root=qm9_sample_path, r_cutoff=5.0, max_num_neighbors=64)
        print(f"Found {len(qm9_dataset)} molecules.")
        if len(qm9_dataset) > 0:
            data_sample = qm9_dataset[0]
            print("Sample data object:", data_sample)
            print("Energy (y):", data_sample.y)
            print("Num atoms:", data_sample.natoms.item())
            print("edge_index shape:", data_sample.edge_index.shape)
    else:
        print(f"Path not found: {qm9_sample_path}")

    print("\n--- Testing TautobaseDataset (QTautobase/QM9 sample) ---")
    tautobase_sample_path = '../../dataset_sample/QTautobase/QM9'
    if os.path.exists(tautobase_sample_path):
        tautobase_dataset = TautobaseDataset(root=tautobase_sample_path, r_cutoff=5.0, max_num_neighbors=64)
        print(f"Found {len(tautobase_dataset)} tautomer pairs.")
        if len(tautobase_dataset) > 0:
            pair_sample = tautobase_dataset[0]
            data_a, data_b = pair_sample
            print("Sample pair (A):", data_a)
            print("Energy A (y):", data_a.y)
            print("Sample pair (B):", data_b)
            print("Energy B (y):", data_b.y)
            print("A edge_index shape:", data_a.edge_index.shape)
            print("B edge_index shape:", data_b.edge_index.shape)
    else:
        print(f"Path not found: {tautobase_sample_path}")