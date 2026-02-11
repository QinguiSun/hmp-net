# schnet.py
from typing import Optional

import torch
from torch.nn import functional as F
from torch_geometric.nn import SchNet
from torch_geometric.nn import global_add_pool, global_mean_pool


class SchNetModel(SchNet):
    """
    SchNet model from "Schnet - a deep learning architecture for molecules and materials".

    This class extends the SchNet base class for PyG.
    """
    def __init__(
        self, 
        hidden_channels: int = 128, 
        num_embeddings: int = 1,
        out_dim: int = 1, 
        num_filters: int = 128, 
        num_layers: int = 6,
        num_gaussians: int = 50, 
        cutoff: float = 10, 
        max_num_neighbors: int = 32, 
        pool: str = 'sum'
    ):
        """
        Initializes an instance of the SchNetModel class with the provided parameters.

        Parameters:
        - hidden_channels (int): Number of channels in the hidden layers (default: 128)
        - num_embeddings (int):  (default: 1)
        - out_dim (int): Output dimension of the model (default: 1)
        - num_filters (int): Number of filters used in convolutional layers (default: 128)
        - num_layers (int): Number of convolutional layers in the model (default: 6)
        - num_gaussians (int): Number of Gaussian functions used for radial filters (default: 50)
        - cutoff (float): Cutoff distance for interactions (default: 10)
        - max_num_neighbors (int): Maximum number of neighboring atoms to consider (default: 32)
        - pool (str): Global pooling method to be used (default: "sum")
        """
        super().__init__(
            # 使用关键字参数，明确地将您类中的参数映射到基类中的参数
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_layers,      # 将 num_layers 赋给 num_interactions
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            readout=pool                      # PyG 2.x 中使用 readout，而不是 aggregation
            # dipole, mean, std, atomref 等参数可以省略，会自动使用基类的默认值
        )
        # 不使用 padding_idx
        self.embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=hidden_channels)

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]
        
        # Overwrite atom embedding and final predictor
        self.lin2 = torch.nn.Linear(hidden_channels // 2, out_dim)

    def forward(self, batch):
        
        # BUG修复: 开始
        # -----------------------------------------------------------------------------------
        # 问题描述: 由于在主脚本中使用了正确的 torch_geometric.nn.DataParallel,
        #           该包装器会负责将整个 batch 对象正确地移动到每个GPU副本上。
        # 解决方案: 删除模型内部的手动设备转移代码, 使模型保持设备无关性, 依赖包装器来管理数据位置。
        # device = self.embedding.weight.device
        # batch = batch.to(device)
        # -----------------------------------------------------------------------------------
        # BUG修复: 结束
        
        
        # 取原子序号：优先 atoms，其次 z，再次 atomic_numbers
        atoms = getattr(batch, "atoms", None)
        if atoms is None:
            atoms = getattr(batch, "z", None)
        if atoms is None:
            atoms = getattr(batch, "atomic_numbers", None)
        if atoms is None:
            raise AttributeError(
                "No atomic numbers tensor found. Expected one of: batch.atoms, batch.z, batch.atomic_numbers"
            )
        atoms = atoms.long()  # 确保是 Long

        h = self.embedding(atoms)  # (n,) -> (n, d)
        #h = self.embedding(batch.atoms)  # (n,) -> (n, d)

        row, col = batch.edge_index
        edge_weight = (batch.pos[row] - batch.pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            # # Message passing layer: (n, d) -> (n, d)
            h = h + interaction(h, batch.edge_index, edge_weight, edge_attr)

        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)  # (batch_size, out_dim)

        return out
