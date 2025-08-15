import sys
sys.path.append('./')

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import e3nn
from functools import partial

from experiments.utils.train_utils import run_experiment
from models.hmp.mace_hmp import HMP_MACEModel

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def create_kchains(k):
    assert k >= 2

    dataset = []

    # Graph 0
    atoms = torch.LongTensor( [0] + [0] + [0]*(k-1) + [0] )
    edge_index = torch.LongTensor( [ [i for i in range((k+2) - 1)], [i for i in range(1, k+2)] ] )
    pos = torch.FloatTensor(
        [[-4, -3, 0]] +
        [[0, 5*i , 0] for i in range(k)] +
        [[4, 5*(k-1) + 3, 0]]
    )
    center_of_mass = torch.mean(pos, dim=0)
    pos = pos - center_of_mass
    y = torch.LongTensor([0])  # Label 0
    data1 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data1.edge_index = to_undirected(data1.edge_index)
    dataset.append(data1)

    # Graph 1
    atoms = torch.LongTensor( [0] + [0] + [0]*(k-1) + [0] )
    edge_index = torch.LongTensor( [ [i for i in range((k+2) - 1)], [i for i in range(1, k+2)] ] )
    pos = torch.FloatTensor(
        [[4, -3, 0]] +
        [[0, 5*i , 0] for i in range(k)] +
        [[4, 5*(k-1) + 3, 0]]
    )
    center_of_mass = torch.mean(pos, dim=0)
    pos = pos - center_of_mass
    y = torch.LongTensor([1])  # Label 1
    data2 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data2.edge_index = to_undirected(data2.edge_index)
    dataset.append(data2)

    return dataset

k = 4

# Create dataset
dataset = create_kchains(k=k)

# Create dataloaders
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(dataset, batch_size=2, shuffle=False)

# --- Run HMP-MACE Experiment ---
print("\n" + "="*40)
print("Running HMP-MACE Experiment")
print("="*40)

for num_layers in range(k // 2, k + 3):
    print(f"\nNumber of layers: {num_layers}")

    correlation = 2 # Same as baseline MACE

    # Instantiate HMP-MACE Model
    # Using hyperparameters from paper (Section 4.3) where available
    hmp_mace_model = HMP_MACEModel(
        num_layers=num_layers,
        in_dim=1,
        out_dim=2,
        correlation=correlation,
        master_rate=0.25 # From paper
    ).to(device)

    run_experiment(
        hmp_mace_model,
        dataloader,
        val_loader,
        test_loader,
        n_epochs=1, # one epoch is enough to trigger the error
        n_times=1,
        device=device,
        verbose=False,
        tau_annealing=True
    )
