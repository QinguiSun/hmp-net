import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
from functools import partial
import sys
sys.path.append('../')

from experiments.utils.train_utils import run_experiment
from models.hmp.egnn_hmp import HMP_EGNNModel
from models.hmp.tfn_hmp import HMP_TFNModel
from models.hmp.gvpgnn_hmp import HMP_GVPGNNModel

def create_kchains(k):
    assert k >= 2
    dataset = []
    atoms = torch.LongTensor([0] + [0] + [0]*(k-1) + [0])
    edge_index = torch.LongTensor([[i for i in range((k+2) - 1)], [i for i in range(1, k+2)]])
    pos1 = torch.FloatTensor([[-4, -3, 0]] + [[0, 5*i, 0] for i in range(k)] + [[4, 5*(k-1) + 3, 0]])
    center_of_mass1 = torch.mean(pos1, dim=0)
    pos1 = pos1 - center_of_mass1
    y1 = torch.LongTensor([0])
    data1 = Data(atoms=atoms, edge_index=to_undirected(edge_index), pos=pos1, y=y1)
    dataset.append(data1)

    pos2 = torch.FloatTensor([[4, -3, 0]] + [[0, 5*i, 0] for i in range(k)] + [[4, 5*(k-1) + 3, 0]])
    center_of_mass2 = torch.mean(pos2, dim=0)
    pos2 = pos2 - center_of_mass2
    y2 = torch.LongTensor([1])
    data2 = Data(atoms=atoms, edge_index=to_undirected(edge_index), pos=pos2, y=y2)
    dataset.append(data2)

    return dataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k = 4
    dataset = create_kchains(k=k)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    models_to_test = {
        "HMP-EGNN": HMP_EGNNModel,
        "HMP-TFN": HMP_TFNModel,
        "HMP-GVP": HMP_GVPGNNModel,
    }

    for name, ModelClass in models_to_test.items():
        print(f"--- Running Experiment for {name} ---")
        for num_layers in range(k // 2, k + 3):
            print(f"  Number of layers: {num_layers}")
            model = ModelClass(num_layers=num_layers, in_dim=1, out_dim=2).to(device)
            run_experiment(
                model, dataloader, val_loader, test_loader,
                n_epochs=10, n_times=1, device=device, verbose=False
            )
        print(f"--- Finished Experiment for {name} ---\n")

if __name__ == '__main__':
    main()
