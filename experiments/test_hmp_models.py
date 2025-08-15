import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import numpy as np
import time
from sklearn.metrics import accuracy_score

from models.hmp.schnet_hmp import HMP_SchNetModel
from models.hmp.dimenet_hmp import HMP_DimeNetPPModel
from models.hmp.spherenet_hmp import HMP_SphereNetModel
from models.hmp.egnn_hmp import HMP_EGNNModel
from models.hmp.gvpgnn_hmp import HMP_GVPGNNModel
from models.hmp.tfn_hmp import HMP_TFNModel
from models.hmp.mace_hmp import HMP_MACEModel

# --- Data Creation (from kchains.ipynb) ---
def create_kchains(k):
    assert k >= 2
    dataset = []
    # Graph 0
    pos1 = torch.FloatTensor([[-4, -3, 0]] + [[0, 5*i , 0] for i in range(k)] + [[4, 5*(k-1) + 3, 0]])
    pos1 -= pos1.mean(dim=0)
    data1 = Data(atoms=torch.zeros(k+2, dtype=torch.long), 
                 edge_index=to_undirected(torch.LongTensor([[i for i in range(k+1)], [i+1 for i in range(k+1)]])), 
                 pos=pos1, y=torch.LongTensor([0]))
    dataset.append(data1)
    # Graph 1
    pos2 = torch.FloatTensor([[4, -3, 0]] + [[0, 5*i , 0] for i in range(k)] + [[4, 5*(k-1) + 3, 0]])
    pos2 -= pos2.mean(dim=0)
    data2 = Data(atoms=torch.zeros(k+2, dtype=torch.long), 
                 edge_index=to_undirected(torch.LongTensor([[i for i in range(k+1)], [i+1 for i in range(k+1)]])), 
                 pos=pos2, y=torch.LongTensor([1]))
    dataset.append(data2)
    return dataset

# --- Modified Training/Evaluation (from train_utils.py) ---
def train_hmp(model, train_loader, optimizer, device, lambda_struct, lambda_rate):
    model.train()
    loss_all = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(batch)
        
        loss = F.cross_entropy(y_pred, batch.y)
        
        loss.backward()
        loss_all += loss.item() * batch.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)

def eval_hmp(model, loader, device):
    model.eval()
    y_pred = []
    y_true = []
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            y_pred.append(model(batch).detach().cpu())
            y_true.append(batch.y.detach().cpu())
    return accuracy_score(torch.cat(y_true, dim=0), np.argmax(torch.cat(y_pred, dim=0), axis=1)) * 100

def run_test_experiment(model, train_loader, val_loader, test_loader, n_epochs=10, device='cpu'):
    print(f"--- Testing {type(model).__name__} ---")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Using hyperparameters from paper
    lambda_struct = 0.01
    lambda_rate = 0.1

    for epoch in range(1, n_epochs + 1):
        loss = train_hmp(model, train_loader, optimizer, device, lambda_struct, lambda_rate)
        val_acc = eval_hmp(model, val_loader, device)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    test_acc = eval_hmp(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"--- Test for {type(model).__name__} complete ---")
    return test_acc

# --- Main Test Execution ---
if __name__ == '__main__':
    k = 4
    n_epochs = 2  # Reduced from 10 to avoid timeout
    
    # Create dataset and dataloaders
    dataset = create_kchains(k=k)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    
    # Test HMP-SchNet
    hmp_schnet = HMP_SchNetModel(num_layers=3, emb_dim=64, in_dim=1, out_dim=2)
    run_test_experiment(hmp_schnet, train_loader, val_loader, test_loader, n_epochs, device)
    
    # Test HMP-DimeNet
    hmp_dimenet = HMP_DimeNetPPModel(num_layers=3, emb_dim=64, in_dim=1, out_dim=2)
    run_test_experiment(hmp_dimenet, train_loader, val_loader, test_loader, n_epochs, device)

    # Test HMP-SphereNet
    hmp_spherenet = HMP_SphereNetModel(num_layers=3, emb_dim=64, in_dim=1, out_dim=2)
    run_test_experiment(hmp_spherenet, train_loader, val_loader, test_loader, n_epochs, device)

    # Test HMP-EGNN
    hmp_egnn = HMP_EGNNModel(num_layers=3, emb_dim=64, in_dim=1, out_dim=2)
    run_test_experiment(hmp_egnn, train_loader, val_loader, test_loader, n_epochs, device)

    # Test HMP-GVPGNN
    hmp_gvpgnn = HMP_GVPGNNModel(num_layers=3, s_dim=32, v_dim=16, in_dim=1, out_dim=2)
    run_test_experiment(hmp_gvpgnn, train_loader, val_loader, test_loader, n_epochs, device)

    # Test HMP-TFN
    hmp_tfn = HMP_TFNModel(num_layers=3, emb_dim=64, in_dim=1, out_dim=2)
    run_test_experiment(hmp_tfn, train_loader, val_loader, test_loader, n_epochs, device)
    
    # Test HMP-MACE
    # MACE requires e3nn which might not be installed. Let's wrap it in a try-except block.
    try:
        import e3nn
        hmp_mace = HMP_MACEModel(num_layers=3, emb_dim=32, in_dim=1, out_dim=2, max_ell=1, correlation=2)
        run_test_experiment(hmp_mace, train_loader, val_loader, test_loader, n_epochs, device)
    except ImportError:
        print("\nCould not import e3nn. Skipping HMP-MACE test.")
    
    print("\nAll tests finished.")
