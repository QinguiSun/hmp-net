import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import numpy as np
import time
from sklearn.metrics import accuracy_score

from models.hmp.egnn_hmp import HMP_EGNNModel

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
        
        out_dict = model(batch)
        y_pred = out_dict['pred']
        
        task_loss = F.cross_entropy(y_pred, batch.y)
        struct_loss = out_dict['l_struct']
        rate_loss = out_dict['l_rate']
        
        loss = task_loss + lambda_struct * struct_loss + lambda_rate * rate_loss
        
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
            y_pred.append(model(batch)['pred'].detach().cpu())
            y_true.append(batch.y.detach().cpu())
    return accuracy_score(torch.cat(y_true, dim=0), np.argmax(torch.cat(y_pred, dim=0), axis=1)) * 100

def run_test_experiment(model, train_loader, val_loader, test_loader, n_epochs=10, device='cpu'):
    print(f"--- Testing {type(model).__name__} ---")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
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
    n_epochs = 2
    
    dataset = create_kchains(k=k)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    hmp_egnn = HMP_EGNNModel(num_layers=3, emb_dim=64, in_dim=1, out_dim=2, s_dim=16)
    run_test_experiment(hmp_egnn, train_loader, val_loader, test_loader, n_epochs, device)
    
    print("\nEGNN test finished.")
