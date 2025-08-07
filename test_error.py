import torch
from experiments.kchains import create_kchains
from models.hmp.egnn_hmp import HMP_EGNNModel
from torch_geometric.loader import DataLoader

k = 4
dataset = create_kchains(k=k)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_layers = 6
hmp_egnn_model = HMP_EGNNModel(
    num_layers=num_layers,
    in_dim=1,
    out_dim=2,
    master_rate=0.25
).to(device)

for batch in dataloader:
    batch = batch.to(device)
    try:
        y_pred = hmp_egnn_model(batch)
        print("Success!")
    except Exception as e:
        print(e)
