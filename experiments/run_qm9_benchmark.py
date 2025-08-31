import argparse
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

# Dictionary mapping property names to their index in the QM9 dataset tensor.
QM9_PROPERTIES = {
    'mu': 0,
    'alpha': 1,
    'homo': 2,
    'lumo': 3,
    'gap': 4,
    'r2': 5,
    'zpve': 6,  # Note: Unit is eV in PyG, not meV as in the user request.
    'U0': 7,
    'U': 8,
    'H': 9,
    'G': 10,
    'Cv': 11,
}

def load_data():
    """Loads the QM9 dataset and creates data loaders for train, validation, and test sets."""
    print("Loading QM9 dataset...")
    dataset = QM9(root='dataset/')

    # The dataset has about 130k molecules. We use the requested split.
    n_train = 110000
    n_val = 10000

    # Create a random permutation of indices
    perm = torch.randperm(len(dataset))

    train_indices = perm[:n_train]
    val_indices = perm[n_train:n_train + n_val]
    test_indices = perm[n_train + n_val:]

    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]

    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test.")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

# --- Model Imports ---
# Backbone Models
from models.schnet import SchNetModel
from models.dimenet import DimeNetPPModel
from models.spherenet import SphereNetModel
from models.egnn import EGNNModel
from models.gvpgnn import GVPGNNModel
from models.tfn import TFNModel
from models.mace import MACEModel
# HMP-Enhanced Models
from models.hmp.schnet_hmp import HMP_SchNetModel
from models.hmp.dimenet_hmp import HMP_DimeNetPPModel
from models.hmp.spherenet_hmp import HMP_SphereNetModel
from models.hmp.egnn_hmp import HMP_EGNNModel
from models.hmp.gvpgnn_hmp import HMP_GVPGNNModel
from models.hmp.tfn_hmp import HMP_TFNModel
from models.hmp.mace_hmp import HMP_MACEModel


import torch.nn as nn
from tqdm import tqdm

# --- Training and Evaluation Functions ---
def train(model, loader, optimizer, device, property_indices):
    """Training loop for one epoch."""
    model.train()
    loss_fn = nn.L1Loss()
    total_loss = 0

    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()

        # Some models might not have all the required arguments
        try:
            out = model(data)
        except TypeError:
            try:
                out = model(data.atoms, data.pos, data.edge_index, data.batch)
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                continue

        target = data.y[:, property_indices]
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, property_indices, property_names):
    """Evaluation loop."""
    model.eval()
    total_mae = torch.zeros(len(property_indices), device=device)
    num_samples = 0

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", leave=False):
            data = data.to(device)

            try:
                out = model(data)
            except TypeError:
                try:
                    out = model(data.atoms, data.pos, data.edge_index, data.batch)
                except Exception as e:
                    print(f"Error during model forward pass: {e}")
                    continue

            target = data.y[:, property_indices]
            mae = torch.abs(out - target).sum(dim=0)
            total_mae += mae
            num_samples += data.num_graphs

    avg_mae = total_mae / num_samples
    return {name: mae.item() for name, mae in zip(property_names, avg_mae)}


# --- Model Configurations ---
def get_model_configs(l_value, num_properties):
    """Returns a dictionary of model configurations."""

    # Common hyperparameters
    # Using defaults from the paper or common practice
    # The user can modify these if needed
    common_params = {
        'schnet': {'hidden_channels': 128, 'num_filters': 128, 'num_gaussians': 50, 'cutoff': 10.0},
        'dimenet': {'hidden_channels': 128, 'out_emb_channels': 256, 'int_emb_channels': 64, 'num_bilinear': 8, 'num_spherical': 7, 'num_radial': 6, 'cutoff': 5.0, 'num_before_skip': 1, 'num_after_skip': 2, 'num_output_layers': 3},
        'spherenet': {'hidden_channels': 128, 'num_spherical': 7, 'num_radial': 6, 'cutoff': 5.0},
        'egnn': {'hidden_dim': 128, 'f_dim': 128},
        'gvpgnn': {'s_dim': 128, 'v_dim': 64},
        'tfn': {'hidden_dim': 128, 'degree': 2, 'f_dim': 128},
        'mace': {'hidden_dim': 128, 'correlation': 3, 'max_ell': 3},
    }

    # HMP specific parameters
    hmp_params = {
        'master_rate': 0.25,
        's_dim': 32,
        'master_selection_hidden_dim': 64,
        'lambda_attn': 0.1
    }

    model_configs = {
        # --- Backbone Models ---
        'SchNet': {
            'class': SchNetModel,
            'params': {**common_params['schnet'], 'num_layers': 6, 'out_dim': num_properties}
        },
        'DimeNet': {
            'class': DimeNetPPModel,
            'params': {**common_params['dimenet'], 'num_blocks': 4, 'out_dim': num_properties}
        },
        'SphereNet': {
            'class': SphereNetModel,
            'params': {**common_params['spherenet'], 'num_layers': 4, 'out_dim': num_properties}
        },
        'EGNN': {
            'class': EGNNModel,
            'params': {**common_params['egnn'], 'num_layers': 6, 'out_dim': num_properties}
        },
        'GVP-GNN': {
            'class': GVPGNNModel,
            'params': {**common_params['gvpgnn'], 'num_layers': 6, 'out_dim': num_properties}
        },
        'TFN': {
            'class': TFNModel,
            'params': {**common_params['tfn'], 'num_layers': 6, 'out_dim': num_properties}
        },
        'MACE': {
            'class': MACEModel,
            'params': {**common_params['mace'], 'num_layers': 2, 'out_dim': num_properties} # MACE is heavy, use fewer layers
        },

        # --- HMP-Enhanced Models ---
        'HMP-SchNet': {
            'class': HMP_SchNetModel,
            'params': {'num_layers': l_value, 'emb_dim': common_params['schnet']['hidden_channels'], 'out_dim': num_properties, **hmp_params}
        },
        'HMP-DimeNet': {
            'class': HMP_DimeNetPPModel,
            'params': {'num_layers': l_value, 'emb_dim': common_params['dimenet']['hidden_channels'], 'out_dim': num_properties, **hmp_params}
        },
        'HMP-SphereNet': {
            'class': HMP_SphereNetModel,
            'params': {'num_layers': l_value, 'emb_dim': common_params['spherenet']['hidden_channels'], 'out_dim': num_properties, **hmp_params}
        },
        'HMP-EGNN': {
            'class': HMP_EGNNModel,
            'params': {'num_layers': l_value, 'emb_dim': common_params['egnn']['hidden_dim'], 'out_dim': num_properties, **hmp_params}
        },
        'HMP-GVPGNN': {
            'class': HMP_GVPGNNModel,
            'params': {'num_layers': l_value, 's_dim': common_params['gvpgnn']['s_dim'], 'v_dim': common_params['gvpgnn']['v_dim'], 'out_dim': num_properties, **hmp_params}
        },
        'HMP-TFN': {
            'class': HMP_TFNModel,
            'params': {'num_layers': l_value, 'emb_dim': common_params['tfn']['hidden_dim'], 'out_dim': num_properties, **hmp_params}
        },
        'HMP-MACE': {
            'class': HMP_MACEModel,
            'params': {'num_layers': l_value, 'emb_dim': common_params['mace']['hidden_dim'], 'correlation': common_params['mace']['correlation'], 'max_ell': common_params['mace']['max_ell'], 'out_dim': num_properties, **hmp_params}
        },
    }
    return model_configs


import time
import csv

def main():
    parser = argparse.ArgumentParser(description='QM9 Benchmark Script')
    parser.add_argument('--L', type=int, default=4, help='Number of hierarchical layers for HMP models')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    args = parser.parse_args()

    print("--- QM9 Benchmark Script ---")
    print(f"Args: L={args.L}, Epochs={args.epochs}, LR={args.lr}, Device={args.device}")

    # Data Loading
    train_loader, val_loader, test_loader = load_data()

    # Model Configurations
    property_names = list(QM9_PROPERTIES.keys())
    property_indices = list(QM9_PROPERTIES.values())
    model_configs = get_model_configs(args.L, len(property_names))

    all_results = []

    for model_name, config in model_configs.items():
        print(f"\n--- Training Model: {model_name} ---")

        # MACE requires e3nn, skip if not installed
        if 'MACE' in model_name:
            try:
                import e3nn
            except ImportError:
                print("e3nn is not installed. Skipping MACE and HMP-MACE.")
                continue

        model = config['class'](**config['params']).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_loss = float('inf')

        start_time = time.time()
        for epoch in range(1, args.epochs + 1):
            # Gumbel-Softmax temperature annealing
            if hasattr(model, 'update_tau'):
                model.update_tau(epoch, args.epochs)

            train_loss = train(model, train_loader, optimizer, args.device, property_indices)
            val_metrics = evaluate(model, val_loader, args.device, property_indices, property_names)
            val_loss = sum(val_metrics.values()) / len(val_metrics) # Average MAE across properties

            scheduler.step()

            print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Val Loss (Avg MAE): {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the model state. In a real scenario, you'd save to disk.
                # For this script, we just keep the state in memory.
                best_model_state = model.state_dict()

        end_time = time.time()
        print(f"Training finished in {end_time - start_time:.2f} seconds.")

        # Load the best model and evaluate on the test set
        print("Evaluating on test set with the best model...")
        model.load_state_dict(best_model_state)
        test_metrics = evaluate(model, test_loader, args.device, property_indices, property_names)

        for prop_name, mae in test_metrics.items():
            all_results.append({
                'model_name': model_name,
                'property_name': prop_name,
                'test_mae': mae
            })
            print(f"  Test MAE for {prop_name}: {mae:.6f}")

    # --- Result Reporting ---
    print("\n--- Benchmark Results ---")

    # Print summary table
    header = f"{'Model':<20} | {'Property':<10} | {'Test MAE':<15}"
    print(header)
    print("-" * len(header))
    for result in all_results:
        print(f"{result['model_name']:<20} | {result['property_name']:<10} | {result['test_mae']:.6f}")

    # Save to CSV
    csv_file = 'qm9_results.csv'
    try:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model_name', 'property_name', 'test_mae'])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults successfully saved to {csv_file}")
    except IOError as e:
        print(f"\nError saving results to {csv_file}: {e}")


if __name__ == '__main__':
    main()
