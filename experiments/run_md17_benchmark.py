"""
Benchmark script for MD17 dataset.
This script benchmarks various backbone models and their HMP-Net enhanced versions
on the MD17 dataset for energy and force prediction.
"""
import argparse
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

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

# --- Constants ---
MOLECULES = [
    'aspirin', 'azobenzene', 'benzene', 'ethanol', 'malonaldehyde',
    'naphthalene', 'salicylic', 'toluene', 'uracil'
]
# A mapping from atomic number to symbol is useful for debugging.
ATOMIC_SYMBOLS = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}


# --- Dataset Class ---

class MD17Dataset(InMemoryDataset):
    """
    Custom PyG Dataset for MD17. This class handles loading and processing data
    from the specified directory structure where each molecule's data is stored
    in its own folder with separate files for energies, forces, coordinates, etc.
    """
    def __init__(self, root, molecule_name, transform=None, pre_transform=None, pre_filter=None):
        self.molecule_name = molecule_name
        super().__init__(os.path.join(root, molecule_name), transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['nuclear_charges.txt', 'energies.txt', 'forces.txt', 'coords.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # The data is expected to be manually placed in the `dataset/md17` directory.
        # No download is necessary.
        pass

    def process(self):
        print(f"Processing data for molecule: {self.molecule_name}...")

        # Read raw data files
        z_path = os.path.join(self.raw_dir, 'nuclear_charges.txt')
        energy_path = os.path.join(self.raw_dir, 'energies.txt')
        forces_path = os.path.join(self.raw_dir, 'forces.txt')
        coords_path = os.path.join(self.raw_dir, 'coords.txt')

        # Use numpy to load text files, which is faster for large numerical data
        atomic_numbers = torch.tensor(np.loadtxt(z_path), dtype=torch.long)
        energies = torch.tensor(np.loadtxt(energy_path), dtype=torch.float)
        forces = torch.tensor(np.loadtxt(forces_path), dtype=torch.float)
        coords = torch.tensor(np.loadtxt(coords_path), dtype=torch.float)

        num_atoms = atomic_numbers.size(0)
        num_snapshots = energies.size(0)

        # Reshape coordinates and forces
        coords = coords.view(num_snapshots, num_atoms, 3)
        forces = forces.view(num_snapshots, num_atoms, 3)

        data_list = []
        for i in tqdm(range(num_snapshots), desc=f"Creating Data objects for {self.molecule_name}"):
            data = Data(
                z=atomic_numbers,
                pos=coords[i],
                energy=energies[i].unsqueeze(0), # Ensure energy is [1]
                force=forces[i]
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def load_molecule_data(molecule_name, batch_size=32):
    """
    Loads, splits, and prepares data loaders for a specific molecule.

    Args:
        molecule_name (str): The name of the molecule to load.
        batch_size (int): The batch size for the data loaders.

    Returns:
        A tuple of (train_loader, val_loader, test_loader), or (None, None, None)
        if the data cannot be loaded.
    """
    dataset_path = os.path.join('dataset', 'md17')

    # Check if the raw data directory exists before proceeding
    raw_dir_path = os.path.join(dataset_path, molecule_name, 'raw')
    if not os.path.exists(raw_dir_path):
        print(f"Warning: Raw data for molecule '{molecule_name}' not found at {raw_dir_path}. Skipping.")
        return None, None, None

    dataset = MD17Dataset(root=dataset_path, molecule_name=molecule_name)

    # Define split sizes
    n_train = 50000
    n_val = 10000

    if len(dataset) < n_train + n_val:
        print(f"Warning: Dataset for {molecule_name} has only {len(dataset)} samples, "
              f"which is not enough for the required {n_train}+{n_val} split. Skipping.")
        return None, None, None

    # Create a random permutation for splitting
    perm = torch.randperm(len(dataset))

    train_indices = perm[:n_train]
    val_indices = perm[n_train:n_train + n_val]
    test_indices = perm[n_train + n_val:]

    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]

    print(f"Dataset for {molecule_name}: "
          f"{len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_model_configs():
    """
    Returns a dictionary containing the configurations for all models to be benchmarked.
    This ensures that backbone models and their HMP-enhanced versions share key
    hyperparameters for a fair comparison.
    """
    # As requested, L (num_layers for HMP models) is a hyperparameter set here.
    # We use L=4 as a default, which can be modified by the user later.
    L_HIERARCHICAL_LAYERS = 4

    # Per-family hyperparameters to ensure consistency.
    # User can modify these defaults.
    common_params = {
        'schnet': {'hidden_channels': 128, 'num_filters': 128, 'num_gaussians': 50, 'cutoff': 10.0},
        'dimenet': {'hidden_channels': 128, 'out_emb_channels': 256, 'int_emb_channels': 64, 'num_bilinear': 8, 'num_spherical': 7, 'num_radial': 6, 'cutoff': 5.0, 'num_before_skip': 1, 'num_after_skip': 2, 'num_output_layers': 3},
        'spherenet': {'hidden_channels': 128, 'cutoff': 5.0},
        'egnn': {'hidden_dim': 128, 'f_dim': 128},
        'gvpgnn': {'s_dim': 128, 'v_dim': 64},
        'tfn': {'hidden_dim': 128, 'degree': 2, 'f_dim': 128},
        'mace': {'hidden_dim': 128, 'correlation': 3, 'max_ell': 3},
    }

    # HMP-specific parameters, as specified in the request.
    hmp_params = {
        'master_rate': 0.25,
        's_dim': 32, # Dimensionality of the master node signal, can be tuned.
        'master_selection_hidden_dim': 64,
        'lambda_attn': 0.1 # Weight for the attention loss component, can be tuned.
    }

    model_configs = {
        # --- Backbone Models ---
        # The number of layers for backbone models are set to common default values.
        # The user can modify these as needed.
        'SchNet': {
            'class': SchNetModel,
            'params': {**common_params['schnet'], 'num_layers': 6, 'out_dim': 1}
        },
        'DimeNet': {
            'class': DimeNetPPModel,
            'params': {**common_params['dimenet'], 'num_blocks': 4, 'out_dim': 1}
        },
        'SphereNet': {
            'class': SphereNetModel,
            'params': {**common_params['spherenet'], 'num_layers': 4, 'out_dim': 1}
        },
        'EGNN': {
            'class': EGNNModel,
            'params': {**common_params['egnn'], 'num_layers': 6, 'out_dim': 1}
        },
        'GVP-GNN': {
            'class': GVPGNNModel,
            'params': {**common_params['gvpgnn'], 'num_layers': 6, 'out_dim': 1}
        },
        'TFN': {
            'class': TFNModel,
            'params': {**common_params['tfn'], 'num_layers': 6, 'out_dim': 1}
        },
        'MACE': {
            'class': MACEModel,
            'params': {**common_params['mace'], 'num_layers': 2, 'out_dim': 1} # MACE is heavy, fewer layers
        },

        # --- HMP-Enhanced Models ---
        'HMP-SchNet': {
            'class': HMP_SchNetModel,
            'params': {'num_layers': L_HIERARCHICAL_LAYERS, 'emb_dim': common_params['schnet']['hidden_channels'], 'out_dim': 1, **hmp_params}
        },
        'HMP-DimeNet': {
            'class': HMP_DimeNetPPModel,
            'params': {'num_layers': L_HIERARCHICAL_LAYERS, 'emb_dim': common_params['dimenet']['hidden_channels'], 'out_dim': 1, **hmp_params}
        },
        'HMP-SphereNet': {
            'class': HMP_SphereNetModel,
            'params': {'num_layers': L_HIERARCHICAL_LAYERS, 'emb_dim': common_params['spherenet']['hidden_channels'], 'out_dim': 1, **hmp_params}
        },
        'HMP-EGNN': {
            'class': HMP_EGNNModel,
            'params': {'num_layers': L_HIERARCHICAL_LAYERS, 'emb_dim': common_params['egnn']['hidden_dim'], 'out_dim': 1, **hmp_params}
        },
        'HMP-GVPGNN': {
            'class': HMP_GVPGNNModel,
            'params': {'num_layers': L_HIERARCHICAL_LAYERS, 's_dim': common_params['gvpgnn']['s_dim'], 'v_dim': common_params['gvpgnn']['v_dim'], 'out_dim': 1, **hmp_params}
        },
        'HMP-TFN': {
            'class': HMP_TFNModel,
            'params': {'num_layers': L_HIERARCHICAL_LAYERS, 'emb_dim': common_params['tfn']['hidden_dim'], 'out_dim': 1, **hmp_params}
        },
        'HMP-MACE': {
            'class': HMP_MACEModel,
            'params': {'num_layers': L_HIERARCHICAL_LAYERS, 'emb_dim': common_params['mace']['hidden_dim'], 'correlation': common_params['mace']['correlation'], 'max_ell': common_params['mace']['max_ell'], 'out_dim': 1, **hmp_params}
        },
    }
    return model_configs

def train(model, loader, optimizer, device, force_weight):
    """
    Training loop for one epoch that considers both energy and force predictions.
    Forces are derived from the energy prediction via auto-differentiation.
    """
    model.train()
    loss_fn = nn.L1Loss()
    total_energy_loss = 0
    total_force_loss = 0

    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        data.pos.requires_grad = True  # Enable gradient tracking for positions

        optimizer.zero_grad()

        # Forward pass to get energy prediction
        predicted_energy = model(data)
        if predicted_energy.shape[-1] == 1:
            predicted_energy = predicted_energy.squeeze(-1)

        # Calculate forces as the negative gradient of energy w.r.t. positions
        # The sum is taken because autograd.grad expects a scalar output.
        grad_outputs = torch.ones_like(predicted_energy)
        predicted_forces = -torch.autograd.grad(
            predicted_energy,
            data.pos,
            grad_outputs=grad_outputs,
            create_graph=True  # Create graph for backprop through forces
        )[0]

        # Calculate loss
        loss_e = loss_fn(predicted_energy, data.energy)
        loss_f = loss_fn(predicted_forces, data.force)

        loss = loss_e + force_weight * loss_f

        loss.backward()
        optimizer.step()

        total_energy_loss += loss_e.item() * data.num_graphs
        total_force_loss += loss_f.item() * data.num_graphs

    avg_energy_loss = total_energy_loss / len(loader.dataset)
    avg_force_loss = total_force_loss / len(loader.dataset)
    return avg_energy_loss, avg_force_loss

def evaluate(model, loader, device):
    """
    Evaluation loop that calculates MAE for both energy and forces.
    """
    model.eval()
    total_energy_mae = 0
    total_force_mae = 0

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", leave=False):
            data = data.to(device)
            data.pos.requires_grad = True # Still need grad for force calculation

            # Temporarily enable gradient calculation for force prediction
            with torch.enable_grad():
                predicted_energy = model(data)
                if predicted_energy.shape[-1] == 1:
                    predicted_energy = predicted_energy.squeeze(-1)

                predicted_forces = -torch.autograd.grad(
                    predicted_energy, data.pos, grad_outputs=torch.ones_like(predicted_energy)
                )[0]

            # MAE calculation
            total_energy_mae += nn.L1Loss(reduction='sum')(predicted_energy, data.energy).item()
            total_force_mae += nn.L1Loss(reduction='sum')(predicted_forces, data.force).item()

    avg_energy_mae = total_energy_mae / len(loader.dataset)
    avg_force_mae = total_force_mae / len(loader.dataset)

    return {'energy_mae': avg_energy_mae, 'force_mae': avg_force_mae}


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='MD17 Benchmark Script')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--force_weight', type=float, default=10.0, help='Weight for the force component of the loss. A value around 10-100 is often a good starting point.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation.')
    args = parser.parse_args()

    print("--- MD17 Benchmark Script ---")
    print(f"Args: Epochs={args.epochs}, LR={args.lr}, Device={args.device}, Force Weight={args.force_weight}, Batch Size={args.batch_size}")

    # Get model configurations
    model_configs = get_model_configs()

    all_results = []

    # Loop over all molecules
    for molecule in MOLECULES:
        print(f"\n{'='*50}\n--- Benchmarking on Molecule: {molecule.upper()} ---\n{'='*50}")

        # Load data for the current molecule
        train_loader, val_loader, test_loader = load_molecule_data(molecule, batch_size=args.batch_size)
        if train_loader is None:
            continue

        # Loop over all models
        for model_name, config in model_configs.items():
            print(f"\n--- Training Model: {model_name} on {molecule} ---")

            # Skip MACE if e3nn is not installed
            if 'MACE' in model_name:
                try:
                    import e3nn
                except ImportError:
                    print("e3nn not installed. Skipping MACE and HMP-MACE.")
                    continue

            # Instantiate model and optimizer
            model = config['class'](**config['params']).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            best_val_loss = float('inf')
            best_model_state = None

            start_time = time.time()
            for epoch in range(1, args.epochs + 1):
                # Gumbel-Softmax temperature annealing for HMP models
                if hasattr(model, 'update_tau'):
                    # Anneal tau from 1.0 to 0.1 over the first 50% of epochs
                    model.update_tau(epoch, args.epochs)

                train_e_loss, train_f_loss = train(model, train_loader, optimizer, args.device, args.force_weight)
                val_metrics = evaluate(model, val_loader, args.device)

                # Use a combined validation loss for model selection
                val_loss = val_metrics['energy_mae'] + args.force_weight * val_metrics['force_mae']

                scheduler.step()

                if epoch % 10 == 0:
                    print(f"Epoch {epoch:03d}: Train E Loss: {train_e_loss:.4f}, Train F Loss: {train_f_loss:.4f} | "
                          f"Val E MAE: {val_metrics['energy_mae']:.4f}, Val F MAE: {val_metrics['force_mae']:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()

            end_time = time.time()
            print(f"Training finished in {end_time - start_time:.2f} seconds.")

            # --- Evaluation on test set ---
            print("Evaluating on test set with the best model...")
            model.load_state_dict(best_model_state)
            test_metrics = evaluate(model, test_loader, args.device)

            print(f"  Test Energy MAE: {test_metrics['energy_mae']:.6f}")
            print(f"  Test Force MAE: {test_metrics['force_mae']:.6f}")

            # --- Store results ---
            all_results.append({
                'model': model_name, 'molecule': molecule,
                'target': 'energy', 'mae': test_metrics['energy_mae']
            })
            all_results.append({
                'model': model_name, 'molecule': molecule,
                'target': 'force', 'mae': test_metrics['force_mae']
            })

    # --- Result Reporting ---
    print(f"\n\n{'='*50}\n--- Final Benchmark Results ---\n{'='*50}")

    # Print summary table to console
    header = f"{'Model':<20} | {'Molecule':<15} | {'Target':<10} | {'Test MAE':<15}"
    print(header)
    print("-" * len(header))
    for result in all_results:
        print(f"{result['model']:<20} | {result['molecule']:<15} | {result['target']:<10} | {result['mae']:.6f}")

    # Save to CSV
    csv_file = 'md17_results.csv'
    print(f"\nSaving results to {csv_file}...")
    try:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model', 'molecule', 'target', 'mae'])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Results successfully saved to {csv_file}")
    except IOError as e:
        print(f"Error: Could not save results to {csv_file}. Reason: {e}")

if __name__ == '__main__':
    main()
