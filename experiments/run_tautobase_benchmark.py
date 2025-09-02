import argparse
import torch
import os
import csv
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

# --- Utility Imports ---
from utils.tautobase_utils import MolecularDataset, TautobaseDataset

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

import time
import torch.nn as nn
from tqdm import tqdm

# --- Model Configurations ---
def get_model_configs(l_value, out_dim=1):
    """
    Returns a dictionary of model configurations.
    Ensures that HMP models share core hyperparameters with their backbone counterparts.
    """
    # --- Common Hyperparameters for Backbone Models ---
    # These are chosen to be reasonable defaults and can be easily modified here.
    common_params = {
        'schnet': {'hidden_channels': 128, 'num_filters': 128, 'num_interactions': 6, 'num_gaussians': 50, 'cutoff': 10.0},
        'dimenet': {'hidden_channels': 128, 'out_emb_channels': 256, 'int_emb_channels': 64, 'num_bilinear': 8, 'num_spherical': 7, 'num_radial': 6, 'cutoff': 5.0, 'num_before_skip': 1, 'num_after_skip': 2, 'num_output_layers': 3, 'num_blocks': 4},
        'spherenet': {'hidden_channels': 128, 'num_layers': 4, 'cutoff': 5.0},
        'egnn': {'hidden_dim': 128, 'num_layers': 6},
        'gvpgnn': {'s_dim': 128, 'v_dim': 64, 'num_layers': 6},
        'tfn': {'hidden_dim': 128, 'degree': 2, 'num_layers': 6},
        'mace': {'hidden_dim': 128, 'correlation': 3, 'max_ell': 3, 'num_layers': 2}, # MACE is heavy, fewer layers
    }

    # --- HMP-Net Specific Hyperparameters ---
    hmp_params = {
        'master_rate': 0.25,
        's_dim': 32, # Note: This is for the master node embedding, not the backbone's s_dim
        'master_selection_hidden_dim': 64,
        'lambda_attn': 0.1
    }

    model_configs = {
        # --- Backbone Models ---
        'SchNet':    {'class': SchNetModel,    'params': {**common_params['schnet'], 'out_dim': out_dim}},
        'DimeNet':   {'class': DimeNetPPModel, 'params': {**common_params['dimenet'], 'out_dim': out_dim}},
        'SphereNet': {'class': SphereNetModel, 'params': {**common_params['spherenet'], 'out_dim': out_dim}},
        'EGNN':      {'class': EGNNModel,      'params': {**common_params['egnn'], 'out_dim': out_dim}},
        'GVP-GNN':   {'class': GVPGNNModel,    'params': {**common_params['gvpgnn'], 'out_dim': out_dim}},
        'TFN':       {'class': TFNModel,       'params': {**common_params['tfn'], 'out_dim': out_dim}},
        'MACE':      {'class': MACEModel,      'params': {**common_params['mace'], 'out_dim': out_dim}},

        # --- HMP-Enhanced Models ---
        'HMP-SchNet': {
            'class': HMP_SchNetModel,
            'params': {'num_layers': l_value, 'emb_dim': common_params['schnet']['hidden_channels'], 'out_dim': out_dim, **hmp_params}
        },
        'HMP-DimeNet': {
            'class': HMP_DimeNetPPModel,
            'params': {'num_layers': l_value, 'emb_dim': common_params['dimenet']['hidden_channels'], 'out_dim': out_dim, **hmp_params}
        },
        'HMP-SphereNet': {
            'class': HMP_SphereNetModel,
            'params': {'num_layers': l_value, 'emb_dim': common_params['spherenet']['hidden_channels'], 'out_dim': out_dim, **hmp_params}
        },
        'HMP-EGNN': {
            'class': HMP_EGNNModel,
            'params': {'num_layers': l_value, 'emb_dim': common_params['egnn']['hidden_dim'], 'out_dim': out_dim, **hmp_params}
        },
        'HMP-GVPGNN': {
            'class': HMP_GVPGNNModel,
            # Note: HMP-GVPGNN's constructor takes s_dim and v_dim directly, not as backbone params.
            # We set them to be the same as the backbone's for a fair comparison.
            'params': {'num_layers': l_value, 's_dim': common_params['gvpgnn']['s_dim'], 'v_dim': common_params['gvpgnn']['v_dim'], 'out_dim': out_dim, **hmp_params}
        },
        'HMP-TFN': {
            'class': HMP_TFNModel,
            # Note: HMP-TFN takes max_ell and emb_dim. We map TFN's degree -> max_ell and hidden_dim -> emb_dim.
            'params': {'num_layers': l_value, 'emb_dim': common_params['tfn']['hidden_dim'], 'max_ell': common_params['tfn']['degree'], 'out_dim': out_dim, **hmp_params}
        },
        'HMP-MACE': {
            'class': HMP_MACEModel,
            'params': {'num_layers': l_value, 'emb_dim': common_params['mace']['hidden_dim'], 'correlation': common_params['mace']['correlation'], 'max_ell': common_params['mace']['max_ell'], 'out_dim': out_dim, **hmp_params}
        },
    }
    return model_configs

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    """
    Main function to train the model.
    Includes internal functions for one epoch of training and validation.
    """
    loss_fn = nn.L1Loss()
    best_val_loss = float('inf')
    best_model_state = None

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Gumbel-Softmax temperature annealing
        if hasattr(model, 'update_tau'):
            # The method signature is update_tau(epoch, n_epochs)
            model.update_tau(epoch, epochs)

        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch:03d} [Val]", leave=False):
                data = data.to(device)
                out = model(data)
                val_loss = loss_fn(out, data.y)
                total_val_loss += val_loss.item() * data.num_graphs
        avg_val_loss = total_val_loss / len(val_loader.dataset)

        scheduler.step()

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            print(f"  -> New best validation loss: {best_val_loss:.6f}")

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds. Best validation loss: {best_val_loss:.6f}")

    return best_model_state

def evaluate_model(model, test_loader, device):
    """
    Evaluation loop for the Tautobase test set.
    Calculates detailed per-pair metrics and overall summary statistics.
    """
    model.eval()
    detailed_results = []

    delta_E_errors = []
    esi_errors = []

    with torch.no_grad():
        for idx, (data_a, data_b) in enumerate(tqdm(test_loader, desc="Evaluating on Test Set")):
            data_a = data_a.to(device)
            data_b = data_b.to(device)

            # Predict energies
            Ea_nn = model(data_a).item()
            Eb_nn = model(data_b).item()

            # Get true energies
            Ea_DFT = data_a.y.item()
            Eb_DFT = data_b.y.item()

            # Calculate metrics
            Diff_MolA = Ea_nn - Ea_DFT
            Diff_MolB = Eb_nn - Eb_DFT
            Diff_ab = Eb_DFT - Ea_DFT  # This is ΔE_DFT
            Diff_NN = Eb_nn - Ea_nn    # This is ΔE_NN
            Diff_NN_ab = Diff_NN - Diff_ab

            # Store errors for summary calculation
            delta_E_errors.append(Diff_NN_ab)
            esi_errors.extend([Diff_MolA, Diff_MolB])

            # Store detailed results for CSV
            detailed_results.append({
                'idx': idx,
                'natoms': data_a.natoms.item(),
                'Ea_DFT': Ea_DFT,
                'Ea_nn': Ea_nn,
                'Diff_MolA': Diff_MolA,
                'Eb_DFT': Eb_DFT,
                'Eb_nn': Eb_nn,
                'Diff_MolB': Diff_MolB,
                'Diff_ab': Diff_ab,
                'Diff_NN': Diff_NN,
                'Diff_NN_ab': Diff_NN_ab,
            })

    # Calculate summary metrics
    delta_E_errors = np.array(delta_E_errors)
    esi_errors = np.array(esi_errors)

    summary_metrics = {
        'delta_E_MAE': np.mean(np.abs(delta_E_errors)),
        'delta_E_RMSE': np.sqrt(np.mean(delta_E_errors**2)),
        'ESI_MAE': np.mean(np.abs(esi_errors)),
        'ESI_RMSE': np.sqrt(np.mean(esi_errors**2)),
    }

    print("--- Test Set Evaluation Summary ---")
    print(f"  ΔE Tauto MAE: {summary_metrics['delta_E_MAE']:.6f}")
    print(f"  ΔE Tauto RMSE: {summary_metrics['delta_E_RMSE']:.6f}")
    print(f"  ESI MAE: {summary_metrics['ESI_MAE']:.6f}")
    print(f"  ESI RMSE: {summary_metrics['ESI_RMSE']:.6f}")

    return detailed_results, summary_metrics

def save_results(model_name, dataset_name, detailed_results, summary_metrics, overall_summary_list):
    """Saves detailed and summary results to CSV files."""

    # Create results directory if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # 1. Save detailed per-pair results for the current model
    if detailed_results:
        detailed_csv_path = os.path.join(results_dir, f"Tautobase_{dataset_name}_{model_name}_detailed.csv")
        print(f"Saving detailed results to {detailed_csv_path}...")
        with open(detailed_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=detailed_results[0].keys())
            writer.writeheader()
            writer.writerows(detailed_results)

    # 2. Update the overall summary list and save it
    summary_data = {
        'model': model_name,
        'dataset': dataset_name,
        **summary_metrics
    }
    overall_summary_list.append(summary_data)

    summary_csv_path = os.path.join(results_dir, "Tautobase_benchmark_summary.csv")
    print(f"Updating overall summary at {summary_csv_path}...")
    if overall_summary_list:
         with open(summary_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=overall_summary_list[0].keys())
            writer.writeheader()
            writer.writerows(overall_summary_list)

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Tautobase Benchmark Script')
    parser.add_argument('--L', type=int, default=4, help='Number of hierarchical layers for HMP models')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--dataset_root', type=str, default='../dataset_sample', help='Root directory of the datasets')
    args = parser.parse_args()

    print("--- Tautobase Benchmark Script ---")
    print(f"Args: L={args.L}, Epochs={args.epochs}, LR={args.lr}, Device={args.device}")

    # Define the experimental setups
    dataset_configs = {
        'QM9': {'train_dir': 'qm9', 'test_dir': 'QTautobase/QM9'},
        'PC9': {'train_dir': 'PC9/XYZ', 'test_dir': 'QTautobase/PC9'},
        'ANI-1E': {'train_dir': 'ANI-1E/xyz', 'test_dir': 'QTautobase/ANI1E'}
    }

    overall_summary = []

    for name, paths in dataset_configs.items():
        print(f"\n===== Running Benchmark for {name} Dataset =====")

        # 1. Load Data
        print(f"Loading {name} training data...")
        full_train_dataset = MolecularDataset(root=os.path.join(args.dataset_root, paths['train_dir']))

        # Split data
        train_indices, val_indices = train_test_split(
            list(range(len(full_train_dataset))), test_size=0.1, random_state=42
        )
        train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        print(f"Loading Tautobase/{name} test data...")
        test_dataset = TautobaseDataset(root=os.path.join(args.dataset_root, paths['test_dir']))
        # Use batch_size=1 for pair-wise evaluation
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test pairs.")

        # 2. Get Model Configurations
        model_configs = get_model_configs(args.L)

        for model_name, config in model_configs.items():
            print(f"\n--- Training Model: {model_name} on {name} ---")

            # MACE and TFN require e3nn, skip if not installed
            if 'MACE' in model_name or 'TFN' in model_name:
                try:
                    import e3nn
                except ImportError:
                    print("e3nn is not installed. Skipping MACE/TFN and their HMP versions.")
                    continue

            # 3. Initialize Model and Optimizer
            model = config['class'](**config['params']).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            # 4. Train Model
            best_model_state = train_model(model, train_loader, val_loader, optimizer, scheduler, args.device, args.epochs)
            model.load_state_dict(best_model_state)

            # 5. Evaluate Model
            detailed_results, summary_metrics = evaluate_model(model, test_loader, args.device)

            # 6. Save Results
            # The save_results function now handles updating the overall_summary list
            save_results(model_name, name, detailed_results, summary_metrics, overall_summary)

    # --- Final Reporting ---
    print("\n--- Overall Benchmark Summary ---")
    if overall_summary:
        # Sort results for consistent display
        overall_summary.sort(key=lambda x: (x['dataset'], x['model']))

        # Display header
        header = f"{'Dataset':<10} | {'Model':<15} | {'dE MAE':<12} | {'dE RMSE':<12} | {'ESI MAE':<12} | {'ESI RMSE':<12}"
        print(header)
        print("-" * len(header))

        # Display results
        for summary in overall_summary:
            print(
                f"{summary['dataset']:<10} | {summary['model']:<15} | "
                f"{summary['delta_E_MAE']:.6f} | {summary['delta_E_RMSE']:.6f} | "
                f"{summary['ESI_MAE']:.6f} | {summary['ESI_RMSE']:.6f}"
            )

        print(f"\nSummary results saved to results/Tautobase_benchmark_summary.csv")
    else:
        print("No results to summarize.")

if __name__ == '__main__':
    main()
