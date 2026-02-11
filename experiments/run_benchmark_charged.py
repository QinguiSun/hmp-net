# file: hmp-net-clean/experiments/run_benchmark_charged.py
import argparse
import torch
import os
import csv
import numpy as np
import time
import json
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from datasets.dataset_input import InputDataDataset
from torch_geometric.utils import scatter 
from torch_geometric.transforms import RadiusGraph
from sklearn.linear_model import LinearRegression 
import torch.nn as nn
import torch.nn.functional as F

# --- Model Imports ---
# Backbone Models
from models.schnet_vn import SchNetVNModel
from models.dimenet import DimeNetPPModel
from models.spherenet import SphereNetModel
from models.egnn import EGNNModel
# HMP-Enhanced Models
from models.hmp.schnet_hmp_mlp import HMP_SchNetModel

HARTREE_TO_MEV = 27.211386 * 1000.0  # meV per Hartree
BOHR_TO_ANGSTROM = 0.529177210903
FORCE_CONV = HARTREE_TO_MEV / BOHR_TO_ANGSTROM

def _n_atoms_per_graph(batch):
    batch_indices = batch.batch
    num_graphs = int(batch_indices.max().item()) + 1 if batch_indices.numel() > 0 else 1
    return torch.bincount(batch_indices, minlength=num_graphs)


def split_dataset(dataset, seed=0, frac_train=0.8, frac_val=0.1):
    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(n * frac_train)
    n_val = int(n * frac_val)
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]
    train_set = [dataset[i] for i in idx_train]
    val_set = [dataset[i] for i in idx_val]
    test_set = [dataset[i] for i in idx_test]
    return train_set, val_set, test_set

def compute_forces(E_graph, pos, create_graph: bool):
    """
    计算能量对坐标的负梯度 (-dE/dx)
    """
    if not pos.requires_grad:
        raise RuntimeError("pos must require grad BEFORE forward() to backprop force loss.")
    
    # 这里的 E_graph 是 (Batch_size, 1) 或者 (Batch_size,)
    E_sum = E_graph.sum()
    
    grad_pos = torch.autograd.grad(
        outputs=E_sum,
        inputs=pos,
        create_graph=create_graph, # 训练时需要为 True 以便对 Force 进行二阶导数优化
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    return -grad_pos # Force = - Gradient


# --- 统计工具函数 ---
def compute_dataset_stats(dataset, num_atom_classes):
    """
    计算数据集的统计信息：AtomRefs, Mean, Std
    用于将数据归一化到 (0, 1) 分布，便于神经网络训练
    """
    print("Computing dataset statistics (AtomRefs, Mean, Std)...")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    atom_counts_list = []
    energies_list = []
    
    for data in tqdm(loader, desc="Collecting stats"):
        # data.atoms 是经过 map 后的索引 (0 ~ num_classes-1)
        node_one_hot = torch.nn.functional.one_hot(data.atoms, num_classes=num_atom_classes).float()
        graph_counts = scatter(node_one_hot, data.batch, dim=0, reduce='sum')
        
        atom_counts_list.append(graph_counts.numpy())
        energies_list.append(data.y.numpy())
        
    atom_counts = np.concatenate(atom_counts_list, axis=0)
    energies = np.concatenate(energies_list, axis=0)
    
    # 1. 线性回归求解单原子能量参考值 (AtomRefs)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(atom_counts, energies)
    atom_refs = torch.tensor(reg.coef_, dtype=torch.float32).view(-1) # (num_classes,)
    
    print("Estimated AtomRefs:", atom_refs)

    # 2. 计算相互作用能 (去除去单原子能量后的残差)
    predicted_atomic_energy = reg.predict(atom_counts)
    interaction_energies = energies - predicted_atomic_energy
    
    # 3. 计算均值和标准差用于归一化
    mean = torch.tensor(np.mean(interaction_energies), dtype=torch.float32)
    std = torch.tensor(np.std(interaction_energies), dtype=torch.float32)
    
    print(f"Interaction Energy Stats -> Mean: {mean.item():.6f}, Std: {std.item():.6f}")
    
    return atom_refs, mean, std

# --- Model Configurations ---
def get_model_configs(l_value, num_atom_classes, out_dim=1):
    common_params = {
        'schnet': {'hidden_channels': 512, 'num_filters': 512, 'num_gaussians': 300, 'cutoff': 10.0, 'num_embeddings': num_atom_classes},
        'dimenet': {'hidden_channels': 128, 'out_emb_channels': 256, 'num_embeddings':num_atom_classes, 'int_emb_size': 64, 'basis_emb_size': 8, 'num_spherical': 7, 'num_radial': 6, 'cutoff': 5.0, 'num_before_skip': 1, 'num_after_skip': 2, 'num_output_layers': 3, 'num_layers': 4},
        'spherenet': {'hidden_channels': 128, 'num_layers': 4, 'cutoff': 5.0, 'num_embeddings': num_atom_classes},
        'egnn': {'emb_dim': 128, 'num_gaussians': 180, 'cutoff': 30.0, 'num_embeddings': num_atom_classes},
        'gvpgnn': {'s_dim': 128, 'v_dim': 64, 'num_layers': 6, 'num_embeddings':num_atom_classes},
        'tfn': {'emb_dim': 128, 'max_ell': 2, 'num_layers': 6, 'num_embeddings':num_atom_classes},
        'mace': {'emb_dim': 64, 'correlation': 3, 'max_ell': 3, 'num_layers': 2, 'num_embeddings':num_atom_classes},
    }
                                                                                        
    hmp_params = {
        'master_rate': 0.25,
        's_dim': 32,
        'master_selection_hidden_dim': 64,
        'lambda_attn': 0.01
    }

    model_configs = {
        # --- Backbone Models ---
        'SchNet':    {'class': SchNetVNModel,    'params': {**common_params['schnet'], 'num_layers': l_value, 'out_dim': out_dim}},

        # --- HMP-Net Models ---
        #'HMP-SchNet': {'class': HMP_SchNetModel, 'params': {**common_params['schnet'], 'num_layers': l_value, 'out_dim': out_dim, **hmp_params, }},

    }
    return model_configs

# [FIX] 引入 stats 参数，以便在评估时进行反归一化
@torch.no_grad()
def eval_rmse_epoch(model, loader, device, stats, wE: float = 1.0, wF: float = 9.0, wq: float = 1.0):
    """
    评估函数：
    1. 计算归一化空间的输出
    2. 反归一化到 Hartree 单位
    3. 转换为 meV 和 meV/A 计算 RMSE
    """
    atom_refs, mean, std = [x.to(device) for x in stats]
    model.eval()
    
    sum_sq_E = 0.0
    n_graphs = 0
    sum_sq_F = 0.0
    n_force_components = 0

    # [FIXED] 正确初始化 accumulators
    sum_sq_q = 0.0
    n_q_total = 0
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        
        # 为了计算力，即使在评估模式也需要 grad，但不需要优化器的 step
        with torch.enable_grad():
            batch.pos = batch.pos.detach().requires_grad_(True)

            # [FIXED] 传递 q_tot 以便模型正确约束电荷总和
            q_tot_input = batch.q_tot.view(-1) if hasattr(batch, 'q_tot') else None
            out = model(batch, q_tot=q_tot_input)

            if isinstance(out, tuple):
                 Energy, q = out
            else:
                 Energy = out
                 q = None # Handle models without charge output

            # 计算 Normalized Force: F_norm = - d(E_norm)/dx
            F_pred_norm = compute_forces(Energy, batch.pos, create_graph=False)

        # --- 反归一化 (Denormalization) ---
        # Energy: E_pred = E_norm * std + mean + sum(atom_refs)
        batch_atom_refs = scatter(atom_refs[batch.atoms], batch.batch, dim=0, reduce='sum')
        E_pred_Ha = Energy.view(-1) * std + mean + batch_atom_refs
        E_true_Ha = batch.y.view(-1)

        # Force: F_pred = F_norm * std (因为 E = E_norm * std + const, 导数只保留系数 std)
        F_pred_HaB = F_pred_norm * std
        F_true_HaB = batch.forces

        q_pred = q
        q_true = batch.q_atom_ref if hasattr(batch, 'q_atom_ref') else None

        # --- 单位转换 (Hartree -> meV) ---
        n_atoms = _n_atoms_per_graph(batch).to(device)
        
        # Energy (meV/atom)
        E_pred_mev = E_pred_Ha * HARTREE_TO_MEV
        E_true_mev = E_true_Ha * HARTREE_TO_MEV
        
        diff_E = (E_pred_mev - E_true_mev) / n_atoms # per atom error
        sum_sq_E += (diff_E ** 2).sum().item()
        n_graphs += diff_E.numel()
        
        # Force (meV/A)
        F_pred_mevA = F_pred_HaB * FORCE_CONV
        F_true_mevA = F_true_HaB * FORCE_CONV
        
        diff_F = F_pred_mevA - F_true_mevA
        sum_sq_F += (diff_F ** 2).sum().item()
        n_force_components += diff_F.numel()

        # [FIXED] 正确累积 Charges 误差
        if q_pred is not None and q_true is not None:
            diff_q = q_pred - q_true
            sum_sq_q += (diff_q ** 2).sum().item()
            n_q_total += diff_q.numel()

    if n_graphs == 0 or n_force_components == 0:
        return {"rmse_E": float("inf"), "rmse_F": float("inf"), "val_total": float("inf")}
        
    rmse_E = (sum_sq_E / n_graphs) ** 0.5
    rmse_F = (sum_sq_F / n_force_components) ** 0.5

    # [FIXED] 计算 RMSE，确保为标量 float
    rmse_q = (sum_sq_q / n_q_total) ** 0.5 if n_q_total > 0 else None
    

    # 使用加权和作为早停指标
    val_total = rmse_E + 5.0 * rmse_F + 2000.0 * rmse_q 

    return {"rmse_E": rmse_E, "rmse_F": rmse_F, "rmse_q": rmse_q, "val_total": val_total}
        

# --- Training Function ---
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs, stats, wE = 1.0, wF = 10.0, wq=100.0):
    # [FIX] 恢复统计数据用于归一化
    atom_refs, mean, std = [x.to(device) for x in stats]
    
    best_val_metric = float('inf') # Changed variable name to generic metric
    best_model_state = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        if hasattr(model, 'update_tau'):
            model.update_tau(epoch, epochs)

        # --- Train ---
        model.train()
        
        # 记录用于显示的 Loss (可以是 Normalized 也可以是 Physical，这里记录原始 Normalized Loss 更直观反应训练状态)
        total_norm_loss = 0.0 
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False):
            batch = batch.to(device)
            batch.pos.requires_grad_(True)
            optimizer.zero_grad()

            # 1. Forward (Model outputs Normalized Energy)
            # [FIXED] 传递 q_tot 以便模型在训练时使用正确的总电荷约束
            q_tot_input = batch.q_tot.view(-1) if hasattr(batch, 'q_tot') else None
            Energy, q = model(batch, q_tot=q_tot_input)
            
            # 2. Compute Force (Normalized: -d(E_norm)/dx)
            F_pred_norm = compute_forces(Energy, batch.pos, create_graph=True)

            # 3. Construct Targets (Normalized)
            # Energy Target: (y - refs - mean) / std
            batch_atom_refs = scatter(atom_refs[batch.atoms], batch.batch, dim=0, reduce='sum')
            y_true = batch.y.view(-1)
            target_E_norm = (y_true - batch_atom_refs - mean) / std
            q_atom_ref = batch.q_atom_ref if hasattr(batch, 'q_atom_ref') else None       
            
            # Force Target: F_true / std
            # 因为 F = -grad(E), E = E_norm * std + ... => F = F_norm * std => F_norm = F / std
            target_F_norm = batch.forces / std

            # 4. Compute Loss in Normalized Space (Stable Gradients)
            # 使用 MSE Loss
            loss_E = F.mse_loss(Energy.view(-1), target_E_norm)
            loss_F = F.mse_loss(F_pred_norm, target_F_norm)
            loss_q = F.mse_loss(q, q_atom_ref) if q_atom_ref is not None else 0.0
            
            # 组合 Loss
            loss = wE * loss_E + wF * loss_F + wq * loss_q
            
            loss.backward()
            optimizer.step()
            
            total_norm_loss += loss.item() * batch.num_graphs
        
        avg_train_loss = total_norm_loss / len(train_loader.dataset)
        
        # --- Validation (Converted Units) ---
        # 使用 eval_rmse_epoch 计算物理单位 (meV) 下的误差
        val_metrics = eval_rmse_epoch(model, val_loader, device, stats, wE=wE, wF=wF, wq=wq)
        
        #scheduler.step()

        # Scheduler Step (ReduceLROnPlateau needs a metric)
        current_lr = optimizer.param_groups[0]['lr']
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics['val_total'])
        else:
            scheduler.step()
        
        print(
            f"Epoch {epoch:03d} | Time: {time.time() - epoch_start_time:.1f}s | "
            f"Train Loss (Norm): {avg_train_loss:.6f} | "
            f"Val RMSE_E: {val_metrics['rmse_E']:.2f} meV/atom | "
            f"Val RMSE_F: {val_metrics['rmse_F']:.2f} meV/A | "
            f"Val RMSE_q: {val_metrics['rmse_q']:.2f} | "
            f"LR: {current_lr:.2e}"
        )

        current_val_metric = val_metrics['val_total']
        if current_val_metric < best_val_metric:
            best_val_metric = current_val_metric
            best_model_state = model.state_dict()
            print(f"  -> New best validation metric: {best_val_metric:.4f}")

    print(f"Training finished. Best Val Metric: {best_val_metric:.4f}")
    return best_model_state

# --- Evaluation Function ---
# 保留旧的 evaluation 函数，但更新以支持反归一化和单位转换
def evaluate_model(model, test_loader, device, stats, wE=1.0, wF=9.0, wq=9.0):
    # 复用 eval_rmse_epoch 逻辑，它已经包含了正确的反归一化和单位转换
    metrics = eval_rmse_epoch(model, test_loader, device, stats, wE=wE, wF=wF, wq=wq)
    
    print("--- Test Set Evaluation Summary ---")
    print(f"  Energy RMSE: {metrics['rmse_E']:.4f} meV/atom")
    print(f"  Force  RMSE: {metrics['rmse_F']:.4f} meV/A")
    print(f"  Charge RMSE: {metrics['rmse_q']:.4f} ")

    return {'MAE': 0.0, 'RMSE_E': metrics['rmse_E'], 'RMSE_F': metrics['rmse_F'], 'RMSE_q': metrics['rmse_q']} # 返回主要指标

# --- Atom Mapping Transform ---
class MapAtomTypes:
    def __init__(self, atom_map):
        self.atom_map = atom_map
        self.unk_idx = atom_map.get('<UNK>', 0)

    def __call__(self, data):
        # 兼容性处理
        if hasattr(data, 'atoms'):
             z_list = data.atoms.tolist()
        elif hasattr(data, 'z'):
             z_list = data.z.tolist()
        else:
             # 如果都没有，假设 data.x 的第一列是原子序数（针对某些数据集）
             z_list = data.x[:, 0].long().tolist()
             
        mapped_atoms = [self.atom_map.get(z, self.unk_idx) for z in z_list]
        data.atoms = torch.tensor(mapped_atoms, dtype=torch.long)
        return data

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description='Charged system Benchmark Script')
    parser.add_argument('--L', type=int, default=4, help='Number of hierarchical layers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_path', type=str, default='/workspace/LRGB/hmp-net-clean/dataset/runner_data_charged_only.pt')
    parser.add_argument('--save_models', action='store_true', help='Save the best model state dict')
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    

    print(f"--- {args.data_path} Benchmark ---")
    print(f"Device: {args.device}")
    device = torch.device(args.device)

    # 1. Load Data
    print(f"Loading data from {args.data_path}...")
    if not os.path.exists(args.data_path):
        print(f"Error: File not found at {args.data_path}")
        return

    dataset = InputDataDataset(args.data_path, keep_ref_charges=True)
    print(f"Loaded n_total={len(dataset)} structures.")

    # 2. Create Atom Map
    print("Scanning dataset for atom types...")
    unique_z = set()
    for data in dataset:
        # 兼容不同数据格式
        if hasattr(data, 'atoms'):
            unique_z.update(data.atoms.tolist())
        elif hasattr(data, 'z'):
             unique_z.update(data.z.tolist())
    
    sorted_z = sorted(list(unique_z))
    atom_map = {z: i for i, z in enumerate(sorted_z)}
    print(f"Atom Map: {atom_map}")
    num_atom_classes = len(atom_map)

    # 3. Preprocessing
    print("Preprocessing data (Mapping atoms & Generating edges)...")
    map_atoms = MapAtomTypes(atom_map)
    compute_edges = RadiusGraph(r=5.0) # 使用合理的截断半径 (5.0 Angstrom ~= 9.5 Bohr)
    
    processed_data_list = []
    for data in tqdm(dataset, desc="Preprocessing"):
        data = map_atoms(data)
        data = compute_edges(data)
        processed_data_list.append(data)

    # 4. Split
    train_set, val_set, test_set = split_dataset(processed_data_list, seed=args.seed)

    # 5. Compute Stats (Critical for Normalization!)
    stats = compute_dataset_stats(train_set, num_atom_classes)

    # 6. DataLoaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 7. Model Configs
    model_configs = get_model_configs(args.L, num_atom_classes)

    results_summary = []

    # 8. Loop Models
    for model_name, config in model_configs.items():
        print(f"\n==========================================")
        print(f"Training Model: {model_name}")
        print(f"==========================================")

        if 'MACE' in model_name or 'TFN' in model_name:
            try:
                import e3nn
            except ImportError:
                print("e3nn not installed, skipping.")
                continue
        
        # Init Model
        model = config['class'](**config['params']).to(args.device)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {param_count:,}")

        # Optimizer
        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5) # Reduced WD
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        
        # Changed to ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )        

        # Train (Using Normalized Loss, Logging meV metrics)
        best_state = train_model(model, train_loader, val_loader, optimizer, scheduler, args.device, args.epochs, stats, wE=1.0, wF=10.0, wq=100.0)
        
        model.load_state_dict(best_state)

        if args.save_models:
            save_dir = os.path.join('results', 'saved_models', 'carbon_chain')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{model_name}_best.pt")
            torch.save(best_state, save_path)

        # Evaluate
        metrics = evaluate_model(model, test_loader, args.device, stats, wE=1.0, wF=10.0, wq=100.0)
        
        results_summary.append({
            'Model': model_name,
            'RMSE_E': metrics['RMSE_E'],
            'RMSE_F': metrics['RMSE_F'],
            'RMSE_q': metrics['RMSE_q'],
            'Params': param_count
        })

    # 9. Report
    print("\n\n=== Final Benchmark Results (meV) ===")
    print(f"{'Model':<15} | {'Params':<10} | {'RMSE_E (meV/atom)':<20}")
    print("-" * 55)
    for res in results_summary:
        print(f"{res['Model']:<15} | {res['Params']:<10,} | {res['RMSE_E']:.4f}, {res['RMSE_F']:.4f}, {res['RMSE_q']:.4f}")

if __name__ == '__main__':
    main()