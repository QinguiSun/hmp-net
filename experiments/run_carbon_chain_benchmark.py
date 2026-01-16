# experiments/run_carbon_chain_benchmark.py
"""
CUDA_VISIBLE_DEVICES=1 python experiments/run_carbon_chain_benchmark.py \\
  --data-path ./dataset/input.data \\
  --num-layers 3 \\
  --epochs 300 --batch-size 64 --lr 5e-4 --cutoff 10.0 \\
  --use-long-range

# 开启 latent-charge Coulomb head:
python experiments/run_carbon_chain_benchmark.py \\
  --data-path ./dataset/input.data \\
  --epochs 300 --batch-size 64 --lr 5e-4 --cutoff 10.0 \\
  --use-long-range

Trains SchNet (+ optional long-range Coulomb head) on Energy + Forces only.
"""
import argparse
import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from models.dataset_input import InputDataDataset
from models.schnet import SchNetModel
from models.hmp.schnet_hmp_mlp import HMP_SchNetModel

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
    E_graph: (B,1) or (B,) per-graph energy
    pos: (N,3) leaf tensor with requires_grad=True
    returns: (N,3) forces = -dE/dR
    """
    if not pos.requires_grad:
        raise RuntimeError("pos must require grad BEFORE forward() to backprop force loss.")

    E_sum = E_graph.sum()
    grad_pos = torch.autograd.grad(
        outputs=E_sum,
        inputs=pos,
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return -grad_pos


@torch.no_grad()
def eval_one_epoch(model, loader, device, wE=1.0, wF=9.0):
    model.eval()
    mae_E = 0.0
    mae_F = 0.0
    total = 0

    for batch in loader:
        batch = batch.to(device)

        # forces need grad, so temporarily enable grad for pos only
        with torch.enable_grad():
            batch.pos = batch.pos.detach().requires_grad_(True)
            E_pred, aux = model(batch, q_tot=batch.q_tot.view(-1), return_aux=True)  # (B,1)
            F_pred = compute_forces(E_pred, batch.pos, create_graph=False)

        E_true = batch.y.view(-1)
        F_true = batch.forces

        mae_E += (E_pred.view(-1) - E_true).abs().mean().item()
        mae_F += (F_pred - F_true).abs().mean().item()
        total += 1

    if total == 0:
        return {"mae_E": float("inf"), "mae_F": float("inf"), "val_total": float("inf")}

    mae_E /= total
    mae_F /= total
    val_total = wE * mae_E + wF * mae_F
    return {"mae_E": mae_E, "mae_F": mae_F, "val_total": val_total}


def main():
    p = argparse.ArgumentParser("CarbonChain training (Energy + Forces only)")
    p.add_argument("--data-path", type=str, required=True, help="path to input.data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # SchNet params
    p.add_argument("--hidden-channels", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--cutoff", type=float, default=10.0, help="cutoff in Angstrom (will be converted to Bohr)")
    p.add_argument("--max-num-neighbors", type=int, default=32)
    p.add_argument("--use-long-range", action="store_true", help="enable latent-charge Coulomb head")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Dataset
    dataset = InputDataDataset(args.data_path, keep_ref_charges=True)  # ref charges kept but NOT used
    train_set, val_set, test_set = split_dataset(dataset, seed=args.seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print(f"[dataset] n_total={len(dataset)} | train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    # cutoff: interpret as Angstrom, convert to Bohr (data is in Bohr)
    ANG_TO_BOHR = 1.8897261245650618
    cutoff_bohr = args.cutoff * ANG_TO_BOHR
    print(f"[cutoff] {args.cutoff:.3f} Å -> {cutoff_bohr:.6f} Bohr")

    model = HMP_SchNetModel(
        hidden_channels=args.hidden_channels,
        num_embeddings=100,              # must cover atomic numbers up to Ag=47
        num_gaussians=300,
        master_rate = 0.25,
            lambda_attn = 0.01,
            s_dim = 32,
        master_selection_hidden_dim = 64,
        num_layers=args.num_layers,
        cutoff=cutoff_bohr,
        max_num_neighbors=args.max_num_neighbors,
        use_long_range=args.use_long_range,
    ).to(device)

    # loss weights (Energy + Forces only)
    wE = 1.0
    wF = 9.0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        sum_loss = 0.0
        sum_e = 0.0
        sum_f = 0.0
        n_batches = 0

        for it, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            batch.pos = batch.pos.detach().requires_grad_(True)  # make leaf, enable grad

            opt.zero_grad(set_to_none=True)

            # Pass graph total charge (q_tot) only for constraint inside model; NOT as supervision
            E_pred, aux = model(batch, q_tot=batch.q_tot.view(-1), return_aux=True)  # (B,1)
            #print(f"Debug: train out shape {E_pred.shape}")  # Debug line
            #print(f"Debug: train out sample {E_pred[:5]}")  # Debug line
            #print(f"Debug: train true sample {batch.y.view(-1)[:5]}")  # Debug line
            #print(f"Debug: aux sample {aux}")  # Debug line
            F_pred = compute_forces(E_pred, batch.pos, create_graph=True)

            E_true = batch.y.view(-1)
            F_true = batch.forces

            loss_E = F.mse_loss(E_pred.view(-1), E_true)
            loss_F = F.mse_loss(F_pred, F_true)
            loss = wE * loss_E + wF * loss_F

            loss.backward()
            opt.step()

            sum_loss += loss.item()
            sum_e += loss_E.item()
            sum_f += loss_F.item()
            n_batches += 1

            # ---- debug prints (only first batch of epoch 1 and every 50 epochs) ----
            if (epoch == 1 and it == 1) or (epoch % 50 == 0 and it == 1):
                # edge count and sanity checks
                n_edges = int(aux.get("n_edges", torch.tensor([-1], device=device)).item())
                q_sum_max = float(aux.get("q_sum_abs_max", torch.tensor([-1.0], device=device)).item())
                lam_lr = float(aux.get("lambda_lr_mean", torch.tensor([-1.0], device=device)).item())
                print(
                    f"[debug] epoch={epoch} it={it} | "
                    f"pos.requires_grad={batch.pos.requires_grad} | "
                    f"E_pred_mean={E_pred.mean().item():.6f} E_true_mean={E_true.mean().item():.6f} | "
                    f"|F_pred|_mean={F_pred.norm(dim=-1).mean().item():.6f} | "
                    f"edges={n_edges} | lambda_lr_mean={lam_lr:.4f} | q_sum_abs_max={q_sum_max:.3e}"
                )

        train_loss = sum_loss / max(n_batches, 1)
        train_e = sum_e / max(n_batches, 1)
        train_f = sum_f / max(n_batches, 1)

        # validation
        val_metrics = eval_one_epoch(model, val_loader, device, wE=wE, wF=wF)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} (mseE={train_e:.6f}, mseF={train_f:.6f}) | "
            f"val_total={val_metrics['val_total']:.6f} "
            f"(maeE={val_metrics['mae_E']:.6f}, maeF={val_metrics['mae_F']:.6f}) | "
            f"time={dt:.1f}s"
        )

        if val_metrics["val_total"] < best_val:
            best_val = val_metrics["val_total"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # load best and test
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = eval_one_epoch(model, test_loader, device, wE=wE, wF=wF)
    print(
        f"[test] val_total={test_metrics['val_total']:.6f} "
        f"(maeE={test_metrics['mae_E']:.6f}, maeF={test_metrics['mae_F']:.6f})"
    )


if __name__ == "__main__":
    main()
