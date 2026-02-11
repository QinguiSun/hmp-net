#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import sys
import argparse
import logging
import string
import random
from shutil import copyfile
from datetime import datetime

# 假设这些模块已经转换为 PyTorch 版本
from neural_network.NeuralNetwork import * 
from neural_network.activation_fn import * 
from training.DataContainer import *
from training.DataProvider  import *

# --- 辅助类与函数 ---

def id_generator(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

def segment_sum(data, segment_ids):
    """PyTorch 版 segment_sum"""
    if data.numel() == 0:
        return torch.tensor(0.0, device=data.device, dtype=data.dtype)
    num_segments = segment_ids.max().item() + 1
    shape = (num_segments,) + data.shape[1:]
    output = torch.zeros(shape, dtype=data.dtype, device=data.device)
    output.index_add_(0, segment_ids, data)
    return output

class EMA:
    """
    PyTorch 实现的指数移动平均 (Exponential Moving Average)
    """
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# --- 主程序 ---

logging.basicConfig(filename='train.log', level=logging.DEBUG)

# 定义命令行参数
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument("--restart", type=str, default=None,  help="restart training from a specific folder")
parser.add_argument("--num_features", type=int,   help="dimensionality of feature vectors")
parser.add_argument("--num_basis", type=int,   help="number of radial basis functions")
parser.add_argument("--num_blocks", type=int,   help="number of interaction blocks")
parser.add_argument("--num_residual_atomic", type=int,   help="number of residual layers for atomic refinements")
parser.add_argument("--num_residual_interaction", type=int,   help="number of residual layers for the message phase")
parser.add_argument("--num_residual_output", type=int,   help="number of residual layers for the output blocks")
parser.add_argument("--cutoff", default=10.0, type=float, help="cutoff distance for short range interactions")
parser.add_argument("--use_electrostatic", default=1, type=int,   help="use electrostatics in energy prediction (0/1)")
parser.add_argument("--use_dispersion", default=1, type=int,   help="use dispersion in energy prediction (0/1)")
parser.add_argument("--grimme_s6", default=None, type=float, help="grimme s6 dispersion coefficient")
parser.add_argument("--grimme_s8", default=None, type=float, help="grimme s8 dispersion coefficient")
parser.add_argument("--grimme_a1", default=None, type=float, help="grimme a1 dispersion coefficient")
parser.add_argument("--grimme_a2", default=None, type=float, help="grimme a2 dispersion coefficient")
parser.add_argument("--dataset", type=str,   help="file path to dataset")
parser.add_argument("--num_train", type=int,   help="number of training samples")
parser.add_argument("--num_valid", type=int,   help="number of validation samples")
parser.add_argument("--seed", default=42, type=int,   help="seed for splitting dataset into training/validation/test")
parser.add_argument("--max_steps", type=int,   help="maximum number of training steps")
parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate used by the optimizer")
parser.add_argument("--max_norm", default=1000.0, type=float, help="max norm for gradient clipping")
parser.add_argument("--ema_decay", default=0.999, type=float, help="exponential moving average decay used by the trainer")
parser.add_argument("--keep_prob", default=1.0, type=float, help="keep probability for dropout regularization of rbf layer")
parser.add_argument("--l2lambda", type=float, help="lambda multiplier for l2 loss (regularization)")
parser.add_argument("--nhlambda", type=float, help="lambda multiplier for non-hierarchicality loss (regularization)")
parser.add_argument("--decay_steps", type=int, help="decay the learning rate every N steps by decay_rate")
parser.add_argument("--decay_rate", type=float, help="factor with which the learning rate gets multiplied by every decay_steps steps")
parser.add_argument("--batch_size", type=int, help="batch size used per training step")
parser.add_argument("--valid_batch_size", type=int, help="batch size used for going through validation_set")
parser.add_argument('--force_weight',  default=52.91772105638412, type=float, help="force contribution to loss")
parser.add_argument('--charge_weight', default=14.399645351950548, type=float, help="charge contribution to loss")
parser.add_argument('--dipole_weight', default=27.211386024367243, type=float, help="dipole contribution to loss")
parser.add_argument('--summary_interval', type=int, help="write a summary every N steps")
parser.add_argument('--validation_interval', type=int, help="check performance on validation set every N steps")
parser.add_argument('--save_interval', type=int, help="save progress every N steps")
parser.add_argument('--record_run_metadata', type=int, help="records metadata like memory consumption etc.")

config_file='config.txt'
if len(sys.argv) == 1:
    if os.path.isfile(config_file):
        args = parser.parse_args(["@"+config_file])
    else:
        args = parser.parse_args(["--help"])
else:
    args = parser.parse_args()

# 创建目录
if args.restart is None:
    directory=datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + id_generator() +"_F"+str(args.num_features)+"K"+str(args.num_basis)+"b"+str(args.num_blocks)+"a"+str(args.num_residual_atomic)+"i"+str(args.num_residual_interaction)+"o"+str(args.num_residual_output)+"cut"+str(args.cutoff)+"e"+str(args.use_electrostatic)+"d"+str(args.use_dispersion)+"l2"+str(args.l2lambda)+"nh"+str(args.nhlambda)+"keep"+str(args.keep_prob)
else:
    directory=args.restart

logging.info("creating directories...")
if not os.path.exists(directory):
    os.makedirs(directory)
best_dir = os.path.join(directory, 'best')
if not os.path.exists(best_dir):
    os.makedirs(best_dir)
log_dir = os.path.join(directory, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
best_loss_file  = os.path.join(best_dir, 'best_loss.npz')
best_checkpoint = os.path.join(best_dir, 'best_model.pth')
step_checkpoint = os.path.join(log_dir,  'model.pth')

logging.info("writing args to file...")
with open(os.path.join(directory, config_file), 'w') as f:
    for arg in vars(args):
        f.write('--'+ arg + '='+ str(getattr(args, arg)) + "\n")

# 加载数据集
logging.info("loading dataset...")
data = DataContainer(args.dataset)
data_provider = DataProvider(data, args.num_train, args.num_valid, args.batch_size, args.valid_batch_size, seed=args.seed)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# 创建神经网络
logging.info("creating neural network...")
nn_model = NeuralNetwork(F=args.num_features,           
                   K=args.num_basis,                
                   sr_cut=args.cutoff,              
                   num_blocks=args.num_blocks, 
                   num_residual_atomic=args.num_residual_atomic,
                   num_residual_interaction=args.num_residual_interaction,
                   num_residual_output=args.num_residual_output,
                   use_electrostatic=(args.use_electrostatic==1),
                   use_dispersion=(args.use_dispersion==1),
                   s6=args.grimme_s6,
                   s8=args.grimme_s8,
                   a1=args.grimme_a1,
                   a2=args.grimme_a2,
                   Eshift=data_provider.EperA_mean,  
                   Escale=data_provider.EperA_stdev,   
                   activation_fn=shifted_softplus,
                   seed=None,
                   scope="neural_network")

nn_model.to(device)

# Added: Print model parameter count
param_count = sum(p.numel() for p in nn_model.parameters() if p.requires_grad)
print(f"Model PhysNet has {param_count:,} parameters.")

# 优化器
optimizer = optim.Adam(nn_model.parameters(), lr=args.learning_rate, amsgrad=True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_steps, gamma=args.decay_rate)

# EMA 辅助类
ema = EMA(nn_model, args.ema_decay)

# TensorBoard
summary_writer = SummaryWriter(log_dir=log_dir)

# 加载最佳 Loss 记录
if os.path.isfile(best_loss_file):
    loss_file   = np.load(best_loss_file, allow_pickle=True)
    best_loss   = loss_file["loss"].item()
    best_emae   = loss_file["emae"].item()
    best_ermse  = loss_file["ermse"].item()
    best_fmae   = loss_file["fmae"].item()
    best_frmse  = loss_file["frmse"].item()
    best_qmae   = loss_file["qmae"].item()
    best_qrmse  = loss_file["qrmse"].item()
    best_dmae   = loss_file["dmae"].item()
    best_drmse  = loss_file["drmse"].item()
    best_step   = loss_file["step"].item()
else:
    best_loss  = np.Inf
    best_emae  = np.Inf
    best_ermse = np.Inf
    best_fmae  = np.Inf
    best_frmse = np.Inf
    best_qmae  = np.Inf
    best_qrmse = np.Inf
    best_dmae  = np.Inf
    best_drmse = np.Inf
    best_step  = 0.
    np.savez(best_loss_file, loss=best_loss, emae=best_emae,   ermse=best_ermse, 
                                             fmae=best_fmae,   frmse=best_frmse, 
                                             qmae=best_qmae,   qrmse=best_qrmse, 
                                             dmae=best_dmae,   drmse=best_drmse, 
                                             step=best_step)

def calculate_errors(val1, val2):
    """计算 Loss (MAE), MSE, MAE"""
    delta = torch.abs(val1 - val2)
    delta2 = delta**2
    mse = torch.mean(delta2)
    mae = torch.mean(delta)
    loss = mae
    return loss, mse, mae

def batch_to_tensor(batch_data):
    """将 DataProvider 返回的 numpy/list 字典转换为 PyTorch Tensor"""
    tensors = {}
    
    # 定义需要转换为 Long (int64) 的键
    long_keys = ["Z", "idx_i", "idx_j", "batch_seg", "N"]
    
    for k, v in batch_data.items():
        if v is None:
            tensors[k] = None
            continue
        
        # --- 修复部分：处理 list 类型 ---
        if isinstance(v, list):
            # 尝试转换为 numpy 数组
            # 注意：如果列表为空，np.array([]) 默认为 float64
            v = np.array(v)
        # -----------------------------

        # 检查是否为空数组（例如 offsets 可能为空）
        if v.size == 0:
            # 创建一个空的 tensor，类型根据 key 决定
            if k in long_keys:
                t = torch.tensor([], dtype=torch.long, device=device)
            else:
                t = torch.tensor([], dtype=torch.float, device=device)
        else:
            t = torch.from_numpy(v).to(device)
            if k in long_keys:
                t = t.long()
            else:
                t = t.float()
        
        tensors[k] = t
    return tensors

def run_step(batch_data, is_training=True):
    # 转换数据
    inputs = batch_to_tensor(batch_data)
    
    Eref = inputs.get("E")
    Earef = inputs.get("Ea")
    Fref = inputs.get("F")
    Z = inputs.get("Z")
    Dref = inputs.get("D")
    Qref = inputs.get("Q")
    Qaref = inputs.get("Qa")
    R = inputs.get("R")
    idx_i = inputs.get("idx_i")
    idx_j = inputs.get("idx_j")
    batch_seg = inputs.get("batch_seg")
    
    # 训练模式设置
    if is_training:
        nn_model.train()
        R.requires_grad_(True) # 需要计算力，必须对 R 求导
    else:
        nn_model.eval()
        if data.F is not None:
            R.requires_grad_(True)
    
    # --- 前向传播 ---
    Ea, Qa, Dij, nhloss = nn_model.atomic_properties(Z, R, idx_i, idx_j)
    
    energy, forces = nn_model.energy_and_forces_from_atomic_properties(
        Ea, Qa, Dij, Z, R, idx_i, idx_j, Qref, batch_seg, create_graph=is_training
    )
    
    Qtot = segment_sum(Qa, batch_seg)
    
    if Qa.dim() == 1:
        Qa_expanded = Qa.unsqueeze(1)
    else:
        Qa_expanded = Qa
    QR = Qa_expanded * R
    D = segment_sum(QR, batch_seg)

    # --- 计算 Loss ---
    # Energy
    if data.E is not None:
        eloss, emse, emae = calculate_errors(Eref, energy)
    else:
        eloss, emse, emae = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    
    # Atomic Energy
    if data.Ea is not None:
        ealoss, eamse, eamae = calculate_errors(Earef, Ea)
    else:
        ealoss, eamse, eamae = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    # Forces
    if data.F is not None:
        floss, fmse, fmae = calculate_errors(Fref, forces)
    else:
        floss, fmse, fmae = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    # Charge
    if data.Q is not None:
        qloss, qmse, qmae = calculate_errors(Qref, Qtot)
    else:
        qloss, qmse, qmae = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    # Atomic Charge
    if data.Qa is not None:
        qaloss, qamse, qamae = calculate_errors(Qaref, Qa)
    else:
        qaloss, qamse, qamae = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    # Dipole
    if data.D is not None:
        dloss, dmse, dmae = calculate_errors(Dref, D)
    else:
        dloss, dmse, dmae = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    # 组合 Loss
    loss_energy = eloss
    loss_force = floss
    loss_charge = qloss
    loss_dipole = dloss
    
    if data.Ea is not None:
        loss_energy = ealoss
    
    if data.Qa is not None:
        loss_charge = qaloss
        loss_dipole = torch.tensor(0.0, device=device)

    # L2 Regularization
    l2_reg = torch.tensor(0.0, device=device)
    for param in nn_model.parameters():
        l2_reg += torch.norm(param, 2)**2
    l2_loss = l2_reg * args.l2lambda 

    total_loss = (loss_energy + 
                  args.force_weight * loss_force + 
                  args.charge_weight * loss_charge + 
                  args.dipole_weight * loss_dipole + 
                  args.nhlambda * nhloss + 
                  l2_loss)

    metrics = {
        "loss": total_loss.item(),
        "emse": emse.item(), "emae": emae.item(),
        "fmse": fmse.item(), "fmae": fmae.item(),
        "qmse": qmse.item(), "qmae": qmae.item(),
        "dmse": dmse.item(), "dmae": dmae.item()
    }

    if is_training:
        optimizer.zero_grad()
        total_loss.backward()
        if args.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), args.max_norm)
        optimizer.step()
        ema.update()

    return metrics

def reset_averages():
    return {k: 0.0 for k in ["loss", "emse", "emae", "fmse", "fmae", "qmse", "qmae", "dmse", "dmae"]}

def update_averages(avgs, step_metrics, count):
    count += 1
    for k in avgs:
        avgs[k] += (step_metrics[k] - avgs[k]) / count
    return avgs, count

# 恢复 Checkpoint
start_step = 0
if os.path.exists(step_checkpoint):
    logging.info(f"Restoring checkpoint from {step_checkpoint}")
    checkpoint = torch.load(step_checkpoint, map_location=device)
    nn_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step']
    if 'ema_shadow' in checkpoint:
        ema.shadow = checkpoint['ema_shadow']
    logging.info(f"Resuming from step {start_step}")

# 初始化训练统计
train_avgs = reset_averages()
train_count = 0

logging.info("starting training...")

# 训练循环
for step in range(start_step + 1, args.max_steps + 1):
    
    batch_data = data_provider.next_batch()
    
    metrics = run_step(batch_data, is_training=True)
    
    train_avgs, train_count = update_averages(train_avgs, metrics, train_count)
    
    scheduler.step()

    if step % args.save_interval == 0:
        torch.save({
            'step': step,
            'model_state_dict': nn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_shadow': ema.shadow
        }, step_checkpoint)

    if step % args.validation_interval == 0:
        ema.apply_shadow()
        
        valid_avgs = reset_averages()
        valid_count = 0
        num_valid_batches = args.num_valid // args.valid_batch_size
        
        for _ in range(num_valid_batches):
            v_batch = data_provider.next_valid_batch()
            v_metrics = run_step(v_batch, is_training=False)
            valid_avgs, valid_count = update_averages(valid_avgs, v_metrics, valid_count)
        
        ema.restore()

        results = {}
        results["loss_valid"] = valid_avgs["loss"]
        if data.E is not None:
            results["energy_mae_valid"] = valid_avgs["emae"]
            results["energy_rmse_valid"] = np.sqrt(valid_avgs["emse"])
        if data.F is not None:
            results["forces_mae_valid"] = valid_avgs["fmae"]
            results["forces_rmse_valid"] = np.sqrt(valid_avgs["fmse"])
        if data.Q is not None:
            results["charge_mae_valid"] = valid_avgs["qmae"]
            results["charge_rmse_valid"] = np.sqrt(valid_avgs["qmse"])
        if data.D is not None:
            results["dipole_mae_valid"] = valid_avgs["dmae"]
            results["dipole_rmse_valid"] = np.sqrt(valid_avgs["dmse"])

        if results["loss_valid"] < best_loss:
            best_loss = results["loss_valid"]
            best_step = step
            
            if data.E is not None:
                best_emae = results["energy_mae_valid"]
                best_ermse = results["energy_rmse_valid"]
            if data.F is not None:
                best_fmae = results["forces_mae_valid"]
                best_frmse = results["forces_rmse_valid"]
            if data.Q is not None:
                best_qmae = results["charge_mae_valid"]
                best_qrmse = results["charge_rmse_valid"]
            if data.D is not None:
                best_dmae = results["dipole_mae_valid"]
                best_drmse = results["dipole_rmse_valid"]

            np.savez(best_loss_file, loss=best_loss, emae=best_emae, ermse=best_ermse, 
                     fmae=best_fmae, frmse=best_frmse, qmae=best_qmae, qrmse=best_qrmse, 
                     dmae=best_dmae, drmse=best_drmse, step=best_step)
            
            ema.apply_shadow()
            torch.save(nn_model.state_dict(), best_checkpoint)
            ema.restore()
            
            logging.info(f"New best loss: {best_loss:.6f} at step {step}")

        for k, v in results.items():
            summary_writer.add_scalar(f"Valid/{k}", v, step)
        summary_writer.add_scalar("Valid/Best_Loss", best_loss, step)

    if step % args.summary_interval == 0 and step > 0:
        results_train = {}
        results_train["loss_train"] = train_avgs["loss"]
        if data.E is not None:
            results_train["energy_mae_train"] = train_avgs["emae"]
            results_train["energy_rmse_train"] = np.sqrt(train_avgs["emse"])
        if data.F is not None:
            results_train["forces_mae_train"] = train_avgs["fmae"]
            results_train["forces_rmse_train"] = np.sqrt(train_avgs["fmse"])
        
        for k, v in results_train.items():
            summary_writer.add_scalar(f"Train/{k}", v, step)
        
        log_str = f"{step}/{args.max_steps} loss: {results_train['loss_train']:.6f} best: {best_loss:.6f}"
        if data.E is not None:
            log_str += f" emae: {results_train['energy_mae_train']:.6f} best_emae: {best_emae:.6f}"
        print(log_str)
        logging.info(log_str)

        train_avgs = reset_averages()
        train_count = 0

summary_writer.close()