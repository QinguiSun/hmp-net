# ⚔️ HMP-Net (Public Release)

  

Reproducible code for our paper. This repository provides training & evaluation scripts for geometric GNNs on 3D atomic systems.

Please see **Getting Started** and **Reproduction** below.

  

## Getting Started

```bash

conda create -n hmpnet python=3.10 -y

conda activate hmpnet

pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117

pip install pyg-lib==0.4.0+pt20cu117 \
  torch-scatter==2.1.2+pt20cu117 \
  torch-sparse==0.6.18+pt20cu117 \
  torch-cluster==1.6.3+pt20cu117 \
  torch-spline-conv==1.2.2+pt20cu117 \
  -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

pip install -r requirements-core.txt

```

  

## Reproductive

To download datasets from the following and save them in the `datasets` directory.
> Ko, T.W., Finkler, J. A., Goedecker, S. & Behler, J. A fourth-generation high-dimensional neural network potential with accurate electrostatics including non-local charge transfer. Materials Cloud Archive 2020.X, [https://doi.org/10.24435/materialscloud:f3-yh](https://doi.org/10.24435/materialscloud:f3-yh) (2020).

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_benchmark_charged.py \
	  --data_path ./datasets/Carbon_chain/input.data \
    --L 6 \
    --epochs 1500 \
    --batch_size 64 \
    --lr 1e-4 \
    --device cuda \
    --save_models  > run_gpu0_schnetVN_CC_L6_1e-4_q-atom_q-tot_v4.log 1>&1 &

```

```bash

CCUDA_VISIBLE_DEVICES=0 python experiments/run_benchmark_charged.py \
	  --data_path ./datasets/Ag_cluster/input.data \
    --L 6 \
    --epochs 1500 \
    --batch_size 64 \
    --lr 1e-4 \
    --device cuda \
    --save_models  > run_gpu0_schnetVN_Ag_L5_1e-4_q-atom_q-512_v8.log 1>&1 &

```

  ```bash
  CUDA_VISIBLE_DEVICES=1 python experiments/run_benchmark_charged.py \
	  --data_path ./datasets/NaCl/input.data \
    --L 6 \
    --epochs 1500 \
    --batch_size 64 \
    --lr 1e-4 \
    --device cuda \
    --save_models  > run_gpu1_schnetVN_NaCl_L6_1e-4_q-atom_q-tot_v5.log 1>&1 &
  ```

## Architectures

  
## Directory Structure and Usage

  

```

.
├── README.md
|
├── experiments # Synthetic experiments
| |
│ ├── run_benchmark_charged.ipynb # Experiment on charged systems
|
└── models # Geometric GNN models library
| |
| ├── schnet.py # SchNet model
| |── dimenet.py # DimeNet model
| |── spherenet.py # SphereNet model
| ├── egnn.py # E(n) Equivariant GNN model
| ├── gvpgnn.py # GVP-GNN model
| ├── tfn.py # Tensor Field Network model
| ├── mace.py # MACE model
| ├── layers # Layers for each model
|
├── hmp # Modules and models for HMP-Net
| |
│ ├── master_selection.py # Master nodes selection module
│ ├── virtual_generation.py # Virtual edges generation module
│ ├── schnet_hmp.py # HMP Augmented SchNet Model
│ ├── dimenet_hmp.py # HMP Augmented DimeNet model
│ ├── spherenet_hmp.py # HMP Augmented SphereNet model
│ ├── egnn_hmp.py # HMP Augmented E(n) Equivariant GNN model
│ ├── gvpgnn_hmp.py # HMP Augmented GVP-GNN model
│ ├── tfn_hmp.py # HMP Augmented Tensor Field Network model
│ ├── mace_hmp.py # HMP Augmented MACE model
│ ├── layers # Layers for each HMP-Net model
└── modules # Modules and layers for MACE

```

  

License

  

MIT (see LICENSE).

  

## Contact

qingui.sun@outlook.com

  
  
  

## Acknowledgements

  

This repository **reuses and adapts baseline implementations** (SchNet, DimeNet, SphereNet, EGNN, GVP-GNN, TFN, MACE) from

[Geometric GNN Dojo](https://github.com/chaitjo/geometric-gnn-dojo) (MIT License).

We sincerely thank the authors for making their work available.

  

If you build on this repository, please also cite:

  

> Chaitanya K. Joshi, Cristian Bodnar, Simon V. Mathis, Taco Cohen, Pietro Liò.

> **On the Expressive Power of Geometric Graph Neural Networks.** ICML 2023.
