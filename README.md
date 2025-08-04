# ⚔️ Geometric GNN Dojo

*Geometric GNN Dojo* is a pedagogical resource for beginners and experts to explore the design space of **Graph Neural Networks for geometric graphs**.

Check out the accompanying paper ['On the Expressive Power of Geometric Graph Neural Networks'](https://arxiv.org/abs/2301.09308), which studies the expressivity and theoretical limits of geometric GNNs.
> Chaitanya K. Joshi*, Cristian Bodnar*, Simon V. Mathis, Taco Cohen, and Pietro Liò. On the Expressive Power of Geometric Graph Neural Networks. *International Conference on Machine Learning*.
>
>[PDF](https://arxiv.org/pdf/2301.09308.pdf) | [Slides](https://www.chaitjo.com/publication/joshi-2023-expressive/Geometric_GNNs_Slides.pdf) | [Video](https://youtu.be/5ulJMtpiKGc)

❓**New to geometric GNNs:** try our practical notebook on [*Geometric GNNs 101*](geometric_gnn_101.ipynb), prepared for MPhil students at the University of Cambridge.

<a target="_blank" href="https://colab.research.google.com/github/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab (recommended!)"/>
</a>

## Architectures

The `/models` directory provides unified implementations of several popular geometric GNN architectures:
- Invariant GNNs: [SchNet](https://arxiv.org/abs/1706.08566), [DimeNet](https://arxiv.org/abs/2003.03123), [SphereNet](https://arxiv.org/abs/2102.05013)
- Equivariant GNNs using cartesian vectors: [E(n) Equivariant GNN](https://proceedings.mlr.press/v139/satorras21a.html), [GVP-GNN](https://arxiv.org/abs/2009.01411)
- Equivariant GNNs using spherical tensors: [Tensor Field Network](https://arxiv.org/abs/1802.08219), [MACE](http://arxiv.org/abs/2206.07697)
- 🔥 Your new geometric GNN architecture?

<figure><center><img src="experiments/fig/axes-of-expressivity.png" width="70%"></center></figure>

## Experiments

The `/experiments` directory contains notebooks with synthetic experiments to highlight practical challenges in building powerful geometric GNNs:
- `kchains.ipynb`: Distinguishing k-chains, which test a model's ability to **propagate geometric information** non-locally and demonstrate oversquashing with increased depth/longer chains.
- `rotsym.ipynb`: Rotationally symmetric structures, which test a layer's ability to **identify neighbourhood orientation** and highlight the utility of higher order tensors in equivariant GNNs.
- `incompleteness.ipynb`: Counterexamples from [Pozdnyakov et al.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.166001), which test a layer's ability to create **distinguishing fingerprints for local neighbourhoods** and highlight the need for higher body order of local scalarisation (distances, angles, and beyond).



## Installation

```bash
# Create new conda environment
conda create --prefix ./env python=3.8
conda activate ./env

# Install PyTorch (Check CUDA version for GPU!)
#
# Option 1: CPU
conda install pytorch==1.12.0 -c pytorch
#
# Option 2: GPU, CUDA 11.3
# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Install dependencies
conda install matplotlib pandas networkx
conda install jupyterlab -c conda-forge
pip install e3nn==0.4.4 ipdb ase

# Install PyG (Check CPU/GPU/MacOS)
#
# Option 1: CPU, MacOS
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cpu.html 
pip install torch-geometric
#
# Option 2: GPU, CUDA 11.3
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
# pip install torch-geometric
#
# Option 3: CPU/GPU, but may not work on MacOS
# conda install pyg -c pyg
```


## Directory Structure and Usage

```
.
├── README.md
|
├── geometric_gnn_101.ipynb             # A gentle introduction to Geometric GNNs
| 
├── experiments                         # Synthetic experiments
|   |
│   ├── kchains.ipynb                   # Experiment on k-chains
│   ├── rotsym.ipynb                    # Experiment on rotationally symmetric structures
│   ├── incompleteness.ipynb            # Experiment on counterexamples from Pozdnyakov et al.
|   └── utils                           # Helper functions for training, plotting, etc.
| 
└── models                              # Geometric GNN models library
    |
    ├── schnet.py                       # SchNet model
    ├── dimenet.py                      # DimeNet model
    ├── spherenet.py                    # SphereNet model
    ├── egnn.py                         # E(n) Equivariant GNN model
    ├── gvpgnn.py                       # GVP-GNN model
    ├── tfn.py                          # Tensor Field Network model
    ├── mace.py                         # MACE model
    ├── layers                          # Layers for each model
    ├── hmp                             # Modules and models for HMP-Net
    |   |
    │   ├── master_selection.py         # Master nodes selection module
    │   ├── virtual_generation.py       # Virtual edges generation module
    │   ├── schnet_hmp.py               # HMP Augmented SchNet Model
    │   ├── dimenet_hmp.py              # HMP Augmented DimeNet model
    │   ├── spherenet_hmp.py            # HMP Augmented SphereNet model
    │   ├── egnn_hmp.py                 # HMP Augmented E(n) Equivariant GNN model
    │   ├── gvpgnn_hmp.py               # HMP Augmented GVP-GNN model
    │   ├── tfn_hmp.py                  # HMP Augmented Tensor Field Network model
    │   ├── mace_hmp.py                 # HMP Augmented MACE model
    │   ├── layers                      # Layers for each HMP-Net model
    └── modules                         # Modules and layers for MACE
```


## Contact

Authors: Chaitanya K. Joshi (chaitanya.joshi@cl.cam.ac.uk), Simon V. Mathis (simon.mathis@cl.cam.ac.uk). 
We welcome your questions and feedback via email or GitHub Issues.


## Citation

```
@inproceedings{joshi2023expressive,
  title={On the Expressive Power of Geometric Graph Neural Networks},
  author={Joshi, Chaitanya K. and Bodnar, Cristian and  Mathis, Simon V. and Cohen, Taco and Liò, Pietro},
  booktitle={International Conference on Machine Learning},
  year={2023},
}
```
