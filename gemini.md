The following is Section 3 (“Proposed Method: HMP-Net”) and Section 4 (“Experimental Settings”) of our paper draft *Hierarchical Message Passing for Quantum Chemistry*, plus the global outline (Sections 1–7).

# paper draft

### **3 Proposed Method: Hierarchical Message Passing Network (HMP-Net)**

The central hypothesis of this work is that the predictive accuracy of graph neural networks for quantum chemistry can be significantly enhanced by creating explicit, learnable pathways for long-range information flow. Standard Message Passing Neural Networks (MPNNs) are inherently local, propagating information only across existing covalent bonds. This limitation, often termed over-squashing, becomes a critical bottleneck in large molecular systems where non-local electronic effects, such as conjugation, electrostatic interactions, and van der Waals forces, dictate key molecular properties.

To address this, we introduce the Hierarchical Message Passing Network (HMP-Net), a novel architecture that dynamically augments the molecular graph with a hierarchy of virtual connections. HMP-Net learns to identify functionally significant atoms or molecular fragments as "master nodes" and then generates "virtual edges" between them. These edges act as information highways, enabling direct message passing between distant parts of a molecule. This process is fully differentiable and integrated into an end-to-end learning framework. In this section, we detail the components of the HMP-Net architecture: the overall framework, the master node selection mechanism, the attention-based generation of virtual edges, the hierarchical message passing scheme, and the composite loss function that guides the learning process.

#### **3.1 Framework Overview (整体框架概述)**

We represent a molecule as a graph $G = (\mathcal{V}, \mathcal{E})$, where $\mathcal{V}$ is the set of $n$ atoms and $\mathcal{E}$ is the set of covalent bonds. Each atom $i \in \mathcal{V}$ is associated with a feature vector $\mathbf{h}_i \in \mathbb{R}^d$ (e.g., encoding atomic number, charge, and hybridization state) and spatial coordinates $\mathbf{x}_i \in \mathbb{R}^3$. The goal is to learn a mapping $f: G \to y$, where $y \in \mathbb{R}$ is a target quantum-chemical property, such as formation energy or dipole moment.

Standard MPNNs learn atomic representations by iteratively updating node features based on local neighborhoods [^Gilmer et al., 2017]. A single layer of message passing can be expressed as:
$$
\mathbf{h}_i^{(k+1)} = \text{UPDATE}^{(k)}\left(\mathbf{h}_i^{(k)}, \bigoplus_{j \in \mathcal{N}(i)} \text{MESSAGE}^{(k)}(\mathbf{h}_i^{(k)}, \mathbf{h}_j^{(k)}, \mathbf{e}_{ij})\right)
$$
where $\bigoplus$ is a permutation-invariant aggregation function (e.g., sum), and MESSAGE and UPDATE are learnable functions, typically multilayer perceptrons (MLPs). For tasks in quantum chemistry, it is often beneficial to use E(n)-equivariant GNNs that also process atomic coordinates [^Satorras et al., 2021].

The HMP-Net architecture builds upon this foundation by introducing a hierarchical structure, as illustrated in Figure 1. An HMP-Net model consists of $L$ stacked layers. Each layer $l$ performs a three-stage process:

1.  **Local Propagation:** A standard MPNN module operates on the covalent bond graph to refine local atomic representations.
2.  **Master Node Selection:** A differentiable, Gumbel-Softmax-based mechanism selects a subset of atoms to serve as master nodes, acting as hubs for long-range communication.
3.  **Virtual Edge Generation:** An attention-based mechanism computes a sparse, weighted adjacency matrix for the master nodes, creating virtual edges that model non-local interactions.

The resulting hierarchical graph, comprising both original atoms and master nodes with their virtual connections, is then used for a subsequent round of message passing before the process repeats in the next layer. This allows the model to reason about molecular structure at multiple scales simultaneously.

<br>
<div align="center">
  [Figure 1: High-level illustration of our proposed method HMP-Net. At each hierarchical layer, we first run a message passing layer to obtain embeddings of atoms. We then use these learned embeddings to select master nodes and to generate virtual edges, forming a master graph. Another message passing layer operates on this master graph to capture long-range interactions. This process is repeated for L layers, and the final representations are used to predict molecular properties.]
</div>
<br>

#### **3.2 Master Node Selection Mechanism (详细描述Gumbel-Softmax选择过程和公式)**

To create an information hierarchy, the model must first identify which atoms are most salient for long-range communication. We formulate this as a differentiable node selection problem. Instead of relying on fixed heuristics, we employ the Gumbel-Softmax estimator to allow the network to learn which atoms should be promoted to master nodes in a data-driven, end-to-end fashion [^Jang et al., 2016].

Given the atom representations $\mathbf{H}^{(l)}$ at layer $l$, we first compute a scalar selection score $\alpha_i$ for each atom $i$ using a small MLP, followed by a softmax to obtain probabilities:
$$
\boldsymbol{\pi}^{(l)} = \text{softmax}(\text{MLP}_{\text{select}}(\mathbf{H}^{(l)}))
$$
where $\boldsymbol{\pi}^{(l)} \in \mathbb{R}^n$ is the vector of probabilities for each atom being selected.

Directly sampling from the categorical distribution parameterized by $\boldsymbol{\pi}^{(l)}$ is non-differentiable. The Gumbel-Max trick provides a way to draw a sample $z$ by finding the index of the maximum of Gumbel-perturbed log-probabilities:
$$
z = \text{one\_hot}\left(\arg\max_i (g_i + \log \pi_i^{(l)})\right)
$$
where $g_i$ are i.i.d. samples from a Gumbel(0,1) distribution. To make this process differentiable, the non-differentiable $\arg\max$ function is replaced with a `softmax` function, yielding the Gumbel-Softmax distribution:
$$
y_i = \frac{\exp((\log(\pi_i^{(l)}) + g_i)/\tau)}{\sum_{j=1}^n \exp((\log(\pi_j^{(l)}) + g_j)/\tau)} \quad \text{for } i=1, \dots, n
$$
Here, $\tau$ is a temperature parameter. As $\tau \to 0$, the samples $\mathbf{y} = [y_1, \dots, y_n]$ approach one-hot vectors, mimicking a true categorical sample. In practice, we anneal $\tau$ from a high initial value to a small, non-zero value during training.

For our task, we need a discrete set of master nodes. We use the Straight-Through (ST) Gumbel-Softmax estimator. In the forward pass, we obtain a discrete binary mask $\mathbf{m}^{(l)} \in \{0,1\}^n$ by thresholding the continuous samples $\mathbf{y}$. In the backward pass, we pass gradients through the continuous approximation $\mathbf{y}$ to update the parameters of $\text{MLP}_{\text{select}}$: $\nabla_{\boldsymbol{\pi}} \mathcal{L} \approx \nabla_{\mathbf{y}} \mathcal{L}$. This allows the model to learn a stochastic selection policy that is sparse and discrete while remaining trainable via backpropagation. Unlike methods that rely on fixed heuristics or non-parameterized distance metrics to select nodes, our Gumbel-Softmax approach provides a flexible, parameterized mechanism that is learned end-to-end with the primary task objective. <<CHECK_ACCURACY>> (The original potentially inaccurate comparison to HGP-SL has been replaced with a more general, accurate statement).
### **3.3 Virtual Edge Generation (详细描述可学习邻接矩阵的生成)**

After master node selection, virtual edges are generated to facilitate long-range interactions. Given selected master nodes $\mathcal{V}_M^{(l)}$, we construct a sparse, weighted adjacency matrix $\mathbf{A}_{\text{virtual}}^{(l)}$ via an attention mechanism.

To ensure that the learned graph topology is invariant to rotations and reflections, the attention scores are computed using only the scalar (invariant) components of the atomic features. Let $\mathbf{s}_p$ and $\mathbf{s}_q$ be the scalar feature vectors for master nodes $p$ and $q$. The unnormalized attention score $e_{pq}$ is calculated as:
$$
e_{pq} = \text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{s}_p \| \mathbf{s}_q]\right) + \lambda \cdot \mathbf{A}_{\text{induced}}^{(l)}(p,q)
$$
where $\mathbf{a}$ is a learnable weight vector, $\|$ denotes concatenation, and $\lambda$ is a hyperparameter balancing learned attention with the original graph structure. This design guarantees that the resulting virtual graph is identical regardless of the molecule's orientation.

To maintain sparsity and computational efficiency, we apply the **sparsemax** function [Martins & Astudillo, 2016] to normalize the scores for each node, which projects the scores onto the probability simplex and sets many to exactly zero:
$$
\mathbf{s}_p = \text{sparsemax}(\mathbf{e}_p) = \left[\mathbf{e}_p - \tau(\mathbf{e}_p)\mathbf{1}\right]_+
$$
The resulting sparse, non-negative, and invariant matrix $\mathbf{A}_{\text{virtual}}^{(l)}$ encodes significant non-local interactions.

### **3.4 Hierarchical Message Passing on the Final Graph (描述异构图消息传递流程)**

A key design principle of HMP-Net is its model-agnosticism. Rather than being a monolithic architecture, HMP-Net is a modular framework designed to augment any existing message-passing GNN, which we term the "backbone," with hierarchical communication channels.

Let $\mathcal{M}$ be an arbitrary geometric GNN layer (e.g., from EGNN or MACE). This layer takes features $\mathbf{H}$, coordinates $\mathbf{X}$, and an adjacency matrix $\mathbf{A}$ to produce updated outputs: $(\mathbf{H}', \mathbf{X}') = \mathcal{M}(\mathbf{H}, \mathbf{X}, \mathbf{A})$. An HMP-Layer wraps this function $\mathcal{M}$ as follows:

1.  **Local Propagation**: The backbone layer $\mathcal{M}$ is first applied to the original covalent graph $G=(\mathcal{V}, \mathcal{E})$:
    $$
    \mathbf{H}'^{(l)}, \mathbf{X}'^{(l)} = \mathcal{M}(\mathbf{H}^{(l)}, \mathbf{X}^{(l)}, \mathcal{E})
    $$

2.  **Invariant Topology Learning**: Using the invariant scalar components of $\mathbf{H}'^{(l)}$, the master node mask $\mathbf{m}^{(l)}$ and virtual adjacency matrix $\mathbf{A}_{\text{virtual}}^{(l)}$ are generated as described in Sections 3.2 and 3.3.

3.  **Hierarchical Propagation**: The *same* backbone layer $\mathcal{M}$ is now applied to the hierarchical graph $G_M^{(l)}$ formed by the master nodes and the composite adjacency matrix $\mathbf{A}_M^{(l)} = \mathbf{A}_{\text{induced}}^{(l)} + \mathbf{A}_{\text{virtual}}^{(l)}$:
    $$
    \mathbf{H}_M''^{(l)}, \mathbf{X}_M''^{(l)} = \mathcal{M}(\mathbf{H}_M'^{(l)}, \mathbf{X}_M'^{(l)}, \mathbf{A}_M^{(l)})
    $$

4.  **Feature Aggregation**: The final features are a multiplexed combination of the local and hierarchical features, controlled by the invariant mask $\mathbf{m}^{(l)}$:
    $$
    \mathbf{h}_i^{(l+1)} = m_i^{(l)} \cdot \mathbf{h}_{M,i}''^{(l)} + (1 - m_i^{(l)}) \cdot \mathbf{h}_i'^{(l)}
    $$

#### **3.4.1 Preservation of Equivariance**
The HMP-Layer architecture is explicitly designed to preserve the E(3) equivariance of its backbone layer $\mathcal{M}$. This property is critical for applications in the physical sciences. Equivariance is maintained through the following sequence:
1.  The local propagation step is equivariant by definition of the backbone $\mathcal{M}$.
2.  The master node selection and virtual edge generation steps are designed to be **invariant** by operating exclusively on scalar features. The resulting mask $\mathbf{m}^{(l)}$ and adjacency matrix $\mathbf{A}_M^{(l)}$ do not change under rotation.
3.  The hierarchical propagation step is equivariant because it is an application of the equivariant function $\mathcal{M}$ to correctly transformed features $(\mathbf{H}'_M, \mathbf{X}'_M)$ and an invariant graph structure $\mathbf{A}_M^{(l)}$.
4.  The final feature aggregation is a linear combination of equivariant features scaled by invariant scalars ($m_i^{(l)}$ and $1-m_i^{(l)}$), which is an equivariant operation.

Therefore, if the backbone layer $\mathcal{M}$ is E(3)-equivariant, the full HMP-Layer is also E(3)-equivariant.
#### **3.5 Loss Function Design (详细定义L_task, L_sparsity等)**

The HMP-Net is trained end-to-end by minimizing a composite loss function that balances predictive accuracy with structural regularization. The total loss $\mathcal{L}$ is a weighted sum of three components:

1.  **Task Loss ($\mathcal{L}_{\text{task}}$):** This is the primary objective function, which measures the discrepancy between the predicted property $\hat{y}$ and the ground truth $y$. For regression tasks like energy prediction, we use the Mean Squared Error (MSE):
    $$
    \mathcal{L}_{\text{task}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
    $$
    where $N$ is the number of molecules in a batch.

2.  **Structure Regularization ($\mathcal{L}_{\text{structure}}$):** To encourage the generation of a sparse and efficient virtual graph, we apply an L1 penalty to the learned virtual adjacency matrix at each layer. This promotes models where long-range information is routed through a minimal set of critical pathways.
    $$
    \mathcal{L}_{\text{structure}} = \sum_{l=1}^{L} ||\mathbf{A}_{\text{virtual}}^{(l)}||_1
    $$

3.  **Master Node Rate Regularization ($\mathcal{L}_{\text{rate}}$):** To control the complexity of the hierarchical graph, we add a term that encourages the proportion of selected master nodes to be close to a predefined target rate $r \in (0, 1)$.
    $$
    \mathcal{L}_{\text{rate}} = \sum_{l=1}^{L} \left( \frac{||\mathbf{m}^{(l)}||_1}{n} - r \right)^2
    $$

The final loss function is:
$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_{\text{struct}} \mathcal{L}_{\text{structure}} + \lambda_{\text{rate}} \mathcal{L}_{\text{rate}}
$$
where $\lambda_{\text{struct}}$ and $\lambda_{\text{rate}}$ are scalar hyperparameters that control the strength of the regularization terms. This multi-part objective ensures that the model not only learns to make accurate predictions but also discovers meaningful and efficient hierarchical representations of molecular structures.

### **4. Experimental Settings**

To rigorously evaluate the performance of HMP-Net, we designed a series of experiments spanning synthetic benchmarks that explicitly test for over-squashing and real-world quantum chemistry datasets where long-range interactions are critical for accurate property prediction. This section details the datasets, backbone architectures, implementation specifics, and evaluation metrics used in our study.

#### **4.1 Datasets and Tasks**
 ethanol, and toluene [^Chmiela 2017]. The task is to predict the potential energy and interatomic forces for different molecular conformations. Accurate force prediction is exceptionally demanding as it corresponds to the gradient of the potential energy surface and is highly sensitive to subtle changes in atomic positions. This benchmark tests HMP-Net's ability to dynamically adapt its learned virtual graph to different geometries of the same molecule.

*   **Tautobase:** We introduce a task based on the Tautobase dataset, which contains pairs of constitutional isomers (tautomers) that readily interconvert [^Wahl 2021]. The task is to predict the tautomerization energy—the difference in formation energy between the two stable forms. This is a challenging problem because tautomerization often involves significant electronic rearrangement, including proton transfers and the shifting of double bonds across the molecular skeleton. Success on this task requires a model to precisely capture the subtle, non-local electronic effects that determine relative stability, making it an ideal use case for HMP-Net.
#### **4.2 Backbone Architectures and Baseline Models**

To demonstrate the generality of our framework and ensure a fair evaluation, we apply HMP-Net as a modular enhancement to a wide range of state-of-the-art GNNs. For each backbone architecture, we compare the performance of the original model against its HMP-Net-enhanced version (e.g., MACE vs. HMP-MACE). This approach guarantees a controlled experiment where the only variable is the presence of the hierarchical message passing mechanism.

The selected backbone models are grouped into two categories:

*   **Invariant Geometric GNNs:**
    *   **SchNet** [^Schuett 2017], **DimeNet** [^Klicpera 2020], **SphereNet** [^Liu 2022]

*   **Equivariant Geometric GNNs:** We select backbones with varying representational complexity to test the robustness of our approach.
    *   **E(n)-GNN (EGNN)** [^Satorras et al., 2021] (uses order-1 tensors/vectors)
    *   **GVP-GNN** [^Jing 2021] (uses multiple vector channels)
    *   **Tensor Field Network (TFN)** [^Thomas 2018] (uses higher-order tensors)
    *   **MACE** [^Batatia 2022] (uses higher-order tensors and high body-order interactions)

#### **4.3 Implementation Details and Hyperparameters**

Our HMP-Net framework is implemented as a wrapper in PyTorch that can be applied to any GNN architecture. For each baseline model, we construct an HMP-Net variant (e.g., HMP-TFN) by replacing the baseline's standard message-passing layers with our HMP-Layers as described in Section 3.4.

*   **Backbone Integrity:** To ensure a fair comparison, the core message-passing function $\mathcal{M}$ within each HMP-Layer is identical to that of the corresponding baseline. We maintain the same hidden dimensions, number of layers, tensor orders, and body orders. For example, HMP-MACE uses the same high-order tensor products as the original MACE for both local and hierarchical propagation.
*   **Architecture:** We tested models with $L \in \{3, 4, 6\}$ hierarchical layers.
*   **Master Node Selection:** The target master node rate $r$ was set to 0.25. The Gumbel-Softmax temperature $\tau$ was annealed from 1.0 down to 0.1 over the first 50% of training epochs.
*   **Training:** All models (baselines and HMP-Net variants) were trained using the AdamW optimizer with an initial learning rate of $1 \times 10^{-3}$ and a cosine annealing schedule.
*   **Loss Function:** The regularization weights were set to $\lambda_{\text{struct}} = 0.01$ and $\lambda_{\text{rate}} = 0.1$.

#### **4.4 Evaluation Metrics**
The performance of all models was assessed using standard metrics appropriate for each task.

*   **Regression Tasks (QM9, MD17, Tautobase):** For all property and energy/force prediction tasks, we report the **Mean Absolute Error (MAE)** as the primary metric. We also report the **Root Mean Squared Error (RMSE)** for completeness.
*   **Classification Task ($k$-chains):** For the synthetic $k$-chain benchmark, we report the **classification accuracy**.

# Reference
[^Geimer et al., 2017]: Gilmer, Justin et al. “Neural Message Passing for Quantum Chemistry.” _International Conference on Machine Learning_ (2017).
[^Schütt et al., 2021]: Schütt, Kristof, Oliver Unke, and Michael Gastegger. "Equivariant message passing for the prediction of tensorial properties and molecular spectra." _International conference on machine learning_. PMLR, 2021.
[^Zhang et al., 2019]: Zhang, Z., Bu, J., Ester, M., Zhang, J., Yao, C., Yu, Z., & Wang, C. (2019). Hierarchical graph pooling with structure learning. _arXiv preprint arXiv:1911.05954_.
[^Ying et al., 2021]: Ying, Chengxuan, et al. "Do transformers really perform badly for graph representation?." _Advances in neural information processing systems_ 34 (2021): 28877-28888.
[^Hamilton et al., 2017]:  Hamilton, Will, Zhitao Ying, and Jure Leskovec. "Inductive representation learning on large graphs." _Advances in neural information processing systems_ 30 (2017).            
[^Perozzi et al., 2014]: 25. B. Perozzi, R. Al-Rfou, and S. Skiena. Deepwalk: Online learning of social representations. In KDD, 2014.


# The global outline for paper draft "Hierarchical Message Passing for Quantum Chemistry".

1.  **Introduction**
    *   引出GNN和MPNN的成功，及其在长程交互上的局限性（Over-squashing）。
    *   简要介绍Master Node和RBM-1等相关思想的启发性。
    *   提出我们的核心思想：通过可学习的多主节点和虚拟边构建分层信息网络。
    *   总结我们的贡献：1) 提出HMP-Net架构；2) 设计了可微的节点选择和拓扑学习机制；3) 在多个基准上验证了其有效性。
2.  **Related Work**
    *   图神经网络与消息传递。
    *   Over-squashing问题及其现有解决方案（图重写、全局池化、Graph Transformer等）。
    *   随机方法在科学计算中的应用（RBM-1）。
3.  **Proposed Method: Hierarchical Message Passing Network (HMP-Net)**
    *   3.1 整体框架概述（配架构图）。
    *   3.2 Master Node Selection Mechanism（详细描述Gumbel-Softmax选择过程和公式）。
    *   3.3 Virtual Edge Generation（详细描述可学习邻接矩阵的生成）。
    *   3.4 Hierarchical Message Passing on the Final Graph（描述异构图消息传递流程）。
    *   3.5 Loss Function Design（详细定义L_task, L_sparsity等）。
4.  **Experimental Settings**
    *   4.1 Datasets and Tasks。
    *   4.2 Baseline Models。
    *   4.3 Implementation Details and Hyperparameters。
    *   4.4 Evaluation Metrics。
5.  **Results and Discussions**
    *   5.1 Main Results（展示各模型在各数据集上的性能对比表）。
    *   5.2 Efficiency Analysis（参数量、速度对比）。
    *   5.3 Over-squashing Analysis（k-chains上的梯度范数和精度对比）。
6.  **Ablation Study and Mechanism Analysis**
    *   6.1 Impact of the Number of Master Nodes。
    *   6.2 Analysis of Regularization Terms。
    *   6.3 Visualization of Master Nodes and Virtual Edges。
7.  **Conclusion and Future Work**
    *   总结我们的方法和发现。
    *   讨论局限性，并展望未来方向（动态调整机制、与Transformer结合、强化学习优化等）。
