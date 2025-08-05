# HMP-Net Code Audit and Action Items

This document presents a detailed audit of the `hmp-net` codebase against the provided paper draft (*Hierarchical Message Passing for Quantum Chemistry*). It also includes a prioritized list of action items to align the implementation with the paper's specifications.

## 1. Code-Paper Consistency Audit

The following is a section-by-section analysis of the code's alignment with the paper draft.

---

### **Section 3: Proposed Method**

#### **3.1 Framework Overview**
- **Status:** ✅ **Aligned**
- **Code Locations:**
  - `models/hmp/egnn_hmp.py`: `HMPLayer.forward()` (lines 24-74)
  - `models/hmp/mace_hmp.py`: `HMP_MACELayer.forward()` (lines 26-92)
- **Comments:** The high-level 3-stage process (Local Propagation, Master Node Selection, Virtual Edge Generation) is correctly implemented in the `forward` methods of the HMP layer wrappers.

#### **3.2 Master Node Selection (Gumbel-Softmax)**
- **Status:** ⚠️ **Partially Aligned**
- **Code Location:** `models/hmp/master_selection.py`: `MasterSelection` class.
- **Discrepancies:**
  - **Missing `tau` Annealing:** The paper specifies that the Gumbel-Softmax temperature `tau` should be annealed during training. The current implementation uses a fixed `tau` value set at initialization.
  - **Distribution Mismatch:** The paper mentions using `softmax` for selection probabilities, implying a categorical distribution over all nodes. The code implements `sigmoid`, treating each node's selection as an independent Bernoulli trial.

#### **3.3 Virtual Edge Generation (attention + sparsemax)**
- **Status:** ✅ **Aligned**
- **Code Location:** `models/hmp/virtual_generation.py`: `VirtualGeneration` class.
- **Comments:** The implementation of attention-based edge generation using scalar features, `LeakyReLU`, and the `sparsemax` function is a faithful representation of the paper's description.

#### **3.4 Hierarchical Message Passing**
- **Status:** ✅ **Aligned**
- **Code Location:** `models/hmp/egnn_hmp.py`: `HMPLayer.forward()`.
- **Comments:** The code correctly implements the four-step process of wrapping a backbone layer, applying it locally and hierarchically, and aggregating the features based on the master node mask.

#### **3.5 Loss Function**
- **Status:** ❌ **Major Discrepancy**
- **Code Locations:**
  - `experiments/utils/train_utils.py`: `train()` function (line 31).
  - `models/hmp/egnn_hmp.py`, `models/hmp/mace_hmp.py`: `forward()` methods.
- **Discrepancies:**
  - **Missing Regularization Terms:** The composite loss function `L = L_task + λ_struct * L_structure + λ_rate * L_rate` is **not implemented**. The training loop only optimizes for `L_task` (`CrossEntropyLoss`). The model `forward` methods were explicitly modified (as per instructions) to return only prediction tensors, preventing the regularization terms from being used. This is the most critical deviation from the paper.

---

### **Section 4: Experimental Settings**

#### **4.1 Datasets & Tasks**
- **Status:** ✅ **Aligned**
- **Code Location:** `experiments/kchains.ipynb`.
- **Comments:** The notebook correctly sets up the `k-chains` dataset for a binary classification task.

#### **4.2 Backbone / Baselines**
- **Status:** ✅ **Aligned**
- **Code Locations:** `models/hmp/egnn_hmp.py`, `models/hmp/mace_hmp.py`.
- **Comments:** The HMP models correctly wrap their respective backbone architectures (EGNN and MACE).

#### **4.3 Hyper-parameters**
- **Status:** ❌ **Major Discrepancy**
- **Code Locations:**
  - `experiments/utils/train_utils.py`: `_run_experiment()` function (lines 53-55).
  - `experiments/kchains.ipynb`: New cell for HMP models.
- **Discrepancies:**
  - **Optimizer Mismatch:** The code uses `torch.optim.Adam`, whereas the paper specifies `AdamW`.
  - **LR Scheduler Mismatch:** The code uses `ReduceLROnPlateau`, whereas the paper specifies a `cosine annealing` schedule.
  - **Unused Regularization Hyperparameters:** `λ_struct = 0.01` and `λ_rate = 0.1` are specified in the paper but have no effect in the code due to the incomplete loss function.
  - **Missing `tau` Annealing:** As noted in 3.2, the annealing schedule for the Gumbel-Softmax temperature is missing.

#### **4.4 Evaluation Metrics**
- **Status:** ✅ **Aligned**
- **Code Location:** `experiments/utils/train_utils.py`: `eval()` function.
- **Comments:** The `eval()` function correctly computes classification accuracy for the `k-chains` task.

---

## 2. Action-Item Report

Based on the audit, the following changes are required to align the codebase with the paper.

### **High Priority (Accuracy-Critical)**
- [ ] **Implement Composite Loss:** Modify the training loop (or create a new HMP-specific one) to incorporate the full loss function: `L = L_task + λ_struct * L_structure + λ_rate * L_rate`. This requires changing the model's `forward` method to return the regularization terms alongside the prediction.
- [ ] **Implement `tau` Annealing:** Add a mechanism to anneal the Gumbel-Softmax temperature `tau` over the course of training, as described in the paper. This should be part of the training loop.

### **Medium Priority (Correctness / Completeness)**
- [ ] **Change Optimizer to AdamW:** Replace `torch.optim.Adam` with `torch.optim.AdamW` in `train_utils.py` to match the paper's experimental setup.
- [ ] **Implement Cosine LR Scheduler:** Replace `ReduceLROnPlateau` with a cosine annealing learning rate scheduler.

### **Low Priority (Style / Documentation)**
- [ ] **Clarify Node Selection Distribution:** Add a code comment to `master_selection.py` clarifying why `sigmoid` (independent Bernoulli) was used instead of `softmax` (Categorical), as mentioned in the paper.
