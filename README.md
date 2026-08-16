# PCA-Based Disentanglement Evaluation for Rotation-Invariant DLVMs

[![ICASSP 2025](https://img.shields.io/badge/ICASSP%202025-Published-4B0082?style=for-the-badge)](https://ieeexplore.ieee.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.15705-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2501.15705)

**Evaluation library for "Disentanglement Analysis in Deep Latent Variable Models Matching Aggregate Posterior Distributions"**  
by Surojit Saha, Sarang Joshi, and Ross Whitaker (ICASSP 2025)

> **⚠️ Important:** This is an **evaluation-only library**. It requires **pre-trained model checkpoints** to compute disentanglement metrics. This repository does not include training code.

---

## Table of Contents
- [Motivation](#motivation)
- [Key Contributions](#key-contributions)
- [Relationship to Prior Work](#relationship-to-prior-work)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Single Metric Evaluation](#single-metric-evaluation)
  - [Full Paper Reproduction](#full-paper-reproduction)
  - [Aggregating Results](#aggregating-results)
- [Metrics Explained](#metrics-explained)
- [Repository Structure](#repository-structure)
- [Results from the Paper](#results-from-the-paper)
- [Extending to New Models](#extending-to-new-models)
- [Citation](#citation)

---

## Motivation

### The Rotation Invariance Problem

Standard disentanglement metrics (FactorVAE, MIG, DCI) assume that **latent axes (cardinal directions) align with ground truth factors**. This assumption holds for VAEs because they use a **factorized Gaussian posterior** $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\mathbf{x}, \boldsymbol{\sigma}_\mathbf{x}^2\mathbf{I})$ that encourages axis-aligned representations.

However, several DLVMs match the **aggregate posterior** $q_\phi(\mathbf{z})$ to an **isotropic Gaussian prior** $\mathcal{N}(\mathbf{0}, \mathbf{I})$:
- **AVAE** (Aggregate Variational Autoencoder)
- **AAE** (Adversarial Autoencoder)
- **WAE-MMD** (Wasserstein Autoencoder)
- **GENs** (Generative Encoding Networks)

Because $\mathcal{N}(\mathbf{0}, \mathbf{I})$ is **rotation-invariant**, there is no preference for cardinal axes. The learned representations can be **rotated** without affecting reconstruction quality, meaning:

$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \quad \Rightarrow \quad \text{rot}(\mathbf{z}) \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**Consequence:** Ground truth factors may align with **principal directions** discovered by the model, **not necessarily the cardinal latent axes**. Standard metrics that rely on axis-aligned factors will **incorrectly penalize** these models.

### Our Solution: PCA-Based Latent Directions

Instead of evaluating disentanglement along cardinal axes $[\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_l]$, we:

1. **Discover latent directions** $\mathcal{D} = [\mathbf{u}_1^\*, \mathbf{u}_2^\*, \dots, \mathbf{u}_k^\*]$ using PCA on encoded representations
2. Fix one ground truth factor at a time and vary others
3. Apply PCA to find the direction of minimal variance (the fixed factor)
4. Aggregate multiple PCA estimates via eigendecomposition of mean outer products
5. Evaluate disentanglement using these **discovered directions** instead of cardinal axes

This provides a **generalized framework** for evaluating any DLVM, especially those matching aggregate posteriors.

---

## Key Contributions

1. **PCA FactorVAE Score** — Adapts the FactorVAE metric to use PCA-discovered latent directions instead of cardinal axes
2. **PCA MIG Score** — Adapts Mutual Information Gap to use projected representations on discovered directions
3. **Generalized Evaluation** — Works for **any DLVM**, not just VAEs (tested on VAE, β-TCVAE, DIP-VAE, AAE, WAE-MMD, AVAE)
4. **Significant Improvements** — Shows substantial metric gains for aggregate-posterior-matching methods (AVAE: +43.23% FactorVAE on DSprites, +0.47 MIG on Shapes3D)

---

## Relationship to Prior Work

### Extension of `disentanglement_lib`

This repository is conceptually an **extension** of [google-research/disentanglement_lib](https://github.com/google-research/disentanglement_lib), which provides:
- Comprehensive comparison of VAE-based disentanglement methods
- Multiple metrics (FactorVAE, MIG, SAP, DCI, IRS, etc.)
- Standardized evaluation protocols

**Key Difference:**  
`disentanglement_lib` assumes **axis-aligned factors** (cardinal directions). Our work introduces **PCA-based metrics** for models where this assumption breaks due to rotation invariance.

### Complementary to AVAE

The [AVAE repository](https://github.com/Surojit-Utah/AVAE) provides:
- **Training code** for the Aggregate Variational Autoencoder
- Generative quality metrics (FID, Precision-Recall, Entropy)
- Reconstruction and latent-space analysis

**This repository provides:**
- **Evaluation-only pipeline** for disentanglement metrics
- Requires trained checkpoints from AVAE (or other DLVMs)
- PCA-based adaptations of standard metrics

---

## Installation

### Requirements

- Python 3.8+
- TensorFlow 2.8.0
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory (for model loading and evaluation)

### Install Dependencies

```bash
cd PCA_Disentanglement
pip install -r requirements.txt
```

### Required Packages

```
tensorflow==2.8.0
scikit-learn>=1.0.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
tqdm>=4.62.0
nvidia-ml-py3>=7.352.0
```

---

## Dataset Setup

The evaluation uses **synthetic datasets with known ground truth factors**. This allows quantitative measurement of disentanglement quality.

### 1. Download Datasets

#### DSprites (737,280 images, 64×64×1)
```bash
wget https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
```

**6 Ground Truth Factors:**
- Color (1 value: white)
- Shape (3 values: square, ellipse, heart)
- Scale (6 values)
- Orientation (40 values)
- Position X (32 values)
- Position Y (32 values)

#### 3D Shapes (480,000 images, 64×64×3)
Download from: https://console.cloud.google.com/storage/browser/3d-shapes  
File: `3dshapes.h5` (convert to `imgs_train.npz` format)

**6 Ground Truth Factors:**
- Floor Hue (10 values)
- Wall Hue (10 values)
- Object Hue (10 values)
- Scale (8 values)
- Shape (4 values: cube, cylinder, sphere, capsule)
- Orientation (15 values)

### 2. Set Environment Variables

Add to your `.bashrc` / `.zshrc` (Linux/Mac):
```bash
export DSPRITES_DATA_DIR="/path/to/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
export SHAPES3D_DATA_DIR="/path/to/imgs_train.npz"
```

Or on Windows (PowerShell):
```powershell
$env:DSPRITES_DATA_DIR="C:\path\to\dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
$env:SHAPES3D_DATA_DIR="C:\path\to\imgs_train.npz"
```

**Important:** The scripts will raise an error if these environment variables are not set.

---

## Configuration

### Update Model Checkpoint Paths

Edit `config/local_config.py` to point to your trained model checkpoints:

```python
configurations = {
    0: {  # DSprites
        'dataset_name': 'DSprites',
        'latent_dim': 6,
        'num_filter': 64,
        
        # UPDATE THIS PATH to your trained models
        'model_checkpoint_dir': '/path/to/logs/DSprites/Run_{run_id}/Models/best_model',
        
        'use_whiten_data': False,
        'num_train_data': 10000,
        'num_eval_data': 5000,
    },
    1: {  # Shapes3D
        'dataset_name': 'Shapes3D',
        'latent_dim': 6,
        'num_filter': 64,
        
        # UPDATE THIS PATH to your trained models
        'model_checkpoint_dir': '/path/to/logs/Shapes3D/Run_{run_id}/Models/best_model',
        
        'use_whiten_data': False,
        'num_train_data': 10000,
        'num_eval_data': 5000,
    },
}

# Run IDs for multiple independent evaluations (paper uses 10 runs)
EVAL_RUN_IDS = list(range(1, 11))  # [1, 2, 3, ..., 10]
```

### Checkpoint Structure Expected

The evaluation scripts expect TensorFlow checkpoints in this structure:
```
logs/
├── DSprites/
│   ├── Run_1/
│   │   └── Models/
│   │       └── best_model/
│   │           ├── checkpoint
│   │           ├── ckpt.data-00000-of-00001
│   │           └── ckpt.index
│   ├── Run_2/
│   └── ...
└── Shapes3D/
    ├── Run_1/
    └── ...
```

**How to Train Models:**  
This repository does not include training code. To obtain trained models:
- **For AVAE models:** Use the [AVAE repository](https://github.com/Surojit-Utah/AVAE) to train AVAE models on DSprites and Shapes3D datasets. The paper results use AVAE-trained checkpoints from this repository.
- **For other DLVMs:** Train your own VAE/β-TCVAE/DIP-VAE/AAE/WAE implementations
- Ensure encoder/decoder architectures match the expected format (see `models/` directory)

---

## Usage

### Single Metric Evaluation

#### Compute PCA FactorVAE Score

```bash
cd PCA_Disentanglement

# DSprites, Run 1
python eval/compute_metrics.py --config_id 0 --metric factor_pca_axis --seed 1

# Shapes3D, Run 1
python eval/compute_metrics.py --config_id 1 --metric factor_pca_axis --seed 1
```

**Output:**
```
Output/DSprites/Factor_PCA/No_Whiten/Run_1/
├── output.txt              # Training accuracy: X%, Test accuracy: Y%
└── mean_angle_between_pca_axis.png  # Orthogonality visualization
```

#### Compute PCA MIG Score

```bash
# DSprites, Run 1
python eval/compute_metrics.py --config_id 0 --metric mig_pca_axis --seed 1

# Shapes3D, Run 1
python eval/compute_metrics.py --config_id 1 --metric mig_pca_axis --seed 1
```

**Output:**
```
Output/DSprites/MIG_PCA/No_Whiten/Run_1/
├── output.txt              # MIG Score: 0.XXXX
└── mutual_entropy.png      # Mutual information visualization
```

#### Compute MSE Reconstruction Error

```bash
# DSprites, Run 1
python eval/compute_mse.py --config_id 0 --seed 1

# Shapes3D, Run 1
python eval/compute_mse.py --config_id 1 --seed 1
```

**Output:**
```
Output/DSprites/
├── mse_run_1.txt           # Run 1 MSE: X.XXXX
└── recons_example_run_1.png  # Example reconstructions
```

### Full Paper Reproduction

To reproduce all results from Table 1 of the paper (10 runs per dataset):

#### Automated Scripts

**Linux/Mac:**
```bash
cd PCA_Disentanglement

# Run all metrics for DSprites (10 runs)
bash scripts/reproduce_paper_results.sh --dataset dsprites --num-seeds 10

# Run all metrics for Shapes3D (10 runs)
bash scripts/reproduce_paper_results.sh --dataset shapes3d --num-seeds 10

# Or run both
bash scripts/reproduce_paper_results.sh
```

**Windows (PowerShell):**
```powershell
cd PCA_Disentanglement

# Run all metrics for DSprites (10 runs)
.\scripts\reproduce_paper_results.ps1 -Dataset dsprites -NumSeeds 10

# Run all metrics for Shapes3D (10 runs)
.\scripts\reproduce_paper_results.ps1 -Dataset shapes3d -NumSeeds 10

# Or run both
.\scripts\reproduce_paper_results.ps1
```

#### Manual Execution

For more control, run metrics manually for each seed:

```bash
# DSprites (seeds 1-10)
for seed in {1..10}; do
    python eval/compute_metrics.py --config_id 0 --metric factor_pca_axis --seed $seed
    python eval/compute_metrics.py --config_id 0 --metric mig_pca_axis --seed $seed
    python eval/compute_mse.py --config_id 0 --seed $seed
done

# Shapes3D (seeds 1-10)
for seed in {1..10}; do
    python eval/compute_metrics.py --config_id 1 --metric factor_pca_axis --seed $seed
    python eval/compute_metrics.py --config_id 1 --metric mig_pca_axis --seed $seed
    python eval/compute_mse.py --config_id 1 --seed $seed
done
```

### Aggregating Results

After running all evaluations, compute **mean ± std** statistics:

```bash
# DSprites aggregate statistics
python aggregate_results.py --config_id 0

# Shapes3D aggregate statistics
python aggregate_results.py --config_id 1
```

**Output:**
```
============================================================
Aggregating Results for DSprites
============================================================

FactorVAE Score (DSprites):
  Runs: 10
  Mean: 79.17%
  Std:  1.64%
  Result: 79.17 ± 1.64

MIG Score (DSprites):
  Runs: 10
  Mean: 0.2000
  Std:  0.0200
  Result: 0.20 ± 0.02

MSE Score (DSprites):
  Runs: 10
  Mean: 2.9800
  Std:  0.2800
  Result: 2.98 ± 0.28

============================================================
✓ Aggregation complete!
============================================================
```

Aggregated statistics are saved to:
```
Output/DSprites/Factor_PCA/No_Whiten/factor_pca_axis_metric_stat.txt
Output/DSprites/MIG_PCA/No_Whiten/mig_pca_axis_metric_stat.txt
Output/DSprites/mse_error.txt
```

---

## Metrics Explained

### 1. PCA FactorVAE Score (Higher is Better)

**What it measures:** Classification accuracy of ground truth factors using PCA-discovered latent directions.

**Algorithm:**
1. For each ground truth factor $k$:
   - Fix factor $k$ to a specific value
   - Vary all other factors
   - Encode $L$ samples → latent representations $\mathcal{Z}^i$
   - Apply PCA → eigenvector with **minimum variance** = direction for factor $k$
2. Repeat $N$ times with different fixed values
3. Aggregate via eigendecomposition of mean outer product → $\mathbf{u}_k^\*$ (final direction)
4. Use **cosine similarity** between test samples and $\{\mathbf{u}_k^\*\}$ to classify factors
5. Report classification accuracy

**Why it works:** Fixing one factor while varying others isolates variance orthogonal to that factor. The minimum-variance PCA direction captures the fixed factor.

**Standard FactorVAE vs. PCA FactorVAE:**
- **Standard:** Uses cardinal axes $[\mathbf{e}_1, \dots, \mathbf{e}_l]$
- **PCA-based:** Uses discovered directions $[\mathbf{u}_1^\*, \dots, \mathbf{u}_k^\*]$

### 2. PCA MIG (Mutual Information Gap) (Higher is Better)

**What it measures:** How much information latent variables carry about ground truth factors.

**Algorithm:**
1. Discover latent directions $\mathcal{D} = [\mathbf{u}_1^\*, \dots, \mathbf{u}_k^\*]$ using PCA
2. Encode dataset → latent representations $\mathcal{Z}$
3. **Project** representations onto discovered directions: $\mathcal{Z}' = \mathcal{Z} \mathcal{D}^T$
4. Discretize projected latents and ground truth factors
5. Compute mutual information $I(\mathcal{Z}'_j; \mathcal{F}_k)$ for each pair
6. For each factor $k$, compute gap between top-2 mutual information values
7. Average gaps across all factors

**Standard MIG vs. PCA MIG:**
- **Standard:** Uses cardinal axes $\mathbf{z}_j$ directly
- **PCA-based:** Uses projected coordinates $\mathbf{z}'_j = \mathbf{z} \cdot \mathbf{u}_j^\*$

### 3. MSE (Mean Squared Error) (Lower is Better)

**What it measures:** Reconstruction quality (per-pixel squared error).

**Algorithm:**
1. Encode test images → latent representations
2. Decode latent representations → reconstructed images
3. Compute $\text{MSE} = \frac{1}{N \cdot d} \sum_{i=1}^{N} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2$

**Note:** MSE is **not PCA-based**. It measures reconstruction fidelity, independent of latent structure.

---

## Repository Structure

```
PCA_Disentanglement/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore patterns
│
├── config/
│   └── local_config.py          # Dataset and model configurations
│
├── data/
│   ├── __init__.py
│   ├── dsprites_loader.py       # DSprites dataset loader
│   └── shapes3d_loader.py       # 3D Shapes dataset loader
│
├── samples/                      # Data samplers for metrics
│   ├── __init__.py
│   ├── factor_pca/
│   │   ├── __init__.py
│   │   ├── factor_pca_sample_data_dsprites.py
│   │   └── factor_pca_sample_data_shapes3d.py
│   ├── mig_pca/
│   │   ├── __init__.py
│   │   ├── mig_pca_sample_data_dsprites.py
│   │   └── mig_pca_sample_data_shapes3d.py
│   └── mse/
│       ├── __init__.py
│       ├── sample_data_dsprites.py
│       └── sample_data_shapes3d.py
│
├── models/                       # Encoder/Decoder architectures
│   ├── __init__.py
│   ├── ae_model_dsprites.py     # DSprites architecture
│   └── ae_model_shapes3d.py     # Shapes3D architecture
│
├── metrics/                      # PCA-based metric implementations
│   ├── __init__.py
│   ├── factor_pca.py            # PCA FactorVAE score
│   └── mig_pca.py               # PCA MIG score
│
├── utils/
│   ├── __init__.py
│   └── gpu_utils.py             # GPU selection and seed setting
│
├── visualization/
│   ├── __init__.py
│   └── plot_disentanglement_traversal.py  # Latent traversal plots
│
├── eval/                         # Evaluation scripts
│   ├── compute_metrics.py       # FactorVAE and MIG computation
│   └── compute_mse.py           # MSE computation
│
├── scripts/                      # Automation scripts
│   ├── reproduce_paper_results.sh   # Bash automation
│   └── reproduce_paper_results.ps1  # PowerShell automation
│
├── aggregate_results.py          # Aggregate statistics (mean ± std)
│
└── Output/                       # Results directory (auto-created)
    ├── DSprites/
    │   ├── Factor_PCA/
    │   │   └── No_Whiten/
    │   │       ├── Run_1/, Run_2/, ...
    │   │       └── factor_pca_axis_metric_stat.txt
    │   ├── MIG_PCA/
    │   │   └── No_Whiten/
    │   │       ├── Run_1/, Run_2/, ...
    │   │       └── mig_pca_axis_metric_stat.txt
    │   └── mse_run_1.txt, mse_run_2.txt, ...
    └── Shapes3D/
        └── (same structure)
```

---

## Results from the Paper

### Table 1: Disentanglement Scores (Mean ± Std, 10 runs)

#### DSprites Dataset

| Method | FactorVAE | PCA FactorVAE | Δ | MIG | PCA MIG | Δ | MSE ↓ |
|--------|-----------|---------------|---|-----|---------|---|-------|
| **VAE** | 64.78 ± 8.05 | 75.56 ± 7.21 | <span style="color:blue">+10.78</span> | 0.06 ± 0.02 | 0.14 ± 0.04 | <span style="color:blue">+0.08</span> | 3.68 ± 0.58 |
| **β-TCVAE** | **75.55 ± 3.52** | 69.12 ± 15.08 | <span style="color:red">-6.43</span> | **0.20 ± 0.06** | 0.18 ± 0.13 | <span style="color:red">-0.02</span> | 6.39 ± 2.05 |
| **DIP-VAE-I** | 61.77 ± 8.96 | 70.68 ± 6.89 | <span style="color:blue">+8.91</span> | 0.13 ± 0.07 | 0.12 ± 0.06 | <span style="color:red">-0.01</span> | 3.61 ± 0.47 |
| **AAE** | 22.18 ± 4.37 | 60.67 ± 7.44 | <span style="color:blue">+38.49</span> | 0.01 ± 0.01 | 0.07 ± 0.02 | <span style="color:blue">+0.06</span> | **2.62 ± 0.05** |
| **WAE-MMD** | 17.82 ± 0.52 | 58.87 ± 9.00 | <span style="color:blue">+41.05</span> | 0.01 ± 0.00 | 0.07 ± 0.02 | <span style="color:blue">+0.06</span> | 2.98 ± 0.17 |
| **AVAE** | 35.94 ± 5.72 | **79.17 ± 1.64** | <span style="color:blue">**+43.23**</span> | 0.02 ± 0.01 | **0.20 ± 0.02** | <span style="color:blue">**+0.18**</span> | 2.98 ± 0.28 |

#### Shapes3D Dataset

| Method | FactorVAE | PCA FactorVAE | Δ | MIG | PCA MIG | Δ | MSE ↓ |
|--------|-----------|---------------|---|-----|---------|---|-------|
| **VAE** | 78.57 ± 3.33 | 81.92 ± 2.87 | <span style="color:blue">+3.35</span> | 0.38 ± 0.02 | 0.38 ± 0.02 | 0.00 | 21.09 ± 1.67 |
| **β-TCVAE** | **86.49 ± 2.96** | 84.93 ± 5.67 | <span style="color:red">-1.56</span> | **0.53 ± 0.04** | 0.52 ± 0.07 | <span style="color:red">-0.01</span> | 20.37 ± 1.49 |
| **DIP-VAE-I** | 81.11 ± 3.01 | 80.32 ± 4.62 | <span style="color:red">-0.79</span> | 0.44 ± 0.03 | 0.42 ± 0.04 | <span style="color:red">-0.02</span> | 13.74 ± 0.88 |
| **AAE** | 55.22 ± 13.39 | 82.40 ± 2.64 | <span style="color:blue">+27.18</span> | 0.15 ± 0.07 | 0.44 ± 0.03 | <span style="color:blue">+0.29</span> | **10.11 ± 0.21** |
| **WAE-MMD** | 43.93 ± 8.88 | 81.79 ± 2.87 | <span style="color:blue">+37.86</span> | 0.10 ± 0.04 | 0.43 ± 0.03 | <span style="color:blue">+0.33</span> | 10.85 ± 0.39 |
| **AVAE** | 56.46 ± 7.91 | **91.93 ± 3.27** | <span style="color:blue">**+35.47**</span> | 0.20 ± 0.06 | **0.67 ± 0.04** | <span style="color:blue">**+0.47**</span> | 10.29 ± 0.37 |

### Key Observations

1. **VAE-based methods (VAE, β-TCVAE, DIP-VAE)** show **small changes** (±10%) — their axis-aligned assumption is mostly valid
2. **Aggregate-posterior-matching methods (AAE, WAE-MMD, AVAE)** show **massive improvements** (+35-43% FactorVAE, +0.18-0.47 MIG)
3. **AVAE achieves best scores** on both datasets with PCA-based metrics, despite poor standard metric scores
4. **Rotation invariance matters** — methods matching $\mathcal{N}(\mathbf{0}, \mathbf{I})$ learn rotated representations

---

## Extending to New Models

To evaluate a new DLVM (trained elsewhere):

### 1. Match Encoder/Decoder Architecture

Ensure your model's encoder/decoder match the expected interface:

```python
# models/ae_model_custom.py

class Encoder(tf.keras.Model):
    def __init__(self, latent_dim, num_filter):
        super().__init__()
        # Your architecture here
    
    def call(self, inputs, use_batch_norm=False, training=False):
        # Must return: (batch_size, latent_dim)
        return latent_representation

class Decoder(tf.keras.Model):
    def __init__(self, latent_dim, num_filter):
        super().__init__()
        # Your architecture here
    
    def call(self, inputs, use_batch_norm=False, training=False):
        # Must return: (batch_size, H, W, C)
        return reconstructed_images
```

### 2. Add Configuration

Edit `config/local_config.py`:

```python
configurations = {
    # ... existing configs ...
    
    2: {  # Your custom model
        'model_name': 'CustomDLVM',
        'dataset_name': 'DSprites',  # or 'Shapes3D'
        'latent_dim': 6,
        'num_filter': 64,
        'model_checkpoint_dir': '/path/to/custom/model/Run_{run_id}/ckpt',
        'use_whiten_data': False,
        'num_train_data': 10000,
        'num_eval_data': 5000,
        'batch_size': 100,
    },
}
```

### 3. Update Evaluation Scripts

If using custom architecture names, update imports in `eval/compute_metrics.py` and `eval/compute_mse.py`:

```python
from models import ae_model_custom

# In main section:
elif dataset_name == 'DSprites' and model_name == 'CustomDLVM':
    encoder = ae_model_custom.Encoder(latent_dim=latent_dim, num_filter=num_filter)
    decoder = ae_model_custom.Decoder(latent_dim=latent_dim, num_filter=num_filter)
```

### 4. Run Evaluation

```bash
python eval/compute_metrics.py --config_id 2 --metric factor_pca_axis --seed 1
python aggregate_results.py --config_id 2
```

---

## Citation

If you use this evaluation library or method, please cite:

```bibtex
@inproceedings{saha2024disentanglement,
  title     = {Disentanglement Analysis in Deep Latent Variable Models Matching Aggregate Posterior Distributions},
  author    = {Saha, Surojit and Joshi, Sarang and Whitaker, Ross},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2024},
  organization = {IEEE},
  url       = {https://arxiv.org/pdf/2501.15705}
}
```

This work builds upon the AVAE, which should also be cited:

```bibtex
@inproceedings{saha2024avae,
  title     = {Matching Aggregate Posteriors in the Variational Autoencoder},
  author    = {Saha, Surojit and Joshi, Sarang and Whitaker, Ross},
  booktitle = {International Conference on Pattern Recognition (ICPR)},
  year      = {2024},
  url       = {https://arxiv.org/pdf/2311.07693}
}
```

And acknowledges the foundational work in disentanglement evaluation:

```bibtex
@inproceedings{kim2018disentangling,
  title     = {Disentangling by Factorising},
  author    = {Kim, Hyunjik and Mnih, Andriy},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2018},
  url       = {https://arxiv.org/abs/1811.12359}
}

@inproceedings{locatello2019challenging,
  title     = {Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations},
  author    = {Locatello, Francesco and Bauer, Stefan and Lucic, Mario and others},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2019}
}
```

---

## Contact

**Authors:**  
Surojit Saha, Sarang Joshi, Ross Whitaker  
Scientific Computing and Imaging Institute, University of Utah

**Correspondence:**  
surojit.saha@utah.edu

**Related Repositories:**
- [AVAE (Training)](https://github.com/Surojit-Utah/AVAE) — Train aggregate-posterior-matching models
- [google-research/disentanglement_lib](https://github.com/google-research/disentanglement_lib) — Comprehensive VAE evaluation suite

---

## License

This code is released for academic and research purposes. Please cite the paper if you use this code in your work.

---

## Troubleshooting

### Common Issues

**1. Environment variables not set:**
```
ValueError: DSPRITES_DATA_DIR environment variable not set
```
**Solution:** Set environment variables (see [Dataset Setup](#dataset-setup))

**2. Checkpoint not found:**
```
Could not find checkpoint directory: /path/to/logs/DSprites/Run_1/Models/best_model
```
**Solution:** Update `model_checkpoint_dir` in `config/local_config.py`

**3. Import errors:**
```
ModuleNotFoundError: No module named 'samples'
```
**Solution:** Scripts in `eval/` add parent directory to `sys.path` automatically. Run from repo root:
```bash
cd PCA_Disentanglement
python eval/compute_metrics.py --config_id 0 --metric factor_pca_axis --seed 1
```

**4. GPU out of memory:**
```
ResourceExhaustedError: OOM when allocating tensor
```
**Solution:** Reduce `batch_size` or `num_train_data` in `config/local_config.py`

**5. TensorFlow version mismatch:**
```
AttributeError: module 'tensorflow' has no attribute 'train'
```
**Solution:** Ensure TensorFlow 2.8.0 is installed:
```bash
pip install tensorflow==2.8.0
```

---

**Last Updated:** 2026-08-16  
**Version:** 1.0.0
