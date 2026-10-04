# Deep Representation Learning & Computer Vision for Remote Sensing

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Computer Vision](https://img.shields.io/badge/Domain-Remote%20Sensing%20%7C%20Satellite%20Imagery-blue.svg)]()
[![Techniques](https://img.shields.io/badge/Methods-Transfer%20Learning%20%7C%20Feature%20Probing%20%7C%20Few--Shot-green.svg)]()
[![IIT Bombay](https://img.shields.io/badge/IIT%20Bombay-GNR%20638%20Remote%20Sensing-red.svg)](https://www.csre.iitb.ac.in/)

> **Academic Affiliation**: Course Projects for **GNR 638: Machine Learning for Remote Sensing - II**, Centre of Studies in Resources Engineering (CSRE), IIT Bombay  
> **Collaborators**: **Dheeraj Kumar Maradana** ([@dheerajkumar2005](https://github.com/dheerajkumar2005)) & **Lohit** ([@lohit2320](https://github.com/lohit2320))

---

## 📌 Executive Summary

This repository contains CNN implementation in C++ , transfer learning experiments, and representation probing benchmarks

Due to unique spectral characteristics, spatial resolutions, and domain shifts in aerial imagery, standard ImageNet-pretrained representations often exhibit distinct transferability dynamics. This project systematically investigates feature representations across deep convolutional neural networks (e.g., ResNet-50) using linear probing, layer-wise probing, few-shot adaptation, and out-of-distribution robustness assessments.

---

## 🏗️ Repository Structure & Modules

```
Remote-Sensing-Deep-Learning-GNR638/
├── A1/                           # Baseline Classification & Feature Extraction Pipeline
│   ├── src/                      # Model architectures, data loaders, and training utilities
│   ├── train.py                  # End-to-end training harness for satellite image classification
│   ├── test.py                   # Model evaluation, confusion matrices, and precision/recall
│   └── weights/                  # Checkpoint storage
└── A2/                           # Representation Learning & Transferability Probing
    ├── Linear_probe_transfer/    # Linear probing of frozen feature backbones (ResNet-50)
    ├── Layer-wise_Feature_Probing/# Diagnosing representational power across intermediate layers
    ├── Fine_Tuning_Strategies/   # Full fine-tuning vs. backbone-frozen vs. gradual unfreezing
    ├── few_shot/                 # Few-shot learning performance under data-constrained regimes
    └── robust/                   # Robustness evaluation against synthetic corruptions & weather shifts
```

---

## 🔬 Core Methodologies & Experiments

### 0. CNN Implementation in C++ (`A1/src/`)
* Complete CNN implementation in C++ using only std library
### 1. Layer-Wise Feature Probing (`A2/Layer-wise_Feature_Probing`)
* Trains linear classifiers on intermediate feature representations extracted from each residual stage of deep vision backbones.
* Evaluates where geospatial and spectral domain features emerge (early edge detectors vs. late high-level semantic representations).

### 2. Linear Probing & Transfer Dynamics (`A2/Linear_probe_transfer`)
* Measures the out-of-the-box representational transferability of self-supervised and supervised pretrained representations to multi-class remote sensing benchmarks.

### 3. Fine-Tuning Strategies (`A2/Fine_Tuning_Strategies`)
* Compares convergence rates, compute requirements, and downstream accuracy across:
  * Linear probe only (classifier head updated, backbone completely frozen).
  * Partial unfreezing (last $k$ residual blocks fine-tuned with smaller learning rate).
  * Full fine-tuning (discriminative layer-wise learning rates).

### 4. Few-Shot Adaptation (`A2/few_shot`)
* Analyzes sample complexity and classification accuracy under $N$-way $K$-shot conditions ($K \in \{1, 5, 10, 20\}$) to simulate scarce labeled satellite data scenarios.

### 5. Out-of-Distribution Robustness (`A2/robust`)
* Evaluates model sensitivity and calibration under atmospheric perturbations, sensor noise, brightness variations, and contrast degradation.

---

## 🚀 Getting Started

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/dheerajkumar2005/Remote-Sensing-Deep-Learning-GNR638.git
cd Remote-Sensing-Deep-Learning-GNR638

# Install dependencies
pip install torch torchvision torchaudio scikit-learn matplotlib seaborn tqdm
```

### Running Experiments
```bash
# Train baseline classifier in A1
cd A1
python train.py --epochs 30 --batch_size 64

# Run linear probe transfer evaluation in A2
cd ../A2/Linear_probe_transfer
python linear_probe.py
```
