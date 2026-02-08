# Fine-Grained and Lightweight OSA Detection: A CRNN-Based Model for Precise Temporal Localization of Respiratory Events in Sleep Audio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Paper Status](https://img.shields.io/badge/Status-Under_Review-blue.svg)]()

This repository contains the official PyTorch implementation of the paper: **"Fine-Grained and Lightweight OSA Detection: A CRNN-Based Model for Precise Temporal Localization of Respiratory Events in Sleep Audio"**.

## 📖 Abstract

Obstructive Sleep Apnea (OSA) is a prevalent chronic disorder, yet traditional diagnosis via Polysomnography (PSG) is resource-intensive and scarce. Audio-based screening offers a scalable alternative but often lacks the granularity required for precise event localization.

In this work, we propose a **Fine-Grained and Lightweight Dual-Stream CRNN** framework. By integrating Log Mel-spectrograms with an auxiliary Energy Profile branch, our model captures both spectral characteristics and fine-grained physiological intensity dynamics. The model is designed for edge deployment, achieving high clinical precision with a compact parameter footprint and low latency.

**Key Features:**
- **Dual-Stream Architecture:** Combines VGG-style CNN (with SE-Block) and a dedicated Energy Branch to enhance Hypopnea detection.
- **Fine-Grained Localization:** Achieves precise frame-level detection (IoU = 0.82) and accurate event boundary regression.
- **Lightweight & Efficient:** Only **1.73M parameters** with an inference latency of **11.60 ms/min** on GPU, suitable for real-time edge computing.
- **Robust Generalization:** Validated on two independent cohorts (Sismanoglio & Beijing Tongren Hospital) with strong AHI correlation ($r=0.96$).

## 🏗️ Model Architecture

The proposed architecture consists of four key components:
1.  **CNN Feature Extractor:** A 4-layer VGG-style CNN with Squeeze-and-Excitation (SE) blocks to extract spectral features.
2.  **Energy Branch:** Explicitly encodes signal intensity dynamics to resolve acoustic ambiguity in hypopnea events.
3.  **Temporal Modeling:** A Bidirectional LSTM (BiLSTM) captures long-term temporal dependencies.
4.  **Hybrid Loss:** Optimized using a combination of Weighted Focal Loss and Dice Loss to handle class imbalance and ensure event continuity.

<div align="center">
  <img src="assets/architecture.tif" alt="Dual-Stream CRNN Architecture" width="800"/>
  <br>
  <em>Figure 1: Schematic illustration of the proposed Dual-Stream CRNN architecture.</em>
</div>

## 📂 Datasets

This study utilizes two datasets:

1.  **Internal Dataset (Sismanoglio Cohort):**
    * **Source:** Sismanoglio-Amalia Fleming General Hospital.
    * **Subjects:** 286 full-night recordings.
    * **Access:** This dataset is publicly available. Please refer to [Reference 22 in Paper] or contact the original authors for access.

2.  **External Dataset (Beijing Tongren Hospital Cohort):**
    * **Source:** Beijing Tongren Hospital.
    * **Subjects:** 60 subjects (Balanced severity distribution).
    * **Access:** Due to privacy regulations and ethical restrictions, this dataset is **not publicly available**.

## 🛠️ Installation

### Prerequisites
* Python >= 3.8
* PyTorch >= 1.12
* NVIDIA GPU (Tested on RTX 5070 Ti)

### Setup
```bash
git clone [https://github.com/yourusername/Fine-Grained-OSA-CRNN.git](https://github.com/yourusername/Fine-Grained-OSA-CRNN.git)
cd Fine-Grained-OSA-CRNN
pip install -r requirements.txt
