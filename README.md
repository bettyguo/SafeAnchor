# SafeAnchor: Preventing Cumulative Safety Erosion in Continual Domain Adaptation of Large Language Models

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.1%2B-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/code%20style-ruff-black" alt="Ruff">
</p>

> **SafeAnchor** anchors safety alignment in place throughout continual domain adaptation of LLMs. It identifies low-rank safety subspaces via Fisher Information eigendecomposition, constrains domain-specific gradient updates to the orthogonal complement of these subspaces, and monitors for residual safety drift with threshold-triggered corrective replay.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results](#key-results)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Repository Structure](#repository-structure)
6. [Method](#method)
7. [Configuration](#configuration)
8. [Evaluation](#evaluation)
9. [Compute Requirements](#compute-requirements)
10. [Troubleshooting](#troubleshooting)
11. [Contact](#contact)

---

## Overview

Large language models are increasingly fine-tuned sequentially across specialized domains (medicine, law, code generation). While this extends model capabilities, safety guardrails erode **cumulatively** — a critical vulnerability left unaddressed by existing single-task safety-preserving methods.

**SafeAnchor** is the first framework to address *sequential multi-domain* safety preservation. It integrates three complementary components:

| Component | Role |
|-----------|------|
| **Safety Subspace Identification (SSI)** | Identifies LoRA parameter directions encoding safety via Fisher Information eigendecomposition |
| **Orthogonal Safety-Constrained Adaptation (OSCA)** | Projects domain gradient updates orthogonally away from the safety subspace |
| **Cumulative Safety Monitoring (CSM)** | Monitors safety after each domain; triggers corrective replay if degradation exceeds threshold |

---

## Key Results

### Llama-2-7B-Chat (Medical → Legal → Code)

| Method | Safety ↑ | Domain ↑ | MMLU ↑ |
|--------|----------|----------|--------|
| Base (no adapt.) | 91.4 | 38.2 | 46.1 |
| Standard LoRA | 43.6 ± 2.1 | **62.7 ± 0.6** | 44.8 ± 0.3 |
| EWC + LoRA | 52.1 ± 1.8 | 59.4 ± 0.7 | 45.3 ± 0.4 |
| O-LoRA | 48.7 ± 2.3 | 60.1 ± 0.8 | 45.0 ± 0.3 |
| Safe LoRA | 61.3 ± 1.5 | 57.8 ± 0.9 | 44.9 ± 0.4 |
| Vaccine + LoRA | 58.9 ± 1.7 | 58.6 ± 0.8 | 44.5 ± 0.5 |
| SafeGrad + LoRA | 67.4 ± 1.4 | 59.2 ± 0.7 | 45.1 ± 0.3 |
| Safety Interleaving | 64.8 ± 1.6 | 60.9 ± 0.7 | 45.2 ± 0.4 |
| **SafeAnchor** | **85.2 ± 0.9** | <u>61.4 ± 0.5</u> | **45.7 ± 0.3** |

> Safety Score = composite of HarmBench, TruthfulQA, and BBQ (higher is better). All values: mean ± std over 5 seeds.

**SafeAnchor retains 93.2% of original safety alignment** — outperforming the best baseline by 17.8 points — while matching unconstrained fine-tuning to within 1.3 points on domain tasks.

### Mistral-7B-Instruct (Medical → Legal → Code)

| Method | Safety ↑ | Domain ↑ | MMLU ↑ |
|--------|----------|----------|--------|
| Base (no adapt.) | 88.7 | 40.5 | 56.3 |
| Standard LoRA | 39.2 ± 2.4 | **65.3 ± 0.7** | 54.9 ± 0.4 |
| EWC + LoRA | 48.8 ± 1.9 | 62.1 ± 0.8 | 55.4 ± 0.3 |
| Safe LoRA | 57.6 ± 1.6 | 60.4 ± 0.9 | 55.1 ± 0.4 |
| SafeGrad + LoRA | 63.8 ± 1.5 | 61.7 ± 0.6 | 55.6 ± 0.3 |
| Safety Interleaving | 61.2 ± 1.8 | 63.1 ± 0.7 | 55.3 ± 0.4 |
| **SafeAnchor** | **82.6 ± 1.0** | <u>63.8 ± 0.5</u> | **55.9 ± 0.3** |

---

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA ≥ 12.1 (recommended)
- GPU: 2× A100 40GB recommended (minimum: 2× RTX 3090 24GB)

### Option 1: pip (recommended)

```bash
git clone https://anonymous.4open.science/r/SafeAnchor
cd SafeAnchor

pip install -e .
```

### Option 2: pip with development dependencies

```bash
pip install -e ".[dev]"
```

### Option 3: conda environment

```bash
conda env create -f environment.yml
conda activate safeanchor
pip install -e .
```

### Verify installation

```bash
python -c "import safeanchor; print(safeanchor.__version__)"
make check-env
```

---

## Quick Start

### 1. Prepare datasets

SafeAnchor requires the following datasets (loaded via HuggingFace Hub):

| Dataset | Purpose | HF ID |
|---------|---------|-------|
| BeaverTails | Safety calibration set (500 examples) | `PKU-Alignment/BeaverTails` |
| MedQA | Medical domain training | `bigbio/med_qa` |
| LegalBench | Legal domain training | `nguha/legalbench` |
| CodeAlpaca | Code domain training | `sahil2801/CodeAlpaca-20k` |
| HarmBench | Safety probe set (200 examples) | `walledai/HarmBench` |

Datasets are downloaded automatically on first use; no manual setup required.

### 2. Configure API keys (optional, for W&B tracking)

```bash
cp .env.example .env
# Edit .env with your WANDB_API_KEY
```

### 3. Run SafeAnchor on Llama-2-7B-Chat

```bash
python train.py
```

This runs the full three-domain pipeline (Medical → Legal → Code) with default hyperparameters on Llama-2-7B-Chat.

### 4. Run on Mistral-7B-Instruct

```bash
python train.py model=mistral
```

### 5. Evaluate a trained model

```bash
python evaluate.py --checkpoint checkpoints/safeanchor_llama2.pt
```

---

## Repository Structure

```
safeanchor/
├── configs/                    # Hydra configuration files
│   ├── config.yaml             # Main configuration entry point
│   ├── model/                  # Model-specific configs (llama2, mistral)
│   ├── training/               # Training configs (default, debug, distributed)
│   └── experiment/             # Experiment configs
│
├── src/
│   ├──  models/                 # SafeAnchor model components
│   │   ├── safety_subspace.py  # SSI: Fisher Information subspace identification
│   │   ├── osca.py             # OSCA: Orthogonal safety-constrained adaptation
│   │   ├── csm.py              # CSM: Cumulative safety monitoring
│   │   ├── safeanchor.py       # Unified SafeAnchor framework
│   │   └── baselines.py        # Baseline method implementations
│   ├── data/                   # Data loading and preprocessing
│   ├── training/               # Training loop and callbacks
│   ├── evaluation/             # Safety metrics and evaluation
│   └── utils/                  # Checkpointing, logging
│
├── tests/                      # Comprehensive test suite
├── train.py                    # Main training entry point
├── evaluate.py                 # Evaluation entry point
├── predict.py                  # Single-sample inference
└── demo.py                     # Interactive demo
```

---

## Method

SafeAnchor addresses the sequential safety preservation problem through three integrated components:

### Safety Subspace Identification (SSI)

For each LoRA layer *i*, we compute the empirical Fisher Information Matrix over a safety calibration set *D*_safe:

```
F_i = (1/N_s) Σ ∇_δᵢ log p_θ(y|x) ∇_δᵢ log p_θ(y|x)ᵀ
```

Eigenvectors capturing 90% of variance (ρ = 0.90) form the **safety subspace basis** V_i^safe. After each domain, the subspace is updated incrementally via SVD truncation to prevent rank growth.

### Orthogonal Safety-Constrained Adaptation (OSCA)

Domain gradient updates are projected onto the **orthogonal complement** of the safety subspace:

```
g̃ᵢᵗ = gᵢᵗ - Πᵢ^safe · gᵢᵗ = (I - Vᵢ^safe (Vᵢ^safe)ᵀ) · gᵢᵗ
```

An **adaptive relaxation coefficient** α_i = max(0, 1 − λ · tr(F_i)) reduces constraint strictness for layers with low safety concentration.

### Cumulative Safety Monitoring (CSM)

After each domain adaptation, the safety refusal rate *s_t* is evaluated on a held-out probe set using LlamaGuard. If *s_t* < (1−τ)*s_0*, a safety replay phase is triggered for 200 steps.

### Training Objective

```
L_total = L_task(D_t) + γ · L_anchor
```

where *L_anchor* is a forward KL divergence term penalizing distributional shifts on safety-relevant inputs.

### Default Hyperparameters

| Hyperparameter | Value | Description |
|---------------|-------|-------------|
| LoRA rank *r* | 16 | LoRA rank |
| LoRA alpha | 32 | LoRA scaling |
| LoRA targets | Q, K, V, O | Attention projection layers |
| Learning rate | 2e-4 | With cosine schedule |
| Batch size | 8 | Per-device |
| Optimizer | AdamW | |
| ρ | 0.90 | Variance threshold for SSI |
| τ | 0.05 | CSM fractional tolerance |
| γ | 0.1 | Anchor loss weight |
| λ | 0.5 | Adaptive projection strictness |
| β | 1.0 | Replay safety-task balance |
| N_s | 500 | Safety calibration set size |
| N_probe | 200 | Safety probe set size |
| E_repair | 200 | Replay steps |
| Training examples | 5,000 | Per domain |
| Epochs | 3 | Per domain |
| Seeds | 5 | For variance reporting |

---

### Expected Runtime

| Experiment | Time | GPUs |
|-----------|------|------|
| Single seed, 3 domains | ~8h | 2× A100 40GB |
| SSI per domain | ~12 min | 2× A100 40GB |
| OSCA overhead vs. standard | +18% per step | 2× A100 40GB |
| CSM per transition | ~5 min | 2× A100 40GB |
| Full 5-seed run | ~40h | 2× A100 40GB |

---

## Configuration

Configuration is managed with [Hydra](https://hydra.cc). Override any parameter at the command line:

```bash
# Change model
python train.py model=mistral

# Change hyperparameters
python train.py training.lora.rank=32 model.rho=0.95

# Debug mode (fast iteration with tiny data)
python train.py training=debug

# Disable W&B
python train.py wandb.enabled=false

# Multi-GPU
torchrun --nproc_per_node=2 train.py training=distributed
```

Full configuration documentation: [`docs/training.md`](docs/training.md)

---

## Evaluation

### Benchmark Overview

SafeAnchor is evaluated on 8 benchmarks:

**Safety metrics:**
- **HarmBench** — refusal rate on 200 harmful prompts
- **TruthfulQA** — truthfulness score
- **BBQ** — bias score (inverted for composite; lower bias = higher score)
- **WildGuard** — jailbreak robustness (reported separately)

**Composite Safety Score:**
```
Safety = (1/3) × [HarmBench/100 + TruthfulQA/100 + (100 − BBQ_bias)/100] × 100
```

**Domain metrics:**
- **MedQA** — medical question answering accuracy
- **LegalBench** — legal reasoning accuracy
- **HumanEval** — code generation pass@1

**General capability:**
- **MMLU** — massive multitask language understanding

### Running Evaluation

```bash
# Evaluate a saved checkpoint on all benchmarks
python evaluate.py --checkpoint checkpoints/safeanchor_llama2.pt

# Evaluate on safety metrics only
python evaluate.py --checkpoint checkpoints/safeanchor_llama2.pt --suite safety

# Evaluate on domain metrics only
python evaluate.py --checkpoint checkpoints/safeanchor_llama2.pt --suite domain

# Adversarial evaluation (GCG attack)
python evaluate.py --checkpoint checkpoints/safeanchor_llama2.pt --adversarial
```

---

## Compute Requirements

### Hardware

| Setup | GPU | VRAM | RAM | Storage |
|-------|-----|------|-----|---------|
| Minimum | 2× RTX 3090 | 24GB each | 64GB | 200GB |
| Recommended | 2× A100 40GB | 40GB each | 128GB | 500GB |
| Paper setup | 2× A100 40GB | 40GB each | 128GB | 500GB |

### Compute Budget

- **Single training run (1 seed, 3 domains):** ~8 GPU-hours (A100)
- **Full paper reproduction (5 seeds, both models):** ~160 GPU-hours (A100)
- **Total GPU-hours for all experiments:** ~250 GPU-hours

### Memory Optimization

For GPUs with less than 40GB VRAM:

```bash
# Enable gradient checkpointing
python train.py training.gradient_checkpointing=true

# Reduce batch size + increase gradient accumulation
python train.py training.batch_size=4 training.gradient_accumulation_steps=2
```

---

**Common issues:**

1. **CUDA OOM during SSI:** Reduce `model.safety_calibration.n_samples` to 250.
2. **LlamaGuard not available:** The CSM module requires `meta-llama/LlamaGuard-7b`; ensure HuggingFace access is configured.
3. **Reproducibility differences:** Ensure `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before training.

---

## Contact

This repository is released anonymously for double-blind review.

Post-review, author contact information will be provided here.

