# Decomposing Contextual Embeddings: Mechanistic Interpretability via Sparse Autoencoders

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A case study investigating the internal representations of Large Language Models (LLMs) through **Sparse Autoencoders (SAEs)**, addressing the problem of **polysemanticity** in contextual embeddings.

## Overview

Neural networks in LLMs encode multiple unrelated concepts per neuron (*polysemanticity*), making their representations difficult to interpret. This project trains a Sparse Autoencoder to decompose dense, entangled activations from an intermediate layer of **Qwen2.5-0.5B** into a dictionary of **interpretable, monosemantic features**. The project then demonstrate **activation steering**, i.e. causally intervening during the forward pass to deterministically alter the model's generative behavior *without fine-tuning*.

## Pipeline Architecture
```
┌─────────────┐  ┌─────────┐  ┌───────────┐  ┌───────────┐  ┌──────────────┐
│ collect     │─>│ train   │─>│ evaluate  │─>│ interpret │─>│ steer        │
│ Activations │  │ SAE     │  │ Quality   │  │ Features  │  │ Intervention │
└─────────────┘  └─────────┘  └───────────┘  └───────────┘  └──────────────┘
```

| Stage | Description |
|--------|-------------------------------------------------------------------|
| `collect`   | Streams text through the LLM and caches residual stream activations         |
| `train`     | Trains the SAE with MSE + L1 loss and decoder norm constraints              |
| `evaluate`  | Computes FVE, sparsity, dead feature statistics                             |
| `interpret` | Builds a feature dictionary mapping features to maximally activating contexts|
| `steer`     | Injects feature directions into the residual stream during generation       |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/mech-interp-case-study.git
cd mech-interp-case-study

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .
```

### Running Pipeline

```bash
# Full pipeline (recommended for first run)
python main.py pipeline

# Or run stages individually
python main.py collect
python main.py train
python main.py evaluate
python main.py interpret
python main.py steer --feature_id 2475 --alpha 25.0

# Override any config parameter
python main.py train --override train.learning_rate=5e-4 --override train.l1_coeff=5e-3
```

### Experiment Tracking

Create a `.env` file with ClearML credentials for experiment dashboard:

```bash
cp .env.example .env
# Edit .env with your ClearML API keys
```

## Project Structure

```
├── main.py                   # Unified CLI entry point
├── src/
│   ├── config.py             # Centralized configuration management
│   ├── model_loader.py       # LLM and dataset loading utilities
│   ├── sae_model.py          # Sparse Autoencoder architecture
│   ├── data_collection.py    # Activation caching pipeline
│   ├── train.py              # SAE training loop with ClearML logging
│   ├── evaluate.py           # Quantitative evaluation suite
│   ├── interpret.py          # Feature dictionary extraction
│   ├── steer.py              # Activation steering experiments
│   ├── logger.py             # ClearML integration with fallback
│   └── utils.py              # Reproducibility and I/O helpers
├── pyproject.toml            # Project metadata and dependencies
└── notebooks/                # Jupyter notebooks with experiments
```

## Configuration

| Parameter               | Default      | Description                               |
| ----------------------- | ------------ | ----------------------------------------- |
| `model.model_name`      | Qwen2.5-0.5B | Target LLM                                |
| `model.layer_idx`       | 12           | Layer to extract activations from         |
| `sae.expansion_factor`  | 32           | Dictionary size multiplier |
| `train.total_tokens`    | 2,000,000    | Tokens for activation collection          |
| `train.l1_coeff`        | 3.5e-3       | Sparsity penalty strength                 |
| `train.l1_warmup_steps` | 150          | Linear warmup for L1 coefficient          |

## Methodology

1. **Activation Extraction** : Collect residual stream activations from layer 11 of Qwen2.5-0.5B on FineWeb-Edu text data.
2. **SAE Training** : Train an overcomplete autoencoder (896 → sae.expansion_factor × 896) with L1 regularization to learn a sparse, interpretable feature dictionary.
3. **Feature Interpretation** : Identify monosemantic features by finding maximally activating text contexts for each learned direction.
4. **Activation Steering** : Inject feature directions into the residual stream during generation to causally test feature semantics.
5. **Evaluation** : Measure reconstruction quality (FVE), sparsity (L0), and steering impact (perplexity shift).
