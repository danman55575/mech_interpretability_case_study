"""
Mechanistic Interpretability via Sparse Autoencoders.

This package provides tools for decomposing contextual embeddings from
Large Language Models into interpretable, monosemantic features using
Sparse Autoencoders (SAEs) with L1 regularization.
"""

from src.config import ExperimentConfig
from src.sae_model import SparseAutoencoder

__all__ = [
    "ExperimentConfig",
    "SparseAutoencoder",
]

__version__ = "0.2.0"
