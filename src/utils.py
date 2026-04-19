"""
Utility functions for reproducibility, device management, and file I/O.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across all libraries for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Selects the optimal available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """Returns the optimal dtype for the given device."""
    if device.type in ("cuda", "mps"):
        return torch.bfloat16
    return torch.float32


def ensure_directories(
    data_dir: str,
    checkpoint_dir: str,
    artifacts_dir: Optional[str] = None,
) -> None:
    """Creates necessary directories if they do not exist."""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if artifacts_dir:
        os.makedirs(artifacts_dir, exist_ok=True)


def verify_local_dataset(raw_dataset_dir: str) -> None:
    """
    Validates the presence of .parquet files in the raw dataset directory.

    Raises:
        FileNotFoundError: If no .parquet files are found.
    """
    parquet_files = glob.glob(os.path.join(raw_dataset_dir, "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"Local dataset mode is enabled, but no .parquet files found in "
            f"'{raw_dataset_dir}'. Either download data there or set "
            f"model.use_local_dataset = False."
        )
    print(f"[INFO] Found {len(parquet_files)} local .parquet file(s).")


def find_checkpoint(checkpoint_dir: str, filename: str = "sae_final.pt") -> str:
    """
    Locates a SAE checkpoint file and validates its existence.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
    """
    path = os.path.join(checkpoint_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{path}'. "
            f"Run the training pipeline first (python main.py train)."
        )
    return path


def find_activation_chunks(data_dir: str) -> list[str]:
    """
    Discovers and returns sorted activation chunk file paths.

    Raises:
        FileNotFoundError: If no chunks exist.
    """
    chunk_files = sorted(glob.glob(os.path.join(data_dir, "activations_chunk_*.pt")))
    if not chunk_files:
        raise FileNotFoundError(
            f"No activation chunks found in '{data_dir}'. "
            f"Run the data collection pipeline first (python main.py collect)."
        )
    print(f"[INFO] Found {len(chunk_files)} activation chunk(s).")
    return chunk_files
