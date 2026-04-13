import os
import glob
import random
import numpy as np
import torch
from src.config import cfg


def set_seed(seed: int = cfg.train.seed) -> None:
    """Sets random seeds for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensures deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Selects the optimal available hardware device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_directories() -> None:
    """Creates necessary directories if they do not exist."""
    os.makedirs(cfg.paths.data_dir, exist_ok=True)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)


def verify_local_dataset() -> bool:
    """
    Checks existence of the .parquet files in raw dataset directory.
    Returns True if the files are found else raises exception.
    """
    parquet_files = glob.glob(os.path.join(cfg.paths.raw_dataset_dir, "*.parquet"))

    if not parquet_files:
        print(
            f"\n[CRITICAL ERROR] Local dataset mode is enabled, but no files were found!"
        )
        print(f"Expected to find .parquet files in {cfg.paths.raw_dataset_dir}")
        print(
            "Either download the .parquet files to that directory, or set 'use_local_dataset = False' in src/config.py"
        )
        raise FileNotFoundError(f"No .parquet files in {cfg.paths.raw_dataset_dir}")

    print(f"[INFO] Found {len(parquet_files)} local .parquet files. Proceeding...")
    return True
