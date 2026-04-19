"""
Centralized model and dataset loading utilities.
"""

from __future__ import annotations

import torch
from datasets import IterableDataset, load_dataset
from transformer_lens import HookedTransformer

from src.config import ExperimentConfig
from src.utils import get_dtype, verify_local_dataset


def load_hooked_model(
    cfg: ExperimentConfig,
    device: torch.device,
) -> HookedTransformer:
    """
    Loads the target LLM with TransformerLens hooks.

    Args:
        cfg: Experiment configuration.
        device: Target compute device.

    Returns:
        A HookedTransformer model in eval mode.
    """
    dtype = get_dtype(device)
    print(f"[INFO] Loading LLM: {cfg.model.model_name} (dtype={dtype}, device={device})")

    model = HookedTransformer.from_pretrained(
        cfg.model.model_name,
        device=device,
        dtype=dtype,
    )
    model.eval()

    # Ensure pad token is set for batched tokenization
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    return model


def load_streaming_dataset(cfg: ExperimentConfig) -> IterableDataset:
    """
    Returns a streaming dataset iterator based on config.

    Supports both local .parquet files and HuggingFace Hub streaming.

    Args:
        cfg: Experiment configuration.

    Returns:
        A streaming IterableDataset.
    """
    if cfg.model.use_local_dataset:
        verify_local_dataset(cfg.paths.raw_dataset_dir)
        print(f"[INFO] Streaming local dataset from: {cfg.paths.raw_dataset_dir}")
        dataset = load_dataset(
            "parquet",
            data_dir=cfg.paths.raw_dataset_dir,
            split="train",
            streaming=True,
        )
    else:
        print(f"[INFO] Streaming from HuggingFace Hub: {cfg.model.dataset_name}")
        dataset = load_dataset(
            cfg.model.dataset_name,
            name=cfg.model.dataset_config,
            split="train",
            streaming=True,
        )
    return dataset
