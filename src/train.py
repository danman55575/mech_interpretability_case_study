"""
SAE training pipeline.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import ExperimentConfig
from src.logger import get_logger, init_clearml
from src.sae_model import SparseAutoencoder
from src.utils import (
    ensure_directories,
    find_activation_chunks,
    get_device,
    set_seed,
)


def compute_l1_coeff(step: int, max_coeff: float, warmup_steps: int) -> float:
    """
    Computes L1 penalty coefficient with linear warmup schedule.
    """
    if warmup_steps <= 0 or step >= warmup_steps:
        return max_coeff
    return max_coeff * (step / warmup_steps)


def compute_fve(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """
    Computes Fraction of Variance Explained (FVE).

    FVE = 1 - (MSE / Var(x)), measuring reconstruction quality
    as a proportion of explained input variance.
    """
    total_variance = x.var(dim=0, unbiased=False).sum()
    mse = (x - x_hat).pow(2).mean(dim=0).sum()

    if total_variance.item() < 1e-10:
        return 0.0
    return (1.0 - mse / total_variance).item()


def run(cfg: ExperimentConfig) -> None:
    """
    Main SAE training loop.

    Iterates over cached activation chunks, trains the SAE with
    MSE reconstruction loss + L1 sparsity penalty, and saves
    checkpoints after each chunk and at completion.
    """
    set_seed(cfg.train.seed)
    ensure_directories(
        cfg.paths.data_dir, cfg.paths.checkpoint_dir, cfg.paths.artifacts_dir
    )

    device = get_device()

    # Initialize tracking
    task = init_clearml(cfg, task_name="SAE_Training")
    logger = get_logger(task)

    # Initialize model
    sae = SparseAutoencoder(cfg.model.d_model, cfg.d_sae).to(device)
    optimizer = Adam(sae.parameters(), lr=cfg.train.learning_rate)

    print(
        f"[INFO] SAE architecture: {cfg.model.d_model} -> {cfg.d_sae} "
        f"({cfg.sae.expansion_factor}x expansion)"
    )
    total_params = sum(p.numel() for p in sae.parameters())
    print(f"[INFO] Total SAE parameters: {total_params:,}")

    # Discover activation chunks
    chunk_files = find_activation_chunks(cfg.paths.data_dir)

    global_step = 0

    for chunk_idx, chunk_file in enumerate(chunk_files):
        print(f"\n[INFO] Training on: {os.path.basename(chunk_file)}")
        activations = torch.load(chunk_file, map_location="cpu", weights_only=True)

        dataloader = DataLoader(
            TensorDataset(activations),
            batch_size=cfg.train.batch_size_train,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )

        desc = f"Chunk {chunk_idx + 1}/{len(chunk_files)}"
        pbar = tqdm(dataloader, desc=desc)

        for (x_batch,) in pbar:
            x = x_batch.to(device)

            # Forward pass
            x_hat, f_x, _ = sae(x)

            # Loss computation with warmup on L1
            mse_loss = nn.functional.mse_loss(x_hat, x)
            current_l1 = compute_l1_coeff(
                global_step, cfg.train.l1_coeff, cfg.train.l1_warmup_steps
            )
            l1_loss = f_x.abs().sum(dim=-1).mean()
            total_loss = mse_loss + current_l1 * l1_loss

            # Backward + optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Apply decoder norm constraint
            sae.normalize_decoder_weights()

            # Compute monitoring metrics
            with torch.no_grad():
                l0_norm = (f_x > 0).float().sum(dim=-1).mean().item()
                fve = compute_fve(x, x_hat)
                dead_features = (f_x.sum(dim=0) == 0).sum().item()
                dead_pct = 100.0 * dead_features / cfg.d_sae

            # Terminal display
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "MSE": f"{mse_loss.item():.4f}",
                "L0": f"{l0_norm:.1f}",
                "FVE": f"{fve:.4f}",
                "Dead%": f"{dead_pct:.1f}",
            })

            # ClearML logging
            logger.report_scalar("Loss", "Total", iteration=global_step, value=total_loss.item())
            logger.report_scalar("Loss", "MSE", iteration=global_step, value=mse_loss.item())
            logger.report_scalar("Loss", "L1 (raw)", iteration=global_step, value=l1_loss.item())
            logger.report_scalar("Metrics", "L0 Norm", iteration=global_step, value=l0_norm)
            logger.report_scalar("Metrics", "FVE", iteration=global_step, value=fve)
            logger.report_scalar("Metrics", "Dead Features", iteration=global_step, value=dead_features)
            logger.report_scalar("Hyperparams", "L1 Coeff", iteration=global_step, value=current_l1)

            global_step += 1

        # Save checkpoint after each chunk
        ckpt_path = os.path.join(
            cfg.paths.checkpoint_dir, f"sae_chunk_{chunk_idx + 1}.pt"
        )
        torch.save(sae.state_dict(), ckpt_path)
        print(f"[INFO] Checkpoint saved: {ckpt_path}")

    # Save final model
    final_path = os.path.join(cfg.paths.checkpoint_dir, "sae_final.pt")
    torch.save(sae.state_dict(), final_path)
    print(f"\n[INFO] Training complete. Final model: {final_path}")
    print(f"[INFO] Total training steps: {global_step:,}")

    if task is not None:
        task.close()
