"""
Quantitative evaluation suite for the trained SAE.
"""

from __future__ import annotations

import json
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import ExperimentConfig
from src.logger import get_logger, init_clearml
from src.sae_model import SparseAutoencoder
from src.utils import (
    ensure_directories,
    find_activation_chunks,
    find_checkpoint,
    get_device,
    set_seed,
)


def run(cfg: ExperimentConfig) -> None:
    """
    Evaluates the trained SAE across all cached activation chunks.

    Reports:
        - Mean Squared Error (MSE)
        - Fraction of Variance Explained (FVE)
        - L0 sparsity (average active features per token)
        - Dead feature count and percentage
        - Feature activation frequency distribution
    """
    set_seed(cfg.train.seed)
    ensure_directories(
        cfg.paths.data_dir, cfg.paths.checkpoint_dir, cfg.paths.artifacts_dir
    )

    device = get_device()

    task = init_clearml(cfg, task_name="SAE_Evaluation")
    logger = get_logger(task)

    # Load SAE
    sae_path = find_checkpoint(cfg.paths.checkpoint_dir, "sae_final.pt")
    sae = SparseAutoencoder(cfg.model.d_model, cfg.d_sae).to(device)
    sae.load_state_dict(torch.load(sae_path, map_location=device, weights_only=True))
    sae.eval()
    print(f"[INFO] Loaded SAE from: {sae_path}")

    # Load activation data
    chunk_files = find_activation_chunks(cfg.paths.data_dir)

    # Accumulators
    total_mse = 0.0
    total_variance = 0.0
    total_l0 = 0.0
    total_tokens = 0
    feature_activation_counts = torch.zeros(cfg.d_sae, device=device)

    for chunk_file in tqdm(chunk_files, desc="Evaluating chunks"):
        activations = torch.load(chunk_file, map_location="cpu", weights_only=True)
        dataloader = DataLoader(
            TensorDataset(activations),
            batch_size=cfg.train.batch_size_train,
            shuffle=False,
        )

        for (x_batch,) in dataloader:
            x = x_batch.to(device)
            n = x.shape[0]

            with torch.no_grad():
                x_hat, f_x, _ = sae(x)

                batch_mse = (x - x_hat).pow(2).sum().item()
                batch_var = x.var(dim=0, unbiased=False).sum().item() * n

                total_mse += batch_mse
                total_variance += batch_var
                total_l0 += (f_x > 0).float().sum().item()
                total_tokens += n

                feature_activation_counts += (f_x > 0).float().sum(dim=0)

    # Compute aggregate metrics
    mean_mse = total_mse / (total_tokens * cfg.model.d_model)
    fve = 1.0 - (total_mse / total_tokens) / (total_variance / total_tokens) if total_variance > 0 else 0.0
    mean_l0 = total_l0 / total_tokens
    dead_features = (feature_activation_counts == 0).sum().item()
    dead_pct = 100.0 * dead_features / cfg.d_sae

    # Feature utilization statistics
    alive_counts = feature_activation_counts[feature_activation_counts > 0]
    median_freq = alive_counts.median().item() if len(alive_counts) > 0 else 0
    max_freq = alive_counts.max().item() if len(alive_counts) > 0 else 0

    # Print report
    print(f"\n{'='*60}")
    print(f"SAE EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Total tokens evaluated:  {total_tokens:,}")
    print(f"SAE dimensions:          {cfg.model.d_model} -> {cfg.d_sae}")
    print(f"{'='*60}")
    print(f"Reconstruction MSE:      {mean_mse:.6f}")
    print(f"Fraction Var. Explained: {fve:.4f} ({fve*100:.2f}%)")
    print(f"Mean L0 Sparsity:        {mean_l0:.1f} / {cfg.d_sae}")
    print(f"Dead Features:           {dead_features} / {cfg.d_sae} ({dead_pct:.1f}%)")
    print(f"Alive Features:          {cfg.d_sae - dead_features}")
    print(f"Median Feature Freq:     {median_freq:.0f}")
    print(f"Max Feature Freq:        {max_freq:.0f}")
    print(f"{'='*60}")

    # Save results
    results = {
        "total_tokens": total_tokens,
        "d_model": cfg.model.d_model,
        "d_sae": cfg.d_sae,
        "mean_mse": mean_mse,
        "fve": fve,
        "mean_l0": mean_l0,
        "dead_features": dead_features,
        "dead_pct": dead_pct,
        "alive_features": cfg.d_sae - dead_features,
        "median_feature_freq": median_freq,
        "max_feature_freq": max_freq,
    }

    report_path = os.path.join(cfg.paths.artifacts_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Evaluation report saved: {report_path}")

    # Log to ClearML
    for key, value in results.items():
        if isinstance(value, (int, float)):
            logger.report_single_value(name=key, value=value)

    if task is not None:
        task.upload_artifact(name="Evaluation_Report", artifact_object=results)
        task.close()
