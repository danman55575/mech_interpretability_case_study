import os
import glob
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from clearml import Logger

from src.config import cfg
from src.utils import set_seed, get_device, ensure_directories
from src.logger import init_clearml
from src.sae_model import SparseAutoencoder


def get_l1_coeff(step: int, max_coeff: float, warmup_steps: int) -> float:
    """Calculates L1 penalty coefficient with linear warmup."""
    if step >= warmup_steps:
        return max_coeff
    return max_coeff * (step / warmup_steps)


def calculate_fve(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Computes Fraction of Variance Explained."""
    variance = torch.var(x, dim=0, unbiased=False).sum()
    mse = nn.functional.mse_loss(x_hat, x, reduction="sum") / x.shape[0]

    if variance.item() == 0:
        return 0.0
    return (1 - (mse / variance)).item()


def main() -> None:
    set_seed()
    ensure_directories()
    device = get_device()

    # Initialize ClearML experiment and sync hyperparameters
    task = init_clearml(task_name="SAE_Training_Run")
    logger = Logger.current_logger()

    # Model Setup
    sae = SparseAutoencoder(cfg.model.d_model, cfg.d_sae).to(device)
    optimizer = Adam(sae.parameters(), lr=cfg.train.learning_rate)

    chunk_files = sorted(
        glob.glob(os.path.join(cfg.paths.data_dir, "activations_chunk_*.pt"))
    )
    if not chunk_files:
        raise FileNotFoundError(
            f"No activation chunks in {cfg.paths.data_dir}. Run data_collection.py first."
        )

    global_step = 0

    for chunk_idx, chunk_file in enumerate(chunk_files):
        print(f"\n[INFO] Training on {os.path.basename(chunk_file)}")

        activations = torch.load(chunk_file, map_location="cpu")
        dataset = TensorDataset(activations)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size_train,
            shuffle=True,
            pin_memory=True,
        )

        pbar = tqdm(dataloader, desc=f"Chunk {chunk_idx + 1}/{len(chunk_files)}")

        for batch in pbar:
            x = batch[0].to(device)
            optimizer.zero_grad()

            # Forward pass
            x_hat, f_x = sae(x)

            # Loss computation
            mse_loss = nn.functional.mse_loss(x_hat, x)
            current_l1 = get_l1_coeff(
                global_step, cfg.train.l1_coeff, cfg.train.l1_warmup_steps
            )
            l1_loss = f_x.abs().sum(dim=1).mean()

            loss = mse_loss + (current_l1 * l1_loss)

            # Optimization step
            loss.backward()
            optimizer.step()

            # Constraint projection
            sae.normalize_decoder_weights()

            # Metrics
            with torch.no_grad():
                l0_norm = (f_x > 0).float().sum(dim=1).mean().item()
                fve = calculate_fve(x, x_hat)
                dead_features = (f_x.sum(dim=0) == 0).sum().item()

            # Update terminal and ClearML dashboard
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.3f}",
                    "L0": f"{l0_norm:.1f}",
                    "FVE": f"{fve:.3f}",
                }
            )

            logger.report_scalar(
                "Loss", "Total Loss", iteration=global_step, value=loss.item()
            )
            logger.report_scalar(
                "Loss", "MSE Loss", iteration=global_step, value=mse_loss.item()
            )
            logger.report_scalar(
                "Metrics", "L0 Norm", iteration=global_step, value=l0_norm
            )
            logger.report_scalar("Metrics", "FVE", iteration=global_step, value=fve)
            logger.report_scalar(
                "Metrics", "Dead Features", iteration=global_step, value=dead_features
            )
            logger.report_scalar(
                "Hyperparameters", "L1 Lambda", iteration=global_step, value=current_l1
            )

            global_step += 1

        # Save intermediary state
        chunk_save_path = os.path.join(
            cfg.paths.checkpoint_dir, f"sae_chunk_{chunk_idx + 1}.pt"
        )
        torch.save(sae.state_dict(), chunk_save_path)

    # Save final artifact
    final_path = os.path.join(cfg.paths.checkpoint_dir, "sae_final.pt")
    torch.save(sae.state_dict(), final_path)
    print(f"\n[INFO] Training completed. Final artifact stored at: {final_path}")

    task.close()


if __name__ == "__main__":
    main()
