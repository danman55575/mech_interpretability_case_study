"""
Activation caching pipeline.
"""

from __future__ import annotations

import os
from typing import Iterator

import torch
from tqdm import tqdm

from src.config import ExperimentConfig
from src.model_loader import load_hooked_model, load_streaming_dataset
from src.utils import ensure_directories, get_device, set_seed


def tokenize_batches(
    dataset_iterator: Iterator,
    tokenizer,
    batch_size: int,
    seq_len: int,
) -> Iterator[torch.Tensor]:
    """
    Yields fixed-length tokenized batches from a streaming dataset.

    Documents are tokenized and split into non-overlapping chunks of
    exactly `seq_len` tokens. Partial trailing segments are discarded.

    Args:
        dataset_iterator: Iterator over dataset items with 'text' field.
        tokenizer: HuggingFace-compatible tokenizer.
        batch_size: Number of sequences per batch.
        seq_len: Fixed sequence length for each sample.

    Yields:
        Tensor of shape (batch_size, seq_len) with token IDs.
    """
    batch_buffer: list[list[int]] = []

    for item in dataset_iterator:
        tokens = tokenizer.encode(item["text"], add_special_tokens=True)
        # Split document into non-overlapping fixed-length chunks
        for start in range(0, len(tokens) - seq_len + 1, seq_len):
            chunk = tokens[start : start + seq_len]
            batch_buffer.append(chunk)

            if len(batch_buffer) == batch_size:
                yield torch.tensor(batch_buffer, dtype=torch.long)
                batch_buffer = []


def run(cfg: ExperimentConfig) -> None:
    """
    Main activation collection pipeline.

    Streams text data through the LLM, extracts residual stream
    activations at the configured layer, and saves them in chunks.
    """
    set_seed(cfg.train.seed)
    ensure_directories(cfg.paths.data_dir, cfg.paths.checkpoint_dir)

    device = get_device()
    print(f"[INFO] Starting activation collection on {device.type.upper()}")

    # Load model
    model = load_hooked_model(cfg, device)

    # Setup dataset stream
    dataset = load_streaming_dataset(cfg)
    dataset_iter = iter(dataset)

    # Collection state
    buffer: list[torch.Tensor] = []
    buffer_tokens: int = 0
    total_collected: int = 0
    chunk_idx: int = 0

    pbar = tqdm(total=cfg.train.total_tokens, desc="Caching Activations", unit="tok")

    for batch in tokenize_batches(
        dataset_iter,
        model.tokenizer,
        cfg.train.batch_size_collection,
        cfg.train.seq_len,
    ):
        if total_collected >= cfg.train.total_tokens:
            break

        batch = batch.to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch, names_filter=[cfg.model.hook_name]
            )

        # Extract and flatten: (batch, seq_len, d_model) -> (batch * seq_len, d_model)
        activations = cache[cfg.model.hook_name].to("cpu", dtype=torch.float32)
        flat_activations = activations.reshape(-1, cfg.model.d_model)

        buffer.append(flat_activations)
        n_new = flat_activations.shape[0]
        buffer_tokens += n_new
        total_collected += n_new
        pbar.update(n_new)

        # Explicit cache cleanup
        del cache
        model.reset_hooks()

        # Flush buffer to disk when chunk is full
        if buffer_tokens >= cfg.train.tokens_per_chunk:
            combined = torch.cat(buffer, dim=0)
            chunk_tensor = combined[: cfg.train.tokens_per_chunk]
            remainder = combined[cfg.train.tokens_per_chunk :]

            save_path = os.path.join(
                cfg.paths.data_dir, f"activations_chunk_{chunk_idx}.pt"
            )
            torch.save(chunk_tensor, save_path)
            print(f"\n[INFO] Saved {save_path} ({chunk_tensor.shape[0]:,} tokens)")

            buffer = [remainder] if remainder.shape[0] > 0 else []
            buffer_tokens = remainder.shape[0]
            chunk_idx += 1

    pbar.close()

    # Save any remaining activations
    if buffer_tokens > 0:
        chunk_tensor = torch.cat(buffer, dim=0)
        save_path = os.path.join(
            cfg.paths.data_dir, f"activations_chunk_{chunk_idx}.pt"
        )
        torch.save(chunk_tensor, save_path)
        print(f"[INFO] Saved final chunk: {save_path} ({chunk_tensor.shape[0]:,} tokens)")

    print(f"[INFO] Collection complete. Total tokens cached: {total_collected:,}")
