import os
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm

from src.config import cfg
from src.utils import set_seed, get_device, ensure_directories, verify_local_dataset


def get_text_batches(dataset_iterator, tokenizer, batch_size: int, seq_len: int):
    """Yields tokenized batches, skipping overly short documents."""
    batch_tokens = []

    for item in dataset_iterator:
        tokens = tokenizer.encode(item["text"], add_special_tokens=True)

        for i in range(0, len(tokens) - seq_len, seq_len):
            chunk = tokens[i : i + seq_len]
            batch_tokens.append(chunk)

            if len(batch_tokens) == batch_size:
                yield torch.tensor(batch_tokens, dtype=torch.long)
                batch_tokens = []


def main() -> None:
    set_seed()
    ensure_directories()

    if cfg.model.use_local_dataset:
        verify_local_dataset()

    device = get_device()

    print(f"[INFO] Initializing collection on {device.type.upper()}")

    # Load Model
    dtype = torch.bfloat16 if device.type in ["cuda", "mps"] else torch.float32
    model = HookedTransformer.from_pretrained(
        cfg.model.model_name, device=device, dtype=dtype
    )
    model.eval()

    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    # Setup Dataset Stream
    if cfg.model.use_local_dataset:
        print(f"[INFO] Streaming local dataset from {cfg.paths.raw_dataset_dir}...")
        dataset = load_dataset(
            "parquet", data_dir=cfg.paths.raw_dataset_dir, split="train", streaming=True
        )
    else:
        print(f"[INFO] Streaming {cfg.model.dataset_name} from Hugging Face Hub...")
        dataset = load_dataset(
            cfg.model.dataset_name,
            name=cfg.model.dataset_config,
            split="train",
            streaming=True,
        )
    dataset_iter = iter(dataset)

    # Collection Loop
    buffer = []
    current_tokens = 0
    total_collected = 0
    chunk_idx = 0

    pbar = tqdm(total=cfg.train.total_tokens, desc="Caching Activations")

    for batch in get_text_batches(
        dataset_iter,
        model.tokenizer,
        cfg.train.batch_size_collection,
        cfg.train.seq_len,
    ):
        if total_collected >= cfg.train.total_tokens:
            break

        batch = batch.to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(batch, names_filter=[cfg.model.hook_name])

        activations = cache[cfg.model.hook_name].to("cpu", dtype=torch.float32)
        batch_cpu = batch.to("cpu")

        # Filter padding tokens to maintain high-quality training distribution
        pad_mask = batch_cpu != model.tokenizer.pad_token_id
        valid_activations = activations[pad_mask]

        buffer.append(valid_activations)
        valid_count = valid_activations.shape[0]

        current_tokens += valid_count
        total_collected += valid_count
        pbar.update(valid_count)

        model.reset_hooks()

        # Flush to Disk
        if current_tokens >= cfg.train.tokens_per_chunk:
            chunk_tensor = torch.cat(buffer, dim=0)[: cfg.train.tokens_per_chunk]
            save_path = os.path.join(
                cfg.paths.data_dir, f"activations_chunk_{chunk_idx}.pt"
            )
            torch.save(chunk_tensor, save_path)

            buffer = [torch.cat(buffer, dim=0)[cfg.train.tokens_per_chunk :]]
            current_tokens = buffer[0].shape[0]
            chunk_idx += 1

    pbar.close()

    # Save remainder
    if current_tokens > 0:
        chunk_tensor = torch.cat(buffer, dim=0)
        save_path = os.path.join(
            cfg.paths.data_dir, f"activations_chunk_{chunk_idx}.pt"
        )
        torch.save(chunk_tensor, save_path)


if __name__ == "__main__":
    main()
