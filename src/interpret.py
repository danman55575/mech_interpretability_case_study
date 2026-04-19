"""
Feature dictionary extraction and interpretation.
"""

from __future__ import annotations

import json
import os

import torch
from tqdm import tqdm

from src.config import ExperimentConfig
from src.logger import get_logger, init_clearml
from src.model_loader import load_hooked_model, load_streaming_dataset
from src.sae_model import SparseAutoencoder
from src.utils import (
    ensure_directories,
    find_checkpoint,
    get_device,
    set_seed,
)


def extract_context(
    tokens: torch.Tensor,
    tokenizer,
    token_pos: int,
    window_before: int = 5,
    window_after: int = 3,
) -> tuple[str, str]:
    """
    Extracts a context window around a token position and highlights the target.

    Args:
        tokens: 1D tensor of token IDs.
        tokenizer: Tokenizer for decoding.
        token_pos: Position of the target token.
        window_before: Number of tokens before target to include.
        window_after: Number of tokens after target to include.

    Returns:
        Tuple of (target_token_string, highlighted_context_string).
    """
    start = max(0, token_pos - window_before)
    end = min(len(tokens), token_pos + window_after + 1)

    target_str = tokenizer.decode([tokens[token_pos].item()])
    context_str = tokenizer.decode(tokens[start:end].tolist())

    # Highlight the target token in context
    highlighted = context_str.replace(
        target_str, f" >>>{target_str.strip()}<<< ", 1
    ).replace("\n", " ").strip()

    return target_str.strip(), highlighted


def run(cfg: ExperimentConfig) -> None:
    """
    Main feature interpretation pipeline.

    For each text sample, computes SAE feature activations and records
    the top-activating contexts per feature. Outputs a JSON artifact
    mapping feature IDs to their maximally activating examples.
    """
    set_seed(cfg.train.seed)
    ensure_directories(
        cfg.paths.data_dir, cfg.paths.checkpoint_dir, cfg.paths.artifacts_dir
    )

    device = get_device()
    interp_cfg = cfg.interp

    # Initialize tracking
    task = init_clearml(cfg, task_name="SAE_Interpretation")

    # Load models
    llm = load_hooked_model(cfg, device)

    sae_path = find_checkpoint(cfg.paths.checkpoint_dir, "sae_final.pt")
    sae = SparseAutoencoder(cfg.model.d_model, cfg.d_sae).to(device)
    sae.load_state_dict(torch.load(sae_path, map_location=device, weights_only=True))
    sae.eval()
    print(f"[INFO] Loaded SAE from: {sae_path}")

    # Load text samples
    dataset = load_streaming_dataset(cfg)
    texts = []
    for i, item in enumerate(dataset):
        if i >= interp_cfg.num_texts:
            break
        texts.append(item["text"][: interp_cfg.max_text_length])

    print(f"[INFO] Analyzing {len(texts)} text samples for feature activations...")

    # Feature activation collection
    feature_activations: dict[int, list[dict]] = {}

    for text in tqdm(texts, desc="Extracting features"):
        tokens = llm.tokenizer.encode(text, return_tensors="pt").to(device)

        if tokens.shape[1] < interp_cfg.attention_sink_tokens + 2:
            continue  # Skip very short sequences

        with torch.no_grad():
            _, cache = llm.run_with_cache(
                tokens, names_filter=[cfg.model.hook_name]
            )
            llm_acts = cache[cfg.model.hook_name][0].float()  # (seq_len, d_model)

            f_x = sae.encode(llm_acts)  # (seq_len, d_sae)

            # Mask attention-sink positions to avoid outlier activations
            f_x[: interp_cfg.attention_sink_tokens, :] = 0.0

            # Find max activation per feature across sequence positions
            max_vals, max_positions = f_x.max(dim=0)  # (d_sae,)

        # Record features that exceed activation threshold
        active_mask = max_vals > interp_cfg.activation_threshold
        active_indices = active_mask.nonzero(as_tuple=True)[0]

        tokens_1d = tokens[0].cpu()

        for feat_idx in active_indices.tolist():
            val = max_vals[feat_idx].item()
            pos = max_positions[feat_idx].item()

            target_str, context_str = extract_context(
                tokens_1d,
                llm.tokenizer,
                pos,
                interp_cfg.context_window_before,
                interp_cfg.context_window_after,
            )

            if feat_idx not in feature_activations:
                feature_activations[feat_idx] = []

            feature_activations[feat_idx].append({
                "score": round(val, 3),
                "target": target_str,
                "context": context_str,
            })

        del cache
        llm.reset_hooks()

    # Sort features by their maximum observed activation
    print("[INFO] Compiling feature dictionary artifact...")
    sorted_features = sorted(
        feature_activations.items(),
        key=lambda x: max(ex["score"] for ex in x[1]),
        reverse=True,
    )

    # Build output artifact
    artifact: dict[str, list[dict]] = {}
    for feat_idx, examples in sorted_features[: interp_cfg.num_features_to_save]:
        examples.sort(key=lambda x: x["score"], reverse=True)
        artifact[str(feat_idx)] = examples[: interp_cfg.top_k_examples]

    # Save to disk
    artifact_path = os.path.join(cfg.paths.artifacts_dir, "feature_dictionary.json")
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Feature dictionary saved: {artifact_path}")
    print(f"[INFO] Total interpretable features found: {len(artifact)}")

    # Upload to ClearML
    if task is not None:
        task.upload_artifact(name="Feature_Dictionary", artifact_object=artifact)
        task.close()
