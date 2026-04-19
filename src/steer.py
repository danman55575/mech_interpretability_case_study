"""
Activation steering experiments.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from src.config import ExperimentConfig
from src.logger import get_logger, init_clearml
from src.model_loader import load_hooked_model
from src.sae_model import SparseAutoencoder
from src.utils import find_checkpoint, get_device, set_seed


def compute_perplexity(
    model: HookedTransformer,
    text: str,
    device: torch.device,
    fwd_hooks: list | None = None,
) -> float:
    """
    Computes model perplexity for generated text, optionally under hooks.

    Args:
        model: HookedTransformer model.
        text: Text string to evaluate.
        device: Compute device.
        fwd_hooks: Optional list of forward hooks for intervention.

    Returns:
        Perplexity as a float.
    """
    tokens = model.tokenizer.encode(text, return_tensors="pt").to(device)

    if fwd_hooks:
        with model.hooks(fwd_hooks=fwd_hooks):
            with torch.no_grad():
                loss = model(tokens, return_type="loss")
    else:
        with torch.no_grad():
            loss = model(tokens, return_type="loss")

    return torch.exp(loss).item()


def run_single_experiment(
    cfg: ExperimentConfig,
    llm: HookedTransformer,
    sae: SparseAutoencoder,
    device: torch.device,
    feature_id: int,
    alpha: float,
    prompt: str,
    logger=None,
) -> dict:
    """
    Runs a single steering experiment: baseline vs. intervention.

    Args:
        cfg: Experiment configuration.
        llm: Loaded HookedTransformer model.
        sae: Loaded SAE model.
        device: Compute device.
        feature_id: SAE feature index to steer with.
        alpha: Steering coefficient (intervention strength).
        prompt: Starting prompt for generation.
        logger: ClearML logger or NullLogger.

    Returns:
        Dictionary with experimental results.
    """
    # Extract steering direction from SAE decoder
    steering_vector = sae.get_feature_vectors([feature_id])[0].to(device)

    def steering_hook(resid_post, hook):
        return resid_post + alpha * steering_vector

    hook_pair = [(cfg.model.hook_name, steering_hook)]

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: Feature #{feature_id} | Alpha: {alpha}")
    print(f"Prompt: \"{prompt}\"")
    print(f"{'='*70}")

    # Baseline generation
    print("\n[BASELINE] No steering:")
    baseline_text = llm.generate(
        prompt,
        max_new_tokens=cfg.steer.max_new_tokens,
        temperature=cfg.steer.temperature,
        top_p=cfg.steer.top_p,
    )
    baseline_ppl = compute_perplexity(llm, baseline_text, device)
    print(f"  Text: {baseline_text}")
    print(f"  Perplexity: {baseline_ppl:.2f}")

    # Steered generation
    print(f"\n[STEERED] Feature #{feature_id} at alpha={alpha}:")
    with llm.hooks(fwd_hooks=hook_pair):
        steered_text = llm.generate(
            prompt,
            max_new_tokens=cfg.steer.max_new_tokens,
            temperature=cfg.steer.temperature,
            top_p=cfg.steer.top_p,
        )

    # Compute steered perplexity under the same hook
    steered_ppl = compute_perplexity(llm, steered_text, device, fwd_hooks=hook_pair)
    print(f"  Text: {steered_text}")
    print(f"  Perplexity: {steered_ppl:.2f}")

    # Also compute perplexity of steered text (without hook)
    # This shows how the unmodified model perceives the steered output
    natural_ppl = compute_perplexity(llm, steered_text, device)

    # Analysis
    ppl_shift = steered_ppl - baseline_ppl
    print(f"\n[ANALYSIS]")
    print(f"  PPL Shift (steered context): {ppl_shift:+.2f}")
    print(f"  Natural PPL of steered text: {natural_ppl:.2f}")

    results = {
        "feature_id": feature_id,
        "alpha": alpha,
        "prompt": prompt,
        "baseline_text": baseline_text,
        "steered_text": steered_text,
        "baseline_ppl": baseline_ppl,
        "steered_ppl": steered_ppl,
        "natural_ppl": natural_ppl,
        "ppl_shift": ppl_shift,
    }

    # Log to ClearML
    if logger:
        prefix = f"Feat_{feature_id}_a{alpha}"
        logger.report_single_value(name=f"{prefix}/Baseline_PPL", value=baseline_ppl)
        logger.report_single_value(name=f"{prefix}/Steered_PPL", value=steered_ppl)
        logger.report_single_value(name=f"{prefix}/Natural_PPL", value=natural_ppl)
        logger.report_single_value(name=f"{prefix}/PPL_Shift", value=ppl_shift)
        logger.report_text(
            f"[{prefix}]\nBaseline:\n{baseline_text}\n\nSteered:\n{steered_text}"
        )

    return results


def run(cfg: ExperimentConfig) -> None:
    """
    Main steering pipeline.

    Loads models, runs baseline vs. steered generation, and logs results.
    """
    set_seed(cfg.train.seed)
    device = get_device()

    # Initialize tracking
    task = init_clearml(
        cfg, task_name=f"Steering_Feat_{cfg.steer.target_feature_id}"
    )
    logger = get_logger(task)

    # Load models
    llm = load_hooked_model(cfg, device)

    sae_path = find_checkpoint(cfg.paths.checkpoint_dir, "sae_final.pt")
    sae = SparseAutoencoder(cfg.model.d_model, cfg.d_sae).to(device)
    sae.load_state_dict(torch.load(sae_path, map_location=device, weights_only=True))
    sae.eval()
    print(f"[INFO] Loaded SAE from: {sae_path}")

    # Run experiment
    run_single_experiment(
        cfg=cfg,
        llm=llm,
        sae=sae,
        device=device,
        feature_id=cfg.steer.target_feature_id,
        alpha=cfg.steer.steering_coeff,
        prompt=cfg.steer.prompt,
        logger=logger,
    )

    if task is not None:
        task.close()
