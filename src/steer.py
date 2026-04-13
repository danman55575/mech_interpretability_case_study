import os
import argparse
import torch
from transformer_lens import HookedTransformer
from clearml import Logger

from src.config import cfg
from src.utils import set_seed, get_device
from src.sae_model import SparseAutoencoder
from src.logger import init_clearml


def calculate_perplexity(
    model: HookedTransformer, text: str, device: torch.device
) -> float:
    """Calculates model perplexity for a given generated text."""
    tokens = model.tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        loss = model(tokens, return_type="loss")
    return torch.exp(loss).item()


def main() -> None:
    # CLI Argument Parsing for dynamic execution
    parser = argparse.ArgumentParser(description="Perform Activation Steering with SAE")
    parser.add_argument(
        "--feature_id",
        type=int,
        default=cfg.steer.target_feature_id,
        help="ID of the SAE feature to steer",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=cfg.steer.steering_coeff,
        help="Intervention strength (coefficient)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=cfg.steer.prompt,
        help="Starting prompt for generation",
    )
    args = parser.parse_args()

    set_seed()
    device = get_device()

    # Initialize ClearML task
    task = init_clearml(task_name=f"SAE_Steering_Feat_{args.feature_id}")
    logger = Logger.current_logger()

    # Log the specific CLI arguments to ClearML
    task.connect(vars(args), name="Steering_Arguments")

    print(f"[INFO] Loading LLM: {cfg.model.model_name}...")
    dtype = torch.bfloat16 if device.type in ["cuda", "mps"] else torch.float32
    llm = HookedTransformer.from_pretrained(
        cfg.model.model_name, device=device, dtype=dtype
    )
    llm.eval()

    print(f"[INFO] Loading SAE from {cfg.paths.checkpoint_dir}...")
    sae_path = os.path.join(cfg.paths.checkpoint_dir, "sae_final.pt")
    sae = SparseAutoencoder(cfg.model.d_model, cfg.d_sae).to(device)
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    sae.eval()

    # Extract Steering Vector
    steering_vector = sae.W_dec[args.feature_id].detach().clone()

    def steering_hook(resid_post, hook):
        return resid_post + (args.alpha * steering_vector)

    print(f"\nEXPERIMENT: STEERING FEATURE #{args.feature_id} (Alpha: {args.alpha})\n")

    # BASELINE
    print("\n--- BASELINE GENERATION (No Steering):")
    baseline_output = llm.generate(
        args.prompt, max_new_tokens=cfg.steer.max_new_tokens, temperature=0.7, top_p=0.9
    )
    baseline_ppl = calculate_perplexity(llm, baseline_output, device)

    print(f"Text: {baseline_output}")
    print(f"Perplexity: {baseline_ppl:.2f}")

    # INTERVENTION
    print("\n--- INTERVENTION GENERATION (With Steering):")
    with llm.hooks(fwd_hooks=[(cfg.model.hook_name, steering_hook)]):
        steered_output = llm.generate(
            args.prompt,
            max_new_tokens=cfg.steer.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )

    print(f"Text: {steered_output}")

    tokens = llm.tokenizer.encode(steered_output, return_tensors="pt").to(device)
    with llm.hooks(fwd_hooks=[(cfg.model.hook_name, steering_hook)]):
        with torch.no_grad():
            loss = llm(tokens, return_type="loss")
            steered_ppl = torch.exp(loss).item()

    print(f"Perplexity: {steered_ppl:.2f}")

    # METRICS & LOGGING
    ppl_shift = steered_ppl - baseline_ppl
    print("\n[3] QUANTITATIVE ANALYSIS:")
    print(f"PPL Shift: {ppl_shift:+.2f}")

    # Report final results to ClearML
    logger.report_single_value(name="Baseline PPL", value=baseline_ppl)
    logger.report_single_value(name="Steered PPL", value=steered_ppl)
    logger.report_single_value(name="PPL Shift", value=ppl_shift)
    logger.report_text(
        f"Baseline Text:\n{baseline_output}\n\nSteered Text:\n{steered_output}"
    )

    task.close()


if __name__ == "__main__":
    main()
