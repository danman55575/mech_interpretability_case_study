import os
import json
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

from src.config import cfg
from src.utils import set_seed, get_device, verify_local_dataset
from src.sae_model import SparseAutoencoder
from src.logger import init_clearml

# Configuration for interpretation
NUM_TEXTS_TO_ANALYZE = 100
TOP_K_EXAMPLES = 5
ACTIVATION_THRESHOLD = 1.0


def main() -> None:
    set_seed()

    if cfg.model.use_local_dataset:
        verify_local_dataset()

    device = get_device()

    # Initialize ClearML task
    task = init_clearml(task_name="SAE_Interpretation")

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

    if cfg.model.use_local_dataset:
        print(f"[INFO] Fetching samples from {cfg.paths.raw_dataset_dir}...")
        dataset = load_dataset(
            "parquet", data_dir=cfg.paths.raw_dataset_dir, split="train", streaming=True
        )
    else:
        print(f"[INFO] Fetching samples from {cfg.model.dataset_name}...")
        dataset = load_dataset(
            cfg.model.dataset_name,
            name=cfg.model.dataset_config,
            split="train",
            streaming=True,
        )
    texts = [
        item["text"][:1000] for i, item in zip(range(NUM_TEXTS_TO_ANALYZE), dataset)
    ]

    feature_activations: dict[int, list[dict]] = {}

    print("[INFO] Running texts through models to extract max activating contexts...")
    for text in texts:
        tokens = llm.tokenizer.encode(text, return_tensors="pt").to(device)

        with torch.no_grad():
            _, cache = llm.run_with_cache(tokens, names_filter=[cfg.model.hook_name])
            llm_activations = cache[cfg.model.hook_name][0].to(
                torch.float32
            )  # [seq_len, d_model]

            f_x = sae.encode(llm_activations)  # [seq_len, d_sae]

            # Mask out early tokens to prevent "Attention Sink" outlier activations
            if f_x.shape[0] > 5:
                f_x[:5, :] = 0.0

            max_vals, max_idx = f_x.max(dim=0)
            active_features = (max_vals > ACTIVATION_THRESHOLD).nonzero(as_tuple=True)[
                0
            ]

            for feat_idx in active_features.tolist():
                val = max_vals[feat_idx].item()
                token_pos = max_idx[feat_idx].item()

                # Extract context (5 tokens before, 3 after)
                start_pos = max(0, token_pos - 5)
                end_pos = min(tokens.shape[1], token_pos + 4)

                target_token_str = llm.tokenizer.decode([tokens[0, token_pos]])
                context_str = llm.tokenizer.decode(tokens[0, start_pos:end_pos])
                clean_context = context_str.replace(
                    target_token_str, f" >>{target_token_str.strip()}<< "
                ).replace("\n", " ")

                if feat_idx not in feature_activations:
                    feature_activations[feat_idx] = []

                feature_activations[feat_idx].append(
                    {
                        "score": round(val, 3),
                        "target": target_token_str,
                        "context": clean_context,
                    }
                )

    # Sort and compile the artifact
    print("[INFO] Compiling feature dictionary artifact...")
    sorted_features = sorted(
        feature_activations.items(),
        key=lambda x: max(v["score"] for v in x[1]),
        reverse=True,
    )

    artifact_dict = {}
    for feat_idx, examples in sorted_features[
        :200
    ]:  # Save top 200 features for analysis
        examples.sort(key=lambda x: x["score"], reverse=True)
        artifact_dict[str(feat_idx)] = examples[:TOP_K_EXAMPLES]

    # Save locally and upload to ClearML
    artifact_path = os.path.join(cfg.paths.base_dir, "data", "feature_annotations.json")
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact_dict, f, indent=4, ensure_ascii=False)

    task.upload_artifact(name="Feature_Dictionary", artifact_object=artifact_dict)
    print(
        f"\n[INFO] Interpretation complete! Artifact saved to {artifact_path} and uploaded to ClearML."
    )
    task.close()


if __name__ == "__main__":
    main()
