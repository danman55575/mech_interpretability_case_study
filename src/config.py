import os
from dataclasses import dataclass


@dataclass
class PathsConfig:
    """Directory paths for data and model artifacts."""

    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(base_dir, "data", "activations_cache")
    checkpoint_dir: str = os.path.join(base_dir, "checkpoints", "sae_checkpoints")


@dataclass
class ModelConfig:
    """Target LLM configuration."""

    model_name: str = "qwen/Qwen2.5-0.5B"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    layer_idx: int = 12
    d_model: int = 896

    @property
    def hook_name(self) -> str:
        return f"blocks.{self.layer_idx}.hook_resid_post"


@dataclass
class TrainingConfig:
    """Hyperparameters for SAE training and data collection."""

    # Data collection
    batch_size_collection: int = 4
    seq_len: int = 512
    total_tokens: int = 10_000
    tokens_per_chunk: int = 500_000

    # SAE Training
    batch_size_train: int = 2048
    learning_rate: float = 1e-3
    l1_coeff: float = 3e-3
    l1_warmup_steps: int = 5
    seed: int = 42


@dataclass
class SteerConfig:
    """Parameters for causal intervention experiments."""

    target_feature_id: int = 2475
    steering_coeff: float = 25.0
    prompt: str = "Can you tell about what is life? Yes, life is"
    max_new_tokens: int = 10


@dataclass
class SAEConfig:
    """Sparse Autoencoder architecture parameters."""

    expansion_factor: int = 32


@dataclass
class ExperimentConfig:
    """Master configuration object linking all sub-configs."""

    paths: PathsConfig = PathsConfig()
    model: ModelConfig = ModelConfig()
    sae: SAEConfig = SAEConfig()
    train: TrainingConfig = TrainingConfig()
    steer: SteerConfig = SteerConfig()

    @property
    def d_sae(self) -> int:
        """Dynamically calculates the SAE dictionary size."""
        return self.model.d_model * self.sae.expansion_factor


# Global configuration instance
cfg = ExperimentConfig()
