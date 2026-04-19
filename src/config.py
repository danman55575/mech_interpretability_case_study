"""
Centralized configuration management.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=False)
class PathsConfig:
    """Directory paths for data and model artifacts."""

    base_dir: str = field(
        default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    @property
    def data_dir(self) -> str:
        return os.path.join(self.base_dir, "data", "activations_cache")

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.base_dir, "checkpoints", "sae_checkpoints")

    @property
    def raw_dataset_dir(self) -> str:
        return os.path.join(self.base_dir, "data", "raw_dataset")

    @property
    def artifacts_dir(self) -> str:
        return os.path.join(self.base_dir, "data", "artifacts")


@dataclass
class ModelConfig:
    """Target LLM configuration."""

    model_name: str = "Qwen/Qwen2.5-0.5B"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    layer_idx: int = 12
    d_model: int = 896
    use_local_dataset: bool = True

    @property
    def hook_name(self) -> str:
        return f"blocks.{self.layer_idx}.hook_resid_post"


@dataclass
class SAEConfig:
    """Sparse Autoencoder architecture parameters."""

    expansion_factor: int = 32


@dataclass
class TrainingConfig:
    """Hyperparameters for SAE training and data collection."""

    # Data collection
    batch_size_collection: int = 16
    seq_len: int = 512
    total_tokens: int = 400_000
    tokens_per_chunk: int = 200_000

    # SAE training
    batch_size_train: int = 4096
    learning_rate: float = 1e-3
    l1_coeff: float = 3.5e-3
    l1_warmup_steps: int = 30
    seed: int = 42


@dataclass
class SteerConfig:
    """Parameters for causal intervention experiments."""

    target_feature_id: int = 2475
    steering_coeff: float = 10.0
    prompt: str = "Can you tell me about what is life? Yes, life is"
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class InterpretConfig:
    """Parameters for feature interpretation."""

    num_texts: int = 200
    top_k_examples: int = 10
    activation_threshold: float = 2.0
    max_text_length: int = 1000
    num_features_to_save: int = 200
    context_window_before: int = 5
    context_window_after: int = 3
    # Number of initial tokens to mask (attention sink mitigation)
    attention_sink_tokens: int = 5


@dataclass
class ExperimentConfig:
    """Master configuration object aggregating all sub-configs."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    steer: SteerConfig = field(default_factory=SteerConfig)
    interp: InterpretConfig = field(default_factory=InterpretConfig)

    @property
    def d_sae(self) -> int:
        """Dynamically calculates the SAE dictionary size."""
        return self.model.d_model * self.sae.expansion_factor

    def override(self, overrides: dict[str, Any]) -> None:
        """
        Applies flat key overrides, for example {'train.learning_rate': 1e-4}.
        """
        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            if len(parts) != 2:
                raise ValueError(
                    f"Override key must be 'section.param', got: '{dotted_key}'"
                )
            section_name, param_name = parts
            section = getattr(self, section_name, None)
            if section is None:
                raise ValueError(f"Unknown config section: '{section_name}'")
            if not hasattr(section, param_name):
                raise ValueError(
                    f"Unknown parameter '{param_name}' in section '{section_name}'"
                )
            current_value = getattr(section, param_name)
            # Cast the override to the same type as the current value
            cast_value = type(current_value)(value)
            setattr(section, param_name, cast_value)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the entire config to a nested dictionary."""
        return {
            "model": asdict(self.model),
            "sae": asdict(self.sae),
            "train": asdict(self.train),
            "steer": asdict(self.steer),
            "interp": asdict(self.interp),
            "d_sae": self.d_sae,
        }


def get_default_config() -> ExperimentConfig:
    """Factory function that returns a fresh default configuration."""
    return ExperimentConfig()
