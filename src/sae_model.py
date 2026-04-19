"""
Sparse Autoencoder (SAE) for mechanistic interpretability.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with L1-regularized latent activations.

    The decoder weights W_dec are constrained to unit L2 norm per row
    to prevent the model from minimizing L1 loss by scaling down weights
    while scaling up decoder norms.

    Args:
        d_model: Dimensionality of the input LLM activations.
        d_sae: Dimensionality of the sparse feature dictionary.
    """

    def __init__(self, d_model: int, d_sae: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        # Pre-encoder bias: centers activations before encoding
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Encoder: projects to overcomplete sparse space
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder: maps sparse features back to activation space
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Tied initialization following Anthropic's approach:
        W_dec is initialized as the transpose of W_enc, then both are
        normalized so that decoder rows have unit L2 norm.
        """
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
        self.normalize_decoder_weights()

    @torch.no_grad()
    def normalize_decoder_weights(self) -> None:
        """
        Projects decoder weight rows to unit L2 norm.

        This constraint must be applied after every optimizer step
        to prevent the model from exploiting the L1 penalty by
        shrinking encoder activations while inflating decoder norms.
        """
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes input activations into sparse feature space.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Sparse feature activations of shape (..., d_sae).
        """
        x_centered = x - self.b_dec
        return torch.relu(x_centered @ self.W_enc + self.b_enc)

    def decode(self, f_x: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs activations from sparse features.

        Args:
            f_x: Sparse features of shape (..., d_sae).

        Returns:
            Reconstructed activations of shape (..., d_model).
        """
        return f_x @ self.W_dec + self.b_dec

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Tuple of (x_hat, f_x, mse_loss_per_sample):
                - x_hat: Reconstructed activations.
                - f_x: Sparse feature activations.
                - mse_loss: Per-sample MSE for diagnostics.
        """
        f_x = self.encode(x)
        x_hat = self.decode(f_x)
        mse_loss = (x - x_hat).pow(2).mean(dim=-1)
        return x_hat, f_x, mse_loss

    @property
    def num_features(self) -> int:
        return self.d_sae

    def get_feature_vectors(self, feature_ids: list[int]) -> torch.Tensor:
        """
        Extracts decoder weight vectors for specified feature IDs.

        Args:
            feature_ids: List of feature indices.

        Returns:
            Tensor of shape (len(feature_ids), d_model).
        """
        return self.W_dec[feature_ids].detach().clone()
