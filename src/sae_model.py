import math
import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder (SAE) with L1 regularization and normalized weights.
    Designed to decompose dense LLM representations into monosemantic features.
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        # Pre-encoder bias (centers the activations before encoding)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Encoder: Linear mapping to expanded dimensional space
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder: Maps sparse features back to original space
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initializes weights using Anthropic's recommended tied initialization."""
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))

        # Tie decoder weights to encoder transpose initially
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())

        self.normalize_decoder_weights()

    def normalize_decoder_weights(self) -> None:
        """
        Projects decoder weights to unit L2 norm.
        Must be explicitly called after every optimizer step to prevent artificial loss scaling.
        """
        with torch.no_grad():
            norms = torch.norm(self.W_dec, dim=1, keepdim=True)
            # Add epsilon to prevent division by zero for dead features
            self.W_dec.copy_(self.W_dec / (norms + 1e-8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input activations into sparse feature representations."""
        x_centered = x - self.b_dec
        f_x = torch.relu(x_centered @ self.W_enc + self.b_enc)
        return f_x

    def decode(self, f_x: torch.Tensor) -> torch.Tensor:
        """Reconstructs the original activations from the sparse features."""
        x_hat = (f_x @ self.W_dec) + self.b_dec
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns both the reconstructed activation and the sparse feature vector."""
        f_x = self.encode(x)
        x_hat = self.decode(f_x)
        return x_hat, f_x
