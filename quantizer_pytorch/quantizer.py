from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import EinMix
from torch import LongTensor, Tensor, einsum, nn

"""
Masking
"""


def rho_schedule(num_steps: int, min: float, max: float, rho: float) -> Tensor:
    """https://www.desmos.com/calculator/ojcpwouiq9?lang=en"""
    i = torch.arange(num_steps, dtype=torch.float32)
    rho_inv = 1.0 / rho
    min_inv, max_inv = min**rho_inv, max**rho_inv
    return (max_inv + (i / (num_steps - 1)) * (min_inv - max_inv)) ** rho


class ImportanceRandomMasker(nn.Module):
    """Masks tokens with increasing probability forcing top to be more important"""

    def __init__(
        self,
        features: int,
        num_tokens: int,
        proba_min: float,
        proba_max: float,
        proba_rho: float,
    ):
        super().__init__()
        self.fixed_tokens = nn.Parameter(torch.randn(1, num_tokens, features))

        mask_proba = rho_schedule(
            num_steps=num_tokens, min=proba_min, max=proba_max, rho=proba_rho
        )
        self.register_buffer("mask_proba", mask_proba)

    def forward(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:
        b = tokens.shape[0]
        # Compute mask according to importance schedule
        mask_proba = repeat(self.mask_proba, "n -> b n 1", b=b)
        mask = torch.bernoulli(mask_proba).to(torch.bool)
        # Repeat mask fixed tokens over batch
        fixed_tokens = repeat(self.fixed_tokens, "1 n d -> b n d", b=b)
        # Replace tokens where masked with fixed tokens
        return torch.where(mask, tokens, fixed_tokens), mask


"""
Vector quantization
"""


def perplexity(onehot: Tensor, eps: float = 1e-10) -> Tensor:
    mean = reduce(onehot, "b h n s -> h s", "mean")
    return torch.exp(-reduce(mean * torch.log(mean + eps), "h s -> h", "sum"))


class Memcodes(nn.Module):
    """Adapted from https://github.com/lucidrains/NWT-pytorch"""

    def __init__(
        self,
        features: int,
        num_heads: int,
        codebook_size: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        assert (features % num_heads) == 0, "features must be disivible by num_heads"
        self.features = features
        self.num_heads = num_heads
        self.scale = (features // num_heads) ** -0.5
        self.codebook_size = codebook_size
        self.temperature = temperature

        num_codebooks = num_heads
        codebook_features = features // num_heads

        self.codebooks = nn.Parameter(
            torch.randn(num_codebooks, codebook_size, codebook_features)
        )
        # Different linear projection for each key/value head
        self.to_k = EinMix(
            pattern="h n d -> h n c",
            weight_shape="h d c",
            h=num_heads,
            d=codebook_features,
            c=codebook_features,
        )
        self.to_v = EinMix(
            pattern="h n d -> h n c",
            weight_shape="h d c",
            h=num_heads,
            d=codebook_features,
            c=codebook_features,
        )

    def from_ids(self, indices: LongTensor) -> Tensor:
        b = indices.shape[0]
        # Compute values from codebook
        v = repeat(self.to_v(self.codebooks), "h n d -> b h n d", b=b)
        # Repeat indices d times
        indices = repeat(indices, "... -> ... d", d=v.shape[-1])
        # Gather values on indices last dim
        out = v.gather(dim=2, index=indices)
        return rearrange(out, "b h n d -> b n (h d)")

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        assert x.shape[-1] == self.features

        # Split heads
        q = rearrange(x, "b n (h d) -> b h n d", h=self.num_heads) * self.scale
        # Compute keys/values of codebook for each head
        k, v = self.to_k(self.codebooks), self.to_v(self.codebooks)
        # Logits matrix between codebooks and input queries
        logits = einsum("b h i d, h j d -> b h i j", q, k)  # b, h, n, s

        if self.training:
            # Attention matrix with hard stochastic (differentiable) argmax
            attn = F.gumbel_softmax(logits, tau=self.temperature, dim=-1, hard=True)
            codebook_indices = attn.argmax(dim=-1)
        else:
            # Attention matrix with hard deterministic argmax
            codebook_indices = logits.argmax(dim=-1)
            attn = F.one_hot(codebook_indices, num_classes=self.codebook_size).float()

        out = einsum("b h i j, h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        info = {"indices": codebook_indices, "perplexity": perplexity(attn)}
        return out, info
