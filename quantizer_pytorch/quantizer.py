from typing import Dict, Optional, Tuple, TypeVar, Union

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import LongTensor, Tensor, einsum, nn
from typing_extensions import TypeGuard

T = TypeVar("T")

"""
Utils
"""


def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None


def default(val: Optional[T], d: T) -> T:
    if exists(val):
        return val
    return d


"""
Quantization Strategies
"""


class Quantization(nn.Module):
    def from_ids(self, indices: LongTensor) -> Tensor:
        raise NotImplementedError()


def perplexity(onehot: Tensor, eps: float = 1e-10) -> Tensor:
    mean = reduce(onehot, "b h n s -> h s", "mean")
    return torch.exp(-reduce(mean * torch.log(mean + eps), "h s -> h", "sum"))


def ema_inplace(
    moving_avg: Union[Tensor, nn.Module], new: Tensor, decay: float
) -> None:
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))  # type: ignore


class VQ(Quantization):
    def __init__(
        self,
        features: int,
        num_heads: int,
        codebook_size: int,
        expire_threshold: int = 0,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        assert (features % num_heads) == 0, "features must be disivible by num_heads"

        self.num_heads = num_heads
        self.head_features = features // num_heads
        self.codebook_size = codebook_size
        self.expire_threshold = expire_threshold
        self.ema_decay = ema_decay

        # Initialize codebook (h, m, d)
        codebooks = torch.randn(num_heads, codebook_size, self.head_features)
        self.codebooks = nn.Parameter(codebooks)

        # Track codebook cluster size to expire dead codes faster
        ema_cluster_size = torch.zeros(num_heads, codebook_size)
        self.register_buffer("ema_cluster_size", ema_cluster_size)
        self.register_buffer("ema_embedding_sum", codebooks.clone())

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        b = x.shape[0]

        q = rearrange(x, "b n (h d) -> b h n d", h=self.num_heads)
        c = repeat(self.codebooks, "h m d -> b h m d", b=b)

        sim = -torch.cdist(q, c, p=2.0)  # b h n m

        codebook_indices = sim.argmax(dim=-1)
        attn = F.one_hot(codebook_indices, num_classes=self.codebook_size).float()

        out = einsum("b h n m, b h m d -> b h n d", attn, c)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = x + (out - x).detach() if self.training else out

        if self.training:
            self.update_codebooks(q, onehot=attn)

        info = {
            "indices": codebook_indices,
            "loss": F.mse_loss(x, out.detach()),
            "perplexity": perplexity(attn),
            "replaced_codes": self.expire_dead_codes(x),
        }
        return out, info

    def update_codebooks(self, q: Tensor, onehot: Tensor) -> None:
        """Update codebooks embeddings with EMA"""

        # Update codebook cluster sizes with EMA
        batch_cluster_size = reduce(onehot, "b h n m -> b h m", "sum")
        avg_cluster_size = reduce(batch_cluster_size, "b h m -> h m", "mean")
        ema_inplace(self.ema_cluster_size, avg_cluster_size, self.ema_decay)

        # Update codebook embedding sums with EMA
        batch_embedding_sum = einsum("b h n m, b h n d -> b h m d", onehot, q)
        avg_embedding_sum = reduce(batch_embedding_sum, "b h m d -> h m d", "mean")
        ema_inplace(self.ema_embedding_sum, avg_embedding_sum, self.ema_decay)

        # Update codebook embedding by averaging vectors
        self.codebooks.data.copy_(
            self.ema_embedding_sum
            / rearrange(self.ema_cluster_size + 1e-5, "h m -> h m 1")  # type: ignore
        )

    def expire_dead_codes(self, x: Tensor) -> Tensor:
        """Replaces dead codes in codebook with random batch elements"""
        is_disabled = self.expire_threshold <= 0

        # Mask is true where codes are expired
        expired_codes_per_head = self.ema_cluster_size < self.expire_threshold  # type: ignore # noqa
        num_expired_codes_per_head = reduce(expired_codes_per_head, "h m -> h", "sum")
        no_expired = torch.all(num_expired_codes_per_head == 0)

        # Return if no heads with expired codes, or if not training, or if disabled
        if not self.training or no_expired or is_disabled:
            return num_expired_codes_per_head

        # Candidate vectors for codebook replacement
        vectors = rearrange(x, "b h d -> (b h) d")
        n, device = vectors.shape[0], x.device
        new_codebooks = self.codebooks.data

        for head_idx in range(self.num_heads):
            num_expired_codes = num_expired_codes_per_head[head_idx]
            expired_codes = expired_codes_per_head[head_idx]  # type: ignore
            if n < num_expired_codes:
                # If fewer new samples than expired codes, repeat random duplicates
                ids = torch.randint(0, n, (num_expired_codes,), device=device)
            else:
                # If more new samples than expired codes, pick random candidates
                ids = torch.randperm(n, device=device)[0:num_expired_codes]
            # Update codebook head
            head_start = head_idx * self.head_features
            head_end = head_start + self.head_features
            new_codebooks[head_idx, expired_codes] = vectors[ids, head_start:head_end]

        self.codebooks.data.copy_(new_codebooks)
        return num_expired_codes_per_head

    def from_ids(self, indices: LongTensor) -> Tensor:
        b = indices.shape[0]
        c = repeat(self.codebooks, "h m d -> b h m d", b=b)
        # Get attention matrix from indices
        attn = F.one_hot(indices, num_classes=self.codebook_size).float()
        # Compute output with codebook
        out = einsum("b h n m, b h m d -> b h n d", attn, c)
        out = rearrange(out, "b h n d -> b n (h d)")
        return out


class ResidualVQ(Quantization):
    def __init__(self, num_residuals: int, shared_codebook: bool = True, **kwargs):
        super().__init__()
        self.num_residuals = num_residuals

        self.quantizers = nn.ModuleList([VQ(**kwargs) for _ in range(num_residuals)])

        if not shared_codebook:
            return

        first_vq, *rest_vq = self.quantizers
        codebooks = first_vq.codebooks

        for quantizer in rest_vq:
            quantizer.codebooks = codebooks

    def from_ids(
        self, indices: LongTensor, num_residuals: Optional[int] = None
    ) -> Tensor:
        r = default(num_residuals, self.num_residuals)
        indices = rearrange(indices, "b h (n r) -> r b h n", r=r)
        out = sum([self.quantizers[i].from_ids(indices[i]) for i in range(r)])
        return out

    def forward(
        self, x: Tensor, num_residuals: Optional[int] = None
    ) -> Tuple[Tensor, Dict]:
        r = default(num_residuals, self.num_residuals)
        assert r <= self.num_residuals, "num_residuals must be <= number of residuals"

        out, residual = torch.zeros_like(x), x
        all_indices, all_perplexities, all_replaced_codes, all_losses = [], [], [], []

        for i in range(r):
            quantized, info = self.quantizers[i](residual)
            residual = residual - quantized
            out = out + quantized
            all_indices += [info["indices"]]
            all_losses += [info["loss"]]
            all_perplexities += [info["perplexity"]]
            all_replaced_codes += [info["replaced_codes"]]

        info = {
            "indices": rearrange(all_indices, "r b h n -> b h (n r)"),
            "loss": reduce(torch.stack(all_losses), "r -> 1", "mean")[0],
            "perplexity": rearrange(all_perplexities, "r h -> (h r)"),
            "replaced_codes": rearrange(all_replaced_codes, "r h -> (h r)"),
        }

        return out, info


"""
Quantizers
"""


class Quantizer1d(nn.Module):
    def __init__(
        self,
        channels: int,
        num_groups: int,
        codebook_size: int,
        num_residuals: int = 1,
        **kwargs
    ):
        super().__init__()
        assert channels % num_groups == 0, "channels must be divisible by num_groups"
        self.num_groups = num_groups
        self.num_residuals = num_residuals

        self.quantize = ResidualVQ(
            features=channels,
            num_heads=num_groups,
            codebook_size=codebook_size,
            num_residuals=num_residuals,
            **kwargs
        )

    def from_ids(self, indices: LongTensor) -> Tensor:
        indices = rearrange(indices, "b g n r -> b g (n r)")
        x = self.quantize.from_ids(indices)
        return rearrange(x, "b t c -> b c t")

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        r = self.num_residuals
        x = rearrange(x, "b c t -> b t c")
        x, info = self.quantize(x)
        x = rearrange(x, "b t c -> b c t")
        # Rearrange indices to expose residual
        info["indices"] = rearrange(info["indices"], "b g (n r) -> b g n r", r=r)
        return x, info


class QuantizerChannelwise1d(nn.Module):
    def __init__(
        self,
        channels: int,
        split_size: int,
        num_groups: int,
        codebook_size: int,
        num_residuals: int = 1,
        **kwargs
    ):
        super().__init__()
        self.split_size = split_size
        self.num_groups = num_groups
        self.quantize = ResidualVQ(
            features=num_groups * split_size,
            num_heads=num_groups,
            codebook_size=codebook_size,
            num_residuals=num_residuals,
            **kwargs
        )

    def from_ids(self, indices: LongTensor) -> Tensor:
        g, s = self.num_groups, indices.shape[-1]
        indices = rearrange(indices, "b (g k) s -> b g (k s)", g=g)
        x = self.quantize.from_ids(indices)
        return rearrange(x, "b (k s) (g d) -> b (g k) (s d)", g=g, s=s)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        b, c, t = x.shape
        g, s = self.num_groups, t // self.split_size
        # Quantize each group in a different head (codebook)
        x = rearrange(x, "b (g k) (s d) -> b (k s) (g d)", g=g, s=s)
        x, info = self.quantize(x)
        x = rearrange(x, "b (k s) (g d) -> (b s) (g k) d", g=g, s=s)
        # Turn back to original shape
        x = rearrange(x, "(b s) (g k) d -> b (g k) (s d)", g=g, s=s)
        # Rearrange info to match input shape
        info["indices"] = rearrange(info["indices"], "b g (k s) -> b (g k) s", s=s)
        return x, info


class Quantizer2d(nn.Module):
    def __init__(self):
        super().__init__()
