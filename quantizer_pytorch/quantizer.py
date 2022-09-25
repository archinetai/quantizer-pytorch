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


def ema(moving_avg: Union[Tensor, nn.Module], new: Tensor, decay: float) -> Tensor:
    return moving_avg * decay + new * (1 - decay)  # type: ignore


def update_inplace(
    old: Union[Tensor, nn.Module],
    new: Tensor,
) -> None:
    old.data.copy_(new)  # type: ignore


class BVQ(Quantization):
    """
    Budgeted Vector Quantization

    Features:
    [x] EMA update
    [x] Multiheaded codebook
    [x] Expiration invariant to the number of tokens, batch size, and codebook size.
    [x] Budgeted random replacement

    The total budget is always equivalent to the codebook size `m`, and each codebook
    element starts with budget of 1. The budget is slowly redistributed according to
    the distribution of the `n` incoming tokens with respect to the codebook. If a
    codebook element is matched by many incoming vectors, its buget will increase.
    The codebook vectors will be updated by averaging the matching incoming vectors.
    If a codebook element budget goes below the expire threshold, the element undergoes
    a hard replacement with a random vector from an incoming batch, and its budget is
    reset to 1. The total budget is then renormalized, at the expense of other codebook
    elements.
    """

    def __init__(
        self,
        features: int,
        num_heads: int,
        codebook_size: int,
        expire_threshold: float = 0.05,
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

        # Each element starts with budget=1, if it goes below threshold, it is replaced
        ema_budget = torch.ones(num_heads, codebook_size)
        self.register_buffer("budget_ema", ema_budget)

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
            "budget": self.budget_ema,
        }
        return out, info

    def update_codebooks(self, q: Tensor, onehot: Tensor) -> None:
        """Update codebooks embeddings with EMA"""
        b, n, m = q.shape[0], q.shape[2], self.codebook_size

        # Compute incoming total hits and avg embedding sum
        tot_incoming = reduce(onehot, "b h n m -> h m 1", "sum")
        sum_incoming = einsum("b h n m, b h n d -> h m d", onehot, q)
        avg_incoming = sum_incoming / (tot_incoming + 1e-5)

        # Mask for codebook elements that have not been hit by any vector
        mask = tot_incoming.bool()

        # Update codebook with EMA
        codebooks_new = torch.where(mask, avg_incoming, self.codebooks)
        codebooks_ema = ema(self.codebooks, codebooks_new, self.ema_decay)
        update_inplace(self.codebooks, codebooks_ema)

        # Compute budgets, update with EMA, renormalize such that: total budget < m
        budget = (rearrange(tot_incoming, "h m 1 -> h m") / (b * n)) * m
        budget_ema = ema(self.budget_ema, budget, self.ema_decay)
        budget_ema_norm = (budget_ema / reduce(budget_ema, "h m -> h 1", "sum")) * m
        update_inplace(self.budget_ema, budget_ema_norm)

    def expire_dead_codes(self, x: Tensor) -> Tensor:
        """Replaces dead codes in codebook with random batch elements"""
        is_disabled = self.expire_threshold <= 0

        # Mask is true where codes are expired
        expired_codes_per_head = self.budget_ema < self.expire_threshold  # type: ignore # noqa
        num_expired_codes_per_head = reduce(expired_codes_per_head, "h m -> h", "sum")
        no_expired = torch.all(num_expired_codes_per_head == 0)

        # Return if no heads with expired codes, or if not training, or if disabled
        if not self.training or no_expired or is_disabled:
            return num_expired_codes_per_head

        # Candidate vectors for codebook replacement
        vectors = rearrange(x, "b h d -> (b h) d")
        n, m, device = vectors.shape[0], self.codebook_size, x.device
        codebooks_new = self.codebooks.data
        budget_new = self.budget_ema

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
            codebooks_new[head_idx, expired_codes] = vectors[ids, head_start:head_end]
            # Update budget head
            budget_new[head_idx] = torch.where(  # type: ignore
                expired_codes, torch.ones(m, device=device), budget_new[head_idx]  # type: ignore # noqa
            )

        # Update codebook
        update_inplace(self.codebooks, codebooks_new)

        # Normalize and update budget
        budget_new_norm = (budget_new / reduce(budget_new, "h m -> h 1", "sum")) * m
        update_inplace(self.budget_ema, budget_new_norm)

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

        self.quantizers = nn.ModuleList([BVQ(**kwargs) for _ in range(num_residuals)])

        if not shared_codebook:
            return

        # Share both codebooks and total budget
        first_vq, *rest_vq = self.quantizers
        codebooks = first_vq.codebooks
        budget_ema = first_vq.budget_ema

        for quantizer in rest_vq:
            quantizer.codebooks = codebooks
            quantizer.budget_ema = budget_ema

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
        all_indices = []
        all_perplexities = []
        all_replaced_codes = []
        all_losses = []
        all_budgets = []

        for i in range(r):
            quantized, info = self.quantizers[i](residual)
            residual = residual - quantized
            out = out + quantized
            all_indices += [info["indices"]]
            all_losses += [info["loss"]]
            all_perplexities += [info["perplexity"]]
            all_replaced_codes += [info["replaced_codes"]]
            all_budgets += [info["budget"]]

        info = {
            "indices": rearrange(all_indices, "r b h n -> b h (n r)"),
            "loss": reduce(torch.stack(all_losses), "r -> 1", "mean")[0],
            "perplexity": rearrange(all_perplexities, "r h -> (h r)"),
            "replaced_codes": rearrange(all_replaced_codes, "r h -> (h r)"),
            "budget": rearrange(all_budgets, "r h m -> (r h) m"),
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

    def from_ids(self, indices: LongTensor, **kwargs) -> Tensor:
        indices = rearrange(indices, "b g n r -> b g (n r)")
        x = self.quantize.from_ids(indices, **kwargs)
        return rearrange(x, "b t c -> b c t")

    def forward(self, x: Tensor, **kwargs) -> Tuple[Tensor, Dict]:
        r = self.num_residuals
        x = rearrange(x, "b c t -> b t c")
        x, info = self.quantize(x, **kwargs)
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

    def from_ids(self, indices: LongTensor, **kwargs) -> Tensor:
        g, s = self.num_groups, indices.shape[-1]
        indices = rearrange(indices, "b (g k) s -> b g (k s)", g=g)
        x = self.quantize.from_ids(indices, **kwargs)
        return rearrange(x, "b (k s) (g d) -> b (g k) (s d)", g=g, s=s)

    def forward(self, x: Tensor, **kwargs) -> Tuple[Tensor, Dict]:
        b, c, t = x.shape
        g, s = self.num_groups, t // self.split_size
        # Quantize each group in a different head (codebook)
        x = rearrange(x, "b (g k) (s d) -> b (k s) (g d)", g=g, s=s)
        x, info = self.quantize(x, **kwargs)
        x = rearrange(x, "b (k s) (g d) -> (b s) (g k) d", g=g, s=s)
        # Turn back to original shape
        x = rearrange(x, "(b s) (g k) d -> b (g k) (s d)", g=g, s=s)
        # Rearrange info to match input shape
        info["indices"] = rearrange(info["indices"], "b g (k s) -> b (g k) s", s=s)
        return x, info


class QuantizerBlock1d(nn.Module):
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
        assert channels % num_groups == 0, "channels must be divisible by num_groups"
        self.split_size = split_size
        self.num_groups = num_groups
        self.num_residuals = num_residuals
        self.quantize = ResidualVQ(
            features=(channels // num_groups) * split_size,
            num_heads=1,
            codebook_size=codebook_size,
            num_residuals=num_residuals,
            **kwargs
        )

    def from_ids(self, indices: LongTensor, **kwargs) -> Tensor:
        cn, sd, r = self.num_groups, self.split_size, self.num_residuals
        indices = rearrange(indices, "b sn r -> b 1 (sn r)", r=r)
        x = self.quantize.from_ids(indices, **kwargs)
        return rearrange(x, "b (cn sn) (cd sd) -> b (cn cd) (sn sd)", cn=cn, sd=sd)

    def forward(self, x: Tensor, **kwargs) -> Tuple[Tensor, Dict]:
        cn, sd, r = self.num_groups, self.split_size, self.num_residuals
        x = rearrange(x, "b (cn cd) (sn sd) -> b (cn sn) (cd sd)", cn=cn, sd=sd)
        x, info = self.quantize(x, **kwargs)
        x = rearrange(x, "b (cn sn) (cd sd) -> b (cn cd) (sn sd)", cn=cn, sd=sd)
        # Rearrange info to match input shape
        info["indices"] = rearrange(info["indices"], "b 1 (sn r) -> b sn r", r=r)
        return x, info


class Quantizer2d(nn.Module):
    def __init__(self):
        super().__init__()
