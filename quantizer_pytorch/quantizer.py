from typing import Dict, Optional, Tuple, TypeVar, Union

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import EinMix
from torch import LongTensor, Tensor, einsum, log, nn
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

    def from_mask(self, tokens: Tensor, mask: Tensor) -> Tensor:
        b = mask.shape[0]
        # Repeat mask fixed tokens over batch
        fixed_tokens = repeat(self.fixed_tokens, "1 n d -> b n d", b=b)
        # Replace tokens where masked with fixed tokens
        return torch.where(mask, tokens, fixed_tokens)

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
    """Vector Quantization Block with EMA"""

    def __init__(
        self,
        features: int,
        codebook_size: int,
        temperature: float = 0.0,
        cluster_size_expire_threshold: int = 2,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.temperature = temperature
        self.cluster_size_expire_threshold = cluster_size_expire_threshold
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        # Embedding parameters
        self.embedding = nn.Embedding(codebook_size, features)
        nn.init.kaiming_uniform_(self.embedding.weight)
        # Exponential Moving Average (EMA) parameters
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embedding_sum", self.embedding.weight.clone())

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        b, n, d = x.shape
        # Flatten
        q = rearrange(x, "b n d -> (b n) d")
        # Compute quantization
        k = self.embedding.weight
        z, indices, onehot = self.quantize(q, k, temperature=self.temperature)
        # Update embedding with EMA
        if self.training:
            self.update_embedding(q, onehot)
            self.expire_codes(new_samples=q)
        # Unflatten all and return
        quantized = rearrange(z, "(b n) d -> b n d", b=b)
        info = {
            "loss": F.mse_loss(quantized.detach(), x),
            "indices": rearrange(indices, "(b n) -> b 1 n", b=b),
            "perplexity": perplexity(rearrange(onehot, "(b n) m -> b 1 n m", b=b)),
            "ema_cluster_size": self.ema_cluster_size,
        }
        return quantized, info

    def from_ids(self, indices: LongTensor) -> Tensor:
        indices = rearrange(indices, "b 1 n -> b n")
        return self.embedding(indices)

    def quantize(self, q: Tensor, k: Tensor, temperature: float) -> Tuple[Tensor, ...]:
        (_, d), (m, d_) = q.shape, k.shape
        # Dimensionality checks
        assert d == d_, "Expected q, k to have same number of dimensions"
        # Compute similarity between queries and value vectors
        similarity = -self.distances(q, k)  # [n, m]
        # Get quatized indeces with highest similarity
        indices = self.get_indices(similarity, temperature=temperature)  # [n]
        # Compute hard attention matrix
        onehot = F.one_hot(indices, num_classes=m).float()  # [n, m]
        # Get quantized vectors
        z = einsum("n m, m d -> n d", onehot, k)
        # Copy gradients to input
        z = q + (z - q).detach() if self.training else z
        return z, indices, onehot

    def get_indices(self, similarity: Tensor, temperature: float) -> Tensor:
        if temperature == 0.0:
            return torch.argmax(similarity, dim=1)
        # Gumbel sample
        noise = torch.zeros_like(similarity).uniform_(0, 1)
        gumbel_noise = -log(-log(noise))
        return ((similarity / temperature) + gumbel_noise).argmax(dim=1)

    def distances(self, q: Tensor, k: Tensor) -> Tensor:
        l2_q = reduce(q**2, "n d -> n 1", "sum")
        l2_k = reduce(k**2, "m d -> m", "sum")
        sim = einsum("n d, m d -> n m", q, k)
        return l2_q + l2_k - 2 * sim

    def update_embedding(self, q: Tensor, z_onehot: Tensor) -> None:
        """Update codebook embeddings with EMA"""
        # Compute batch number of hits per codebook element
        batch_cluster_size = reduce(z_onehot, "n m -> m", "sum")
        # Compute batch overlapped embeddings
        batch_embedding_sum = einsum("n m, n d -> m d", z_onehot, q)
        # Update with EMA
        ema_inplace(self.ema_cluster_size, batch_cluster_size, self.ema_decay)  # [m]
        ema_inplace(self.ema_embedding_sum, batch_embedding_sum, self.ema_decay)
        # Update codebook embedding by averaging vectors
        new_embedding = self.ema_embedding_sum / rearrange(
            self.ema_cluster_size + 1e-5, "k -> k 1"  # type: ignore
        )
        self.embedding.weight.data.copy_(new_embedding)

    def expire_codes(self, new_samples: Tensor) -> None:
        """Replaces dead codes in codebook with random batch elements"""
        if self.cluster_size_expire_threshold == 0:
            return

        # Mask is true where codes are expired
        expired_codes = self.ema_cluster_size < self.cluster_size_expire_threshold  # type: ignore # noqa
        num_expired_codes: int = expired_codes.sum().item()  # type: ignore

        if num_expired_codes == 0:
            return

        n, device = new_samples.shape[0], new_samples.device

        if n < num_expired_codes:
            # If fewer new samples than expired codes, repeat with duplicates at random
            indices = torch.randint(0, n, (num_expired_codes,), device=device)
        else:
            # If more new samples than expired codes, pick random candidates
            indices = torch.randperm(n, device=device)[0:num_expired_codes]

        # Update codebook embedding
        self.embedding.weight.data[expired_codes] = new_samples[indices]


class Memcodes(Quantization):
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

    def from_ids(self, indices: LongTensor) -> Tensor:
        b = indices.shape[0]
        # Compute values from codebook
        v = repeat(self.to_v(self.codebooks), "h n d -> b h n d", b=b)
        # Repeat indices d times
        indices = repeat(indices, "... -> ... d", d=v.shape[-1])
        # Gather values on indices last dim
        out = v.gather(dim=2, index=indices)
        return rearrange(out, "b h n d -> b n (h d)")


def l2norm(x: Tensor) -> Tensor:
    return F.normalize(x, dim=-1)


class HVQ(Quantization):
    """Hybrid vector quantization: multiheaded, cosine sim, gumbel"""

    def __init__(
        self,
        features: int,
        num_heads: int,
        codebook_size: int,
        temperature: float = 0.5,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        assert (features % num_heads) == 0, "features must be disivible by num_heads"
        assert temperature > 0.0, "temperature cannot be <= 0"

        self.temperature = temperature
        self.num_heads = num_heads
        self.head_features = features // num_heads
        self.codebook_size = codebook_size

        # Initialize codebook (h, m, d)
        codebooks = torch.randn(num_heads, codebook_size, self.head_features)
        self.codebooks = nn.Parameter(codebooks)
        nn.init.kaiming_uniform_(self.codebooks)

    def from_ids(self, indices: LongTensor) -> Tensor:
        b = indices.shape[0]
        c = repeat(self.codebooks, "h m d -> b h m d", b=b)
        # Get attention matrix from indices
        attn = F.one_hot(indices, num_classes=self.codebook_size).float()
        # Compute output with codebook
        out = einsum("b h n i, b h j d -> b h n d", attn, c)
        out = rearrange(out, "b h n d -> b n (h d)")
        return out

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        b = x.shape[0]

        q = rearrange(x, "b n (h d) -> b h n d", h=self.num_heads)
        c = repeat(self.codebooks, "h m d -> b h m d", b=b)

        q2, c2 = map(l2norm, (q, c))
        sim = einsum("b h i d, b h j d -> b h i j", q2, c2)  # b h n m
        # sim = -torch.cdist(q, c, p=2.0)

        if self.training:
            # Attention matrix with hard stochastic (differentiable) argmax
            attn = F.gumbel_softmax(sim, tau=self.temperature, dim=-1, hard=True)
            codebook_indices = attn.argmax(dim=-1)
        else:
            # Attention matrix with hard deterministic argmax
            codebook_indices = sim.argmax(dim=-1)
            attn = F.one_hot(codebook_indices, num_classes=self.codebook_size).float()

        out = einsum("b h n i, b h j d -> b h n d", attn, c)
        out = rearrange(out, "b h n d -> b n (h d)")

        info = {"indices": codebook_indices, "perplexity": perplexity(attn)}
        return out, info


class ResidualHVQ(Quantization):
    def __init__(self, num_residuals: int, shared_codebook: bool = True, **kwargs):
        super().__init__()
        self.num_residuals = num_residuals

        self.quantizers = nn.ModuleList([HVQ(**kwargs) for _ in range(num_residuals)])

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
        all_indices, all_perplexities = [], []

        for i in range(r):
            quantized, info = self.quantizers[i](residual)
            residual = residual - quantized
            out = out + quantized
            all_indices += [info["indices"]]
            all_perplexities += [info["perplexity"]]

        info = {
            "indices": rearrange(all_indices, "r b h n -> b h (n r)"),
            "perplexity": rearrange(all_perplexities, "r h -> (h r)"),
        }

        return out, info


"""
Quantizers
"""


class Quantizer1d(nn.Module):
    def __init__(
        self,
        channels: int,
        split_size: int,
        num_groups: int,
        codebook_size: int,
        quantizer_type: str = "hvq",
        mask_proba_min: float = 1.0,
        mask_proba_max: float = 1.0,
        mask_proba_rho: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.split_size = split_size
        self.num_groups = num_groups
        quantize: Optional[Quantization] = None

        if quantizer_type == "memcodes":
            quantize = Memcodes(
                features=num_groups * split_size,
                num_heads=num_groups,
                codebook_size=codebook_size,
                **kwargs
            )
        elif quantizer_type == "vq":
            assert num_groups == 1, "num_groups must be 1 with with vq quantization"
            quantize = VQ(features=split_size, codebook_size=codebook_size, **kwargs)
        elif quantizer_type == "hvq":
            quantize = HVQ(
                features=num_groups * split_size,
                num_heads=num_groups,
                codebook_size=codebook_size,
                **kwargs
            )
        elif quantizer_type == "rhvq":
            quantize = ResidualHVQ(
                features=num_groups * split_size,
                num_heads=num_groups,
                codebook_size=codebook_size,
                **kwargs
            )
        else:
            raise ValueError("Invalid quantizer type")

        self.quantize = quantize

        self.mask = ImportanceRandomMasker(
            features=split_size,
            num_tokens=channels,
            proba_min=mask_proba_min,
            proba_max=mask_proba_max,
            proba_rho=mask_proba_rho,
        )

    def from_ids(self, indices: LongTensor, mask: Optional[Tensor] = None) -> Tensor:
        g, s = self.num_groups, indices.shape[-1]
        indices = rearrange(indices, "b (g k) s -> b g (k s)", g=g)
        x = self.quantize.from_ids(indices)

        if exists(mask):
            # Rearrange quantized into mask groups
            tokens = rearrange(x, "b (k s) (g d) -> (b s) (g k) d", g=g, s=s)
            mask = rearrange(mask, "b c s -> (b s) c 1")
            tokens = self.mask.from_mask(tokens, mask)  # type: ignore
            return rearrange(tokens, "(b s) c d -> b c (s d)", s=s)

        return rearrange(x, "b (k s) (g d) -> b (g k) (s d)", g=g, s=s)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        b, c, t = x.shape
        g, s = self.num_groups, t // self.split_size
        # Quantize each group in a different head (codebook)
        x = rearrange(x, "b (g k) (s d) -> b (k s) (g d)", g=g, s=s)
        x, info = self.quantize(x)
        # Mask channel tokens with increasing probability
        tokens = rearrange(x, "b (k s) (g d) -> (b s) (g k) d", g=g, s=s)
        tokens, mask = self.mask(tokens)
        # Turn back to original shape
        x = rearrange(tokens, "(b s) (g k) d -> b (g k) (s d)", g=g, s=s)
        # Rearrange info to match input shape
        info["indices"] = rearrange(info["indices"], "b g (k s) -> b (g k) s", s=s)
        info["mask"] = rearrange(mask, "(b s) (g k) 1 -> b (g k) s", g=g, s=s)
        return x, info


class Quantizer2d(nn.Module):
    def __init__(self):
        super().__init__()
