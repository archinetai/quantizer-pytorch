
# Quantizer - PyTorch

Different vector quantization methods, in PyTorch.

```bash
pip install quantizer-pytorch
```
[![PyPI - Python Version](https://img.shields.io/pypi/v/quantizer-pytorch?style=flat&colorA=black&colorB=black)](https://pypi.org/project/quantizer-pytorch/)

## Usage

### Timewise Quantizer 1d
```py
from quantizer_pytorch import Quantizer1d

quantizer = Quantizer1d(
    channels=32,
    num_groups=1,
    codebook_size=1024,
    num_residuals=2
)
quantizer.eval() # If the model is set to training mode quantizer will train with EMA by simply forwarding values

# Quantize sequence of shape [batch_size, channels, length]
x = torch.randn(1, 32, 80)
x_quantized, info = quantizer(x)

print(info.keys())                  # ['indices', 'loss', 'perplexity', 'replaced_codes']
print(x_quantized.shape)            # [1, 32, 80], same as input but quantized
print(info['indices'].shape)        # [1, 1, 80, 2], i.e. [batch, num_groups, length, num_residuals]
print(info['loss'])                 # 0.8637, the mean squared error between x and x_quantized
print(info['perplexity'])           # [70.3995, 67.5581], a metric used to check the codebook usage of each codebook (max=codebook_size)
print(info['replaced_codes'])       # [0, 0], number of replaced codes per group
print(info['budget'].shape)         # [num_residuals * num_groups, codebook_size], budget of each codebook element

# Reconstruct x_quantized from indices
x_quantiezed_recon = quantizer.from_ids(info['indices'])
assert torch.allclose(x_quantized, x_quantiezed_recon) # This assert should pass if in eval mode
```


### Channelwise Quantizer 1d
```py
from quantizer_pytorch import QuantizerChannelwise1d

quantizer = QuantizerChannelwise1d(
    channels=32,
    split_size=4, # Each channels will be split into vectors of size split_size and quantized
    num_groups=1,
    codebook_size=1024,
    num_residuals=1
)
quantizer.eval() # If the model is set to training mode quantizer will train with EMA by simply forwarding values

# Quantize sequence of shape [batch_size, channels, length]
x = torch.randn(1, 32, 80)
x_quantized, info = quantizer(x)

print(info.keys())                  # ['indices', 'loss', 'perplexity', 'replaced_codes']
print(x_quantized.shape)            # [1, 32, 80], same as input but quantized
print(info['indices'].shape)        # [1, 32, 20], since the length is 80 and we use a split_size (you can think of this as kernel_size=stride=split_size) we have 20 indices
print(info['loss'])                 # 0.0620, the mean squared error between x and x_quantized
print(info['perplexity'])           # [412.2037], a metric used to check the codebook usage of each codebook (max=codebook_size)
print(info['replaced_codes'])       # [1], number of replaced codes per group
print(info['budget'].shape)         # [num_residuals * num_groups, codebook_size], budget of each codebook element

# Reconstruct x_quantized from indices
x_quantiezed_recon = quantizer.from_ids(info['indices'])
assert torch.allclose(x_quantized, x_quantiezed_recon) # This assert should pass if in eval mode
```

### Blockwise Quantizer 1d
```py
from quantizer_pytorch import QuantizerBlock1d

quantizer = QuantizerBlock1d(
    channels=32,
    split_size=4,
    num_groups=2,
    codebook_size=1024,
    num_residuals=2
)
quantizer.eval() # If the model is set to training mode quantizer will train with EMA by simply forwarding values
x = torch.randn(1, 32, 40)
x_quantized, info = quantizer(x)

print(info.keys())                  # ['indices', 'loss', 'perplexity', 'replaced_codes', 'budget']
print(x_quantized.shape)            # [1, 32, 40], same as input but quantized
print(info['indices'].shape)        # [1, 20, 2], since the length is 80 and we use a split_size (you can think of this as kernel_size=stride=split_size) we have 20 indices
print(info['loss'])                 # 0.0620, the mean squared error between x and x_quantized
print(info['perplexity'])           # [20.0000, 18.6607], a metric used to check the codebook usage of each codebook (max=codebook_size)
print(info['replaced_codes'])       # [0, 0], number of replaced codes per codebook
print(info['budget'].shape)         # [2, 1024], budget of each codebook element

# Reconstruct x_quantized from indices
x_quantiezed_recon = quantizer.from_ids(info['indices'])
assert torch.allclose(x_quantized, x_quantiezed_recon) # This assert should pass if in eval mode
```

## Citations


```bibtex
@misc{2106.04283,
Author = {Rayhane Mama and Marc S. Tyndel and Hashiam Kadhim and Cole Clifford and Ragavan Thurairatnam},
Title = {NWT: Towards natural audio-to-video generation with representation learning},
Year = {2021},
Eprint = {arXiv:2106.04283},
}
```

```bibtex
@misc{2107.03312,
Author = {Neil Zeghidour and Alejandro Luebs and Ahmed Omran and Jan Skoglund and Marco Tagliasacchi},
Title = {SoundStream: An End-to-End Neural Audio Codec},
Year = {2021},
Eprint = {arXiv:2107.03312},
}
```
