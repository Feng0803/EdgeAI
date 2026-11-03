"""Fixed, calibration-free NF4 weight-only fake quantization.

Only original weight values enter the quantizer. The codebook and block size
are constants; there is no dataset, activation collection, clipping search,
layer selection, double quantization, or gradient training.
NF4 source: Dettmers et al., https://arxiv.org/abs/2305.14314.
Codebook matches bitsandbytes 0.48.1 get_4bit_type('nf4').
"""

import torch
from torch import nn
from torch.nn import functional as F


BLOCK_SIZE = 64
NF4_VALUES = (
    -1.0, -0.6961928009986877, -0.5250730514526367,
    -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
    -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434,
    0.5626170039176941, 0.7229568362236023, 1.0,
)
POLICY = dict(method='fixed NF4 weight-only fake PTQ', nominal_weight_bits=4,
              activation_dtype='original FP16 path', block_size=BLOCK_SIZE,
              codebook=list(NF4_VALUES), scale_rule='FP32 block absmax',
              zero_block_rule='scale=0, code=7, exact zero reconstruction',
              rounding='nearest fixed codepoint; midpoint ties choose lower code',
              targets='all nn.Linear under model.layers; fixed architecture scope',
              excluded='input embedding and tied lm_head stay FP16',
              calibration_dataset=None, calibration_forward_passes=0,
              activation_calibration=False, clipping_search=False,
              precision_search=False, double_quantization=False,
              gradients=False, actual_weight_storage='FP16 fake quantized values')


def codebook(device):
    return torch.tensor(NF4_VALUES, dtype=torch.float32, device=device)


@torch.no_grad()
def quantize(weight):
    """Return logical NF4 uint8 codes and FP32 scales, not packed storage."""
    if not weight.is_floating_point():
        raise TypeError('Expected floating-point weights')
    if not torch.isfinite(weight).all():
        raise ValueError('Non-finite weights')
    if not weight.numel():
        return torch.empty_like(weight, dtype=torch.uint8), torch.empty(0, device=weight.device)
    values = weight.float().reshape(-1)
    padded = F.pad(values, (0, (-values.numel()) % BLOCK_SIZE))
    groups = padded.reshape(-1, BLOCK_SIZE)
    scales = groups.abs().amax(1)
    divisor = torch.where(scales == 0, torch.ones_like(scales), scales)
    normalized = (groups/divisor[:, None]).reshape(-1)[:values.numel()]
    levels = codebook(weight.device)
    boundaries = (levels[:-1]+levels[1:])*.5
    codes = torch.bucketize(normalized, boundaries, right=False).byte().reshape(weight.shape)
    return codes, scales


@torch.no_grad()
def dequantize(codes, scales, dtype=torch.float16):
    if not codes.numel():
        return torch.empty_like(codes, dtype=dtype)
    if scales.numel() != (codes.numel()+BLOCK_SIZE-1)//BLOCK_SIZE:
        raise ValueError('NF4 scale count does not match weights')
    flat = codes.reshape(-1).long()
    scale_per_weight = torch.repeat_interleave(scales, BLOCK_SIZE)[:flat.numel()]
    return (codebook(codes.device)[flat]*scale_per_weight).reshape(codes.shape).to(dtype)


@torch.no_grad()
def fake_quantize_decoder(model, rows_per_chunk=256, progress=None):
    """Freeze every policy choice before evaluation; modify decoder weights only."""
    if rows_per_chunk < 1:
        raise ValueError('Positive chunk size required')
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if embedding.weight.data_ptr() != head.weight.data_ptr():
        raise ValueError('Expected original Qwen3 tied embedding/lm_head')
    shared_version = head.weight._version
    statistics = []
    for name, layer in model.named_modules():
        if not name.startswith('model.layers.') or not isinstance(layer, nn.Linear):
            continue
        if layer.weight.ndim != 2 or layer.in_features % BLOCK_SIZE:
            raise ValueError('Expected decoder Linear rows divisible by fixed block size')
        squared_error = 0.0
        weight_energy = 0.0
        for start in range(0, layer.out_features, rows_per_chunk):
            original = layer.weight[start:start+rows_per_chunk]
            codes, scales = quantize(original)
            reconstructed = dequantize(codes, scales, original.dtype)
            squared_error += (original.float()-reconstructed.float()).square().sum().item()
            weight_energy += original.float().square().sum().item()
            original.copy_(reconstructed)
        row = dict(name=name, shape=list(layer.weight.shape), weights=layer.weight.numel(),
                   sse=squared_error, mse=squared_error/layer.weight.numel(),
                   relative_weight_mse=squared_error/max(weight_energy, 1e-30))
        statistics.append(row)
        if progress is not None:
            progress(row, len(statistics))
    if head.weight._version != shared_version:
        raise AssertionError('Fixed FP16 shared matrix was modified')
    if embedding.weight.data_ptr() != head.weight.data_ptr():
        raise AssertionError('Original embedding/head sharing was lost')
    return statistics
