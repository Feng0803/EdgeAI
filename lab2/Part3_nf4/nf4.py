"""Fixed, calibration-free NF4 W4A6 fake quantization.

Only the current weight or activation values enter the quantizers.  Every
format choice is fixed before the target model is loaded: there is no dataset,
activation-statistics collection, clipping search, layer selection, precision
search, or gradient training.  Activations are quantized dynamically during
inference, but their values never change the fixed bit width or block rule.
NF4 source: Dettmers et al., https://arxiv.org/abs/2305.14314.
Codebook matches bitsandbytes 0.48.1 get_4bit_type('nf4').
"""

import torch
from torch import nn
from torch.nn import functional as F


BLOCK_SIZE = 64
ACTIVATION_BITS = 6
ACTIVATION_QMAX = (1 << (ACTIVATION_BITS - 1)) - 1
SCALE_BITS = 16
NF4_VALUES = (
    -1.0, -0.6961928009986877, -0.5250730514526367,
    -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
    -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434,
    0.5626170039176941, 0.7229568362236023, 1.0,
)
POLICY = dict(method='fixed calibration-free NF4 W4A6 fake PTQ', nominal_weight_bits=4,
              activation_bits=ACTIVATION_BITS, block_size=BLOCK_SIZE,
              codebook=list(NF4_VALUES), scale_rule='FP32 block absmax',
              scale_storage='FP16 for both weight and activation blocks',
              activation_rule='dynamic symmetric INT6, one FP16 absmax scale per 64 contiguous values',
              activation_qrange=[-ACTIVATION_QMAX, ACTIVATION_QMAX],
              zero_block_rule='scale=0, code=7, exact zero reconstruction',
              activation_zero_block_rule='scale=0, integer code=0, exact zero reconstruction',
              rounding='NF4 nearest fixed codepoint; INT6 torch round-to-nearest-even',
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
    """Return logical NF4 uint8 codes and FP16 scales, not packed storage."""
    if not weight.is_floating_point():
        raise TypeError('Expected floating-point weights')
    if not torch.isfinite(weight).all():
        raise ValueError('Non-finite weights')
    if not weight.numel():
        return torch.empty_like(weight, dtype=torch.uint8), torch.empty(0, device=weight.device)
    values = weight.float().reshape(-1)
    padded = F.pad(values, (0, (-values.numel()) % BLOCK_SIZE))
    groups = padded.reshape(-1, BLOCK_SIZE)
    # Absmax is computed in FP32, then rounded to the format's stored FP16
    # scale before code selection.  This makes fake quantization reproduce the
    # scale precision included by the effective-bit accounting.
    scales = groups.abs().amax(1).to(torch.float16)
    divisor = torch.where(scales == 0, torch.ones_like(scales), scales).float()
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
def fake_quantize_activation(value, rows_per_chunk=256):
    """Apply fixed dynamic symmetric INT6 fake quantization in blocks of 64.

    Blocks never cross the last (Linear input-feature) dimension.  Chunking is
    only an execution-memory choice and cannot change block boundaries or the
    quantized values.
    """
    if not value.is_floating_point():
        raise TypeError('Expected floating-point activation')
    if value.ndim < 1 or value.shape[-1] % BLOCK_SIZE:
        raise ValueError('Linear input dimension must be divisible by block size')
    if rows_per_chunk < 1:
        raise ValueError('Positive activation chunk size required')
    rows = value.reshape(-1, value.shape[-1])
    reconstructed = torch.empty_like(rows)
    for start in range(0, rows.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, rows.shape[0])
        groups = rows[start:stop].float().reshape(-1, BLOCK_SIZE)
        scales = (groups.abs().amax(1) / ACTIVATION_QMAX).to(torch.float16)
        divisor = torch.where(scales == 0, torch.ones_like(scales), scales).float()
        # Reuse the FP32 group buffer to keep peak memory bounded on a T4.
        groups.div_(divisor[:, None]).round_().clamp_(-ACTIVATION_QMAX, ACTIVATION_QMAX)
        groups.mul_(divisor[:, None])
        reconstructed[start:stop].copy_(groups.reshape(stop-start, -1).to(value.dtype))
    return reconstructed.reshape(value.shape)


def install_activation_fake_quant(model, rows_per_chunk=256):
    """Install the fixed A6 rule on every decoder Linear and return its count."""
    if getattr(model, '_a6_fake_quant_handles', None):
        raise ValueError('Activation fake quantization is already installed')
    handles = []

    def pre_hook(module, inputs):
        if not inputs:
            raise ValueError('Linear activation hook received no input')
        return (fake_quantize_activation(inputs[0], rows_per_chunk), *inputs[1:])

    for name, layer in model.named_modules():
        if name.startswith('model.layers.') and isinstance(layer, nn.Linear):
            handles.append(layer.register_forward_pre_hook(pre_hook))
    if len(handles) != 252:
        for handle in handles:
            handle.remove()
        raise AssertionError(f'Expected 252 decoder Linear activations, got {len(handles)}')
    # Keeping handles on the model makes their lifetime explicit.  Hooks are
    # runtime behavior only and are reinstalled after checkpoint reload.
    model._a6_fake_quant_handles = handles
    return len(handles)


def realized_effective_bit_rate(model):
    """Return exact W4A6 logical-bit accounting for the 252 target Linears."""
    linears = [(name, layer) for name, layer in model.named_modules()
               if name.startswith('model.layers.') and isinstance(layer, nn.Linear)]
    if len(linears) != 252:
        raise AssertionError(f'Expected 252 decoder Linear layers, got {len(linears)}')
    if any(layer.bias is not None for _, layer in linears):
        raise AssertionError('Qwen3-4B target decoder Linears are expected to be bias-free')
    weight_values = sum(layer.weight.numel() for _, layer in linears)
    weight_scales = sum((layer.weight.numel()+BLOCK_SIZE-1)//BLOCK_SIZE
                        for _, layer in linears)
    activation_values_per_token = sum(layer.in_features for _, layer in linears)
    activation_scales_per_token = sum((layer.in_features+BLOCK_SIZE-1)//BLOCK_SIZE
                                      for _, layer in linears)
    codebook_bits = len(NF4_VALUES) * 32
    weight_total_bits = weight_values*4 + weight_scales*SCALE_BITS + codebook_bits
    activation_total_bits_per_token = (activation_values_per_token*ACTIVATION_BITS +
                                       activation_scales_per_token*SCALE_BITS)
    weight_effective_bits = weight_total_bits/weight_values
    activation_effective_bits = activation_total_bits_per_token/activation_values_per_token
    realized = (weight_effective_bits+activation_effective_bits)/2
    return dict(
        formula='(weight_effective_bits + activation_effective_bits) / 2',
        scope='252 decoder Linear weights and their input activations; embedding, lm_head, norms excluded',
        weight_values=weight_values,
        weight_code_bits=weight_values*4,
        weight_scale_count=weight_scales,
        weight_scale_bits=weight_scales*SCALE_BITS,
        global_nf4_codebook_bits=codebook_bits,
        weight_total_bits=weight_total_bits,
        weight_effective_bits=weight_effective_bits,
        activation_values_per_token=activation_values_per_token,
        activation_code_bits_per_token=activation_values_per_token*ACTIVATION_BITS,
        activation_scale_count_per_token=activation_scales_per_token,
        activation_scale_bits_per_token=activation_scales_per_token*SCALE_BITS,
        activation_total_bits_per_token=activation_total_bits_per_token,
        activation_effective_bits=activation_effective_bits,
        realized_effective_bit_rate=realized,
        assignment_limit=5.3,
        requirement_met=realized <= 5.3,
    )


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
