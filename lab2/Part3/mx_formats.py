"""MX/NVFP4 fake quantizers used by the AdaptQuant reproduction.

The implementation follows the OCP MX value grids.  A MX block contains 32
values and one E8M0 scale.  AdaptQuant assigns one element format to a
hardware-aligned 16 x 32 tile, while the 32 values in every row keep their own
shared scale.  T4 cannot execute Blackwell MX MMA instructions, so the result
is dequantized back to the input dtype before the ordinary Linear operation.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


BLOCK_VALUES = 32
TILE_ROWS = 16
TILE_VALUES = BLOCK_VALUES * TILE_ROWS

# name: (mantissa bits, exponent bias, largest finite magnitude)
FORMATS = {
    "mxfp4": (1, 1, 6.0),       # E2M1
    "mxfp6": (3, 1, 7.5),       # E2M3
    "mxfp8": (3, 7, 448.0),     # E4M3
}

# Element payload plus one 8-bit E8M0 scale per 32 values.
FORMAT_BITS = {"mxfp4": 4.25, "mxfp6": 6.25, "mxfp8": 8.25}
NVFP4_EFFECTIVE_BITS = 4.5

STATISTIC_MAX = 0.5 * math.log2(TILE_VALUES)


def _round_element(values: torch.Tensor, format_name: str) -> torch.Tensor:
    """Round to the selected floating-point grid (round-to-nearest-even)."""
    mantissa_bits, exponent_bias, largest = FORMATS[format_name]
    magnitude = values.abs()
    exponent = torch.floor(torch.log2(magnitude.clamp_min(1e-30)))
    exponent.clamp_(min=1 - exponent_bias)
    step = torch.exp2(exponent - mantissa_bits)
    rounded = torch.round(magnitude / step) * step
    return torch.copysign(rounded.clamp_max(largest), values)


def mx_fake_quant(values: torch.Tensor, format_name: str) -> torch.Tensor:
    """Quantize the last dimension in 32-value MX blocks and dequantize."""
    if format_name not in FORMATS:
        raise ValueError(f"Unsupported MX format: {format_name}")
    if values.shape[-1] % BLOCK_VALUES:
        raise ValueError("MX reduction dimension must be divisible by 32")
    original_shape = values.shape
    blocks = values.float().reshape(*original_shape[:-1], -1, BLOCK_VALUES)
    absmax = blocks.abs().amax(dim=-1, keepdim=True)
    largest = FORMATS[format_name][2]
    max_element_exponent = math.floor(math.log2(largest))
    scale_exponent = torch.floor(torch.log2(absmax.clamp_min(1e-30))) - max_element_exponent
    scale = torch.where(absmax > 0, torch.exp2(scale_exponent), torch.ones_like(absmax))
    quantized = _round_element(blocks / scale, format_name) * scale
    return quantized.reshape(original_shape)


def nvfp4_fake_quant(values: torch.Tensor) -> torch.Tensor:
    """NVFP4 reference: E2M1 values, 16-value blocks, E4M3 block scale."""
    if values.shape[-1] % 16:
        raise ValueError("NVFP4 reduction dimension must be divisible by 16")
    original_shape = values.shape
    source = values.float()
    tensor_scale = (source.abs().amax() / (6.0 * 448.0)).clamp_min(1e-30)
    blocks = source.reshape(*original_shape[:-1], -1, 16)
    block_scale = _round_element(blocks.abs().amax(dim=-1, keepdim=True) / (6.0 * tensor_scale),
                                 "mxfp8")
    scale = torch.where(block_scale > 0, block_scale * tensor_scale, torch.ones_like(block_scale))
    return (_round_element(blocks / scale, "mxfp4") * scale).reshape(original_shape)


def as_tiles(values: torch.Tensor) -> torch.Tensor:
    """View [rows, reduction] as [row_tiles, reduction_tiles, 16, 32]."""
    if values.ndim != 2:
        raise ValueError("AdaptQuant operands must be two-dimensional")
    rows, reduction = values.shape
    if rows % TILE_ROWS or reduction % BLOCK_VALUES:
        raise ValueError("AdaptQuant operand must be padded to 16 x 32 tiles")
    return values.reshape(rows // TILE_ROWS, TILE_ROWS,
                          reduction // BLOCK_VALUES, BLOCK_VALUES).transpose(1, 2)


def tile_statistic(values: torch.Tensor) -> torch.Tensor:
    """AdaptQuant's one-pass tile statistic: log2(absmax / RMS)."""
    tiles = as_tiles(values.float())
    absmax = tiles.abs().amax(dim=(-1, -2))
    rms = tiles.square().mean(dim=(-1, -2)).sqrt()
    return torch.where(absmax > 0,
                       torch.log2(absmax / rms.clamp_min(1e-30)),
                       torch.zeros_like(absmax))


def choose_formats(statistic: torch.Tensor, threshold_4_to_6: float,
                   threshold_6_to_8: float = math.inf) -> torch.Tensor:
    """Return format ids: 0=MXFP4, 1=MXFP6, 2=MXFP8."""
    return ((statistic >= threshold_4_to_6).long()
            + (statistic >= threshold_6_to_8).long())


def adaptquant_chunk(values: torch.Tensor, threshold_4_to_6: float,
                     threshold_6_to_8: float = math.inf) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply per-tile format selection to one tile-row-aligned 2-D chunk."""
    formats = choose_formats(tile_statistic(values), threshold_4_to_6, threshold_6_to_8)
    output = mx_fake_quant(values, "mxfp4")
    for format_id, format_name in ((1, "mxfp6"), (2, "mxfp8")):
        selected = formats == format_id
        if selected.any():
            alternative = mx_fake_quant(values, format_name)
            mask = selected[:, None, :, None].expand(-1, TILE_ROWS, -1, BLOCK_VALUES)
            output = torch.where(mask.reshape_as(values), alternative, output)
    return output.to(values.dtype), formats


@torch.no_grad()
def adaptquant_operand(values: torch.Tensor, threshold_4_to_6: float,
                       threshold_6_to_8: float = math.inf,
                       chunk_rows: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Memory-bounded AdaptQuant fake quantization without changing tile boundaries."""
    if values.ndim != 2 or values.shape[1] % BLOCK_VALUES:
        raise ValueError("Expected a 2-D operand with reduction dimension divisible by 32")
    if chunk_rows < TILE_ROWS or chunk_rows % TILE_ROWS:
        raise ValueError("chunk_rows must be a positive multiple of 16")
    original_rows = values.shape[0]
    padded_rows = (TILE_ROWS - original_rows % TILE_ROWS) % TILE_ROWS
    source = values
    if padded_rows:
        source = torch.cat((values, values.new_zeros(padded_rows, values.shape[1])), dim=0)
    output = torch.empty_like(source)
    counts = torch.zeros(3, dtype=torch.int64, device=values.device)
    for start in range(0, source.shape[0], chunk_rows):
        stop = min(start + chunk_rows, source.shape[0])
        quantized, formats = adaptquant_chunk(source[start:stop], threshold_4_to_6,
                                              threshold_6_to_8)
        output[start:stop].copy_(quantized)
        counts.add_(torch.bincount(formats.reshape(-1), minlength=3))
    return output[:original_rows], counts


def effective_bits(counts: torch.Tensor) -> Tuple[float, Dict[str, float]]:
    """Return assignment bits/value and the three format proportions.

    Per the TA's latest clarification, this includes each MX element payload
    and its E8M0 scale overhead, but excludes AdaptQuant's five-byte tile
    format-tag/offset metadata.
    """
    counts = counts.double().cpu()
    if counts.numel() != 3 or counts.sum() <= 0:
        raise ValueError("Expected non-empty [MXFP4, MXFP6, MXFP8] tile counts")
    shares = counts / counts.sum()
    names = ("mxfp4", "mxfp6", "mxfp8")
    bits = sum(shares[index].item() * FORMAT_BITS[name]
               for index, name in enumerate(names))
    return bits, {name: shares[index].item() for index, name in enumerate(names)}


def tile_relative_errors(source: torch.Tensor, column_weights: torch.Tensor,
                         reconstructed: torch.Tensor) -> torch.Tensor:
    """Per-tile normalized squared error used during offline policy derivation."""
    weights = column_weights.float()[None, :]
    error = (source.float() - reconstructed.float()).square() * weights
    signal = source.float().square() * weights
    numerator = as_tiles(error).sum(dim=(-1, -2))
    denominator = as_tiles(signal).sum(dim=(-1, -2))
    return numerator / denominator.clamp_min(1e-30)
