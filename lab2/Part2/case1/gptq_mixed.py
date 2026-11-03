"""Packed weight-only PTQ primitives for the Lab 2 Case 1 experiment.

GPTQ uses the damped inverse input Gram matrix to compensate rounding error.
Algorithm reference: Frantar et al., https://arxiv.org/abs/2210.17323.
This implementation uses contiguous groups, affine 3-bit codes, FP16 scales,
uint8 zero points, and no activation permutation or inference weight cache.
"""

from linear_accounting import original_bytes

import math

import bitsandbytes as bnb
import torch
from torch import nn
from torch.nn import functional as F


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def pack_3bit(codes):
    """Eight unsigned 3-bit codes occupy exactly three bytes (zero padded)."""
    flat = codes.reshape(-1)
    if flat.numel() % 8:
        flat = F.pad(flat, (0, 8 - flat.numel() % 8))
    values = flat.reshape(-1, 8).to(torch.int32)
    shifts = torch.arange(8, device=values.device) * 3
    words = (values << shifts).sum(1)
    return torch.stack([words & 255, (words >> 8) & 255, (words >> 16) & 255], 1).to(torch.uint8).reshape(-1)


def unpack_3bit(packed, count):
    values = packed.reshape(-1, 3).to(torch.int32)
    words = values[:, 0] | (values[:, 1] << 8) | (values[:, 2] << 16)
    shifts = torch.arange(8, device=packed.device) * 3
    return ((words[:, None] >> shifts) & 7).reshape(-1)[:count].to(torch.uint8)


class Packed3BitLinear(nn.Module):
    def __init__(self, packed, scales, zeros, in_features, out_features, group_size, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        self.register_buffer("bias", bias)

    def dequantize(self, dtype=torch.float16):
        codes = unpack_3bit(self.packed, self.in_features * self.out_features)
        codes = codes.reshape(self.out_features, -1, self.group_size).to(dtype)
        return ((codes - self.zeros.to(dtype).unsqueeze(-1)) * self.scales.to(dtype).unsqueeze(-1)).reshape(self.out_features, self.in_features)

    def forward(self, x):
        # Only this layer is reconstructed; no persistent FP16 copy is retained.
        return F.linear(x, self.dequantize(x.dtype), self.bias)

    def spec(self):
        return dict(kind="gptq3", in_features=self.in_features,
                    out_features=self.out_features, group_size=self.group_size,
                    bias=self.bias is not None)


class SharedNF4Embedding(nn.Module):
    """Embedding lookup uses the packed lm_head parameter without registering twice."""
    def __init__(self, head):
        super().__init__()
        object.__setattr__(self, "_head", head)
        self.num_embeddings = head.out_features
        self.embedding_dim = head.in_features

    def forward(self, ids):
        weight = self._head.weight
        dense = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        return F.embedding(ids, dense)


def set_module(root, name, module):
    parent_name, _, attr = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, attr, module)


def module_payload(module):
    # Includes bnb's quantization state, codebooks and serialized metadata.
    return sum(tensor_bytes(t) for t in module.state_dict().values())


def target_linear_specs(model):
    return {name: dict(in_features=mod.in_features, out_features=mod.out_features,
                       bias=mod.bias is not None)
            for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}




@torch.no_grad()
def install_shared_nf4(model):
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if embed.weight.data_ptr() != head.weight.data_ptr():
        raise ValueError("Expected tied Qwen3 input/output weights")
    packed_head = bnb.nn.Linear4bit(head.in_features, head.out_features,
                                   bias=False, compute_dtype=torch.float16,
                                   compress_statistics=True, quant_type="nf4")
    packed_head.weight = bnb.nn.Params4bit(head.weight.detach(), requires_grad=False,
                                         compress_statistics=True, quant_type="nf4")
    packed_head = packed_head.to(head.weight.device)
    model.set_output_embeddings(packed_head)
    model.set_input_embeddings(SharedNF4Embedding(packed_head))


def affine_parameters(weight, importance, ratios):
    """Search scales on activation-weighted weight error; store exact FP16 scales."""
    lo = weight.amin(1, keepdim=True).clamp_max(0)
    hi = weight.amax(1, keepdim=True).clamp_min(0)
    best = torch.full_like(lo, float("inf"))
    best_scale = torch.ones_like(lo)
    best_zero = torch.zeros_like(lo)
    for ratio in ratios:
        scale = ((hi - lo) * ratio / 7).clamp_min(1e-7).half().float()
        zero = torch.round(-lo * ratio / scale).clamp(0, 7)
        code = torch.round(weight / scale + zero).clamp(0, 7)
        error = (((code - zero) * scale - weight).square() * importance).sum(1, keepdim=True)
        better = error < best
        best = torch.minimum(best, error)
        best_scale = torch.where(better, scale, best_scale)
        best_zero = torch.where(better, zero, best_zero)
    return best_scale, best_zero


@torch.no_grad()
def quantize_gptq3(weight, gram, group_size=128, damp=0.01,
                   ratios=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5)):
    """Block GPTQ error feedback, with group parameters updated after prior blocks.

    gram is X.T @ X / number_of_tokens. No gradients, labels, or optimizer.
    Returned module contains packed codes, rather than fake quantized weights.
    """
    rows, columns = weight.shape
    if columns % group_size:
        raise ValueError(f"{columns=} must be divisible by {group_size=}")
    work = weight.float().clone()
    hessian = gram.float().clone()
    importance = hessian.diagonal().clone()
    dead = importance == 0
    work[:, dead] = 0
    hessian[dead, dead] = 1
    diagonal = torch.arange(columns, device=weight.device)
    regularization = damp * hessian.diagonal().mean().clamp_min(1e-8)
    # Escalation only addresses numerical conditioning, never looks at test PPL.
    for attempt in range(5):
        candidate = hessian.clone()
        candidate[diagonal, diagonal] += regularization * (10 ** attempt)
        chol, info = torch.linalg.cholesky_ex(candidate)
        if info.item() == 0:
            inverse = torch.cholesky_inverse(chol)
            inverse_chol, info = torch.linalg.cholesky_ex(inverse, upper=True)
            if info.item() == 0:
                break
    else:
        raise RuntimeError("Cannot factor damped calibration Hessian")
    del hessian, inverse, chol, candidate
    codes = torch.empty((rows, columns), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((rows, columns // group_size), dtype=torch.float16, device=weight.device)
    zeros = torch.empty_like(scales, dtype=torch.uint8)
    for start in range(0, columns, group_size):
        end = start + group_size
        block = work[:, start:end].clone()
        scale, zero = affine_parameters(block, importance[start:end][None], ratios)
        scales[:, start // group_size] = scale.squeeze(1).half()
        zeros[:, start // group_size] = zero.squeeze(1).byte()
        feedback = torch.empty_like(block)
        inv_block = inverse_chol[start:end, start:end]
        for column in range(group_size):
            current = block[:, column].clone()
            code = torch.round(current / scale[:, 0] + zero[:, 0]).clamp(0, 7)
            reconstructed = (code - zero[:, 0]) * scale[:, 0]
            codes[:, start + column] = code.byte()
            residual = (current - reconstructed) / inv_block[column, column]
            feedback[:, column] = residual
            block[:, column:] -= residual[:, None] * inv_block[column, column:][None]
        if end < columns:
            work[:, end:] -= feedback @ inverse_chol[start:end, end:]
    return Packed3BitLinear(pack_3bit(codes), scales, zeros, columns, rows, group_size)


@torch.no_grad()
def reconstruction_errors(weight, candidate, inputs):
    """Held-out calibration output errors relative to the original layer output."""
    reference = F.linear(inputs, weight).float()
    denominator = reference.square().sum().clamp_min(1e-12)
    q4, state = bnb.functional.quantize_4bit(weight.contiguous(), blocksize=64,
                                          compress_statistics=True, quant_type="nf4")
    dense4 = bnb.functional.dequantize_4bit(q4, state)
    error4 = (F.linear(inputs, dense4).float() - reference).square().sum() / denominator
    error3 = (candidate(inputs).float() - reference).square().sum() / denominator
    return error3.item(), error4.item()
