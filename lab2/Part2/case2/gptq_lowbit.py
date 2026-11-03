"""Case 2: packed GPTQ 2/3-bit PTQ. Reference: https://arxiv.org/abs/2210.17323."""

from linear_accounting import original_bytes

import bitsandbytes as bnb
import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


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


def pack_codes(codes, bits):
    if bits not in (2, 3):
        raise ValueError("Only 2 and 3-bit packing are supported")
    flat = codes.reshape(-1)
    count = 4 if bits == 2 else 8
    flat = F.pad(flat, (0, (-flat.numel()) % count))
    shifts = torch.arange(count, device=codes.device) * bits
    words = (flat.reshape(-1, count).int() << shifts).sum(1)
    if bits == 2:
        return words.byte()
    return torch.stack([words & 255, words >> 8 & 255, words >> 16 & 255], 1).byte().reshape(-1)


def unpack_codes(packed, count, bits):
    if bits == 2:
        words = packed.int()
        shifts = torch.arange(4, device=packed.device) * 2
    elif bits == 3:
        parts = packed.reshape(-1, 3).int()
        words = parts[:, 0] | parts[:, 1] << 8 | parts[:, 2] << 16
        shifts = torch.arange(8, device=packed.device) * 3
    else:
        raise ValueError("Only 2 and 3-bit packing are supported")
    return ((words[:, None] >> shifts) & ((1 << bits)-1)).reshape(-1)[:count].byte()


@triton.jit
def _dequantize_kernel(Packed, Scales, Zeros, Perm, Output,
                       COUNT: tl.constexpr, WIDTH: tl.constexpr,
                       GROUP: tl.constexpr, BITS: tl.constexpr,
                       PERMUTED: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < COUNT
    if BITS == 2:
        word = tl.load(Packed + i // 4, valid, 0).to(tl.int32)
        code = (word >> ((i % 4) * 2)) & 3
    else:
        pos = i // 8 * 3
        word = tl.load(Packed + pos, valid, 0).to(tl.int32)
        word |= tl.load(Packed + pos+1, valid, 0).to(tl.int32) << 8
        word |= tl.load(Packed + pos+2, valid, 0).to(tl.int32) << 16
        code = (word >> ((i % 8) * 3)) & 7
    scale = tl.load(Scales + i // GROUP, valid, 0).to(tl.float32)
    zero = tl.load(Zeros + i // GROUP, valid, 0).to(tl.float32)
    values = (code.to(tl.float32)-zero) * scale
    if PERMUTED:
        original_column = tl.load(Perm + i % WIDTH, valid, 0)
        output_i = i // WIDTH * WIDTH + original_column
    else:
        output_i = i
    tl.store(Output + output_i, values, valid)


class PackedLowBitLinear(nn.Module):
    def __init__(self, packed, scales, zeros, in_features, out_features,
                 group_size, bits, bias=None, permutation=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.bits = bits
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        self.register_buffer("bias", bias)
        self.register_buffer("permutation", permutation)

    def dequantize_reference(self, dtype=torch.float16):
        codes = unpack_codes(self.packed, self.in_features*self.out_features, self.bits)
        grouped = codes.reshape(self.out_features, -1, self.group_size).to(dtype)
        weight = ((grouped-self.zeros.to(dtype)[..., None])*self.scales.to(dtype)[..., None]).reshape(self.out_features, self.in_features)
        if self.permutation is not None:
            weight = weight[:, torch.argsort(self.permutation)]
        return weight

    def dequantize(self, dtype=torch.float16):
        if not self.packed.is_cuda or dtype != torch.float16:
            return self.dequantize_reference(dtype)
        out = torch.empty((self.out_features, self.in_features), device=self.packed.device, dtype=dtype)
        count = out.numel()
        _dequantize_kernel[(triton.cdiv(count, 1024),)](
            self.packed, self.scales, self.zeros,
            self.permutation if self.permutation is not None else self.packed,
            out, count, self.in_features, self.group_size, self.bits,
            self.permutation is not None, 1024)
        return out

    def forward(self, x):
        return F.linear(x, self.dequantize(x.dtype), self.bias)

    def spec(self):
        return dict(kind="gptq", bits=self.bits, in_features=self.in_features,
                    out_features=self.out_features, group_size=self.group_size,
                    bias=self.bias is not None, act_order=self.permutation is not None)


def candidate_module(payload, device):
    spec = payload["spec"]
    state = {k: v.to(device) for k, v in payload["state"].items()}
    return PackedLowBitLinear(state["packed"], state["scales"], state["zeros"],
                              spec["in_features"], spec["out_features"],
                              spec["group_size"], spec["bits"], state.get("bias"),
                              state.get("permutation"))


def affine_parameters(weight, importance, bits, ratios):
    max_code = (1 << bits)-1
    lo = weight.amin(1, keepdim=True).clamp_max(0)
    hi = weight.amax(1, keepdim=True).clamp_min(0)
    best_error = torch.full_like(lo, float("inf"))
    best_scale = torch.ones_like(lo)
    best_zero = torch.zeros_like(lo)
    for ratio in ratios:
        scale = ((hi-lo)*ratio/max_code).clamp_min(1e-7).half().float()
        zero = torch.round(-lo*ratio/scale).clamp(0, max_code)
        code = torch.round(weight/scale+zero).clamp(0, max_code)
        error = (((code-zero)*scale-weight).square()*importance).sum(1, keepdim=True)
        better = error < best_error
        best_error = torch.minimum(best_error, error)
        best_scale = torch.where(better, scale, best_scale)
        best_zero = torch.where(better, zero, best_zero)
    return best_scale, best_zero


@torch.no_grad()
def quantize_gptq(weight, gram, bits=3, group_size=256, damp=0.01,
                  act_order=True,
                  ratios=(1.0, .95, .9, .85, .8, .75, .7, .65, .6, .55, .5, .45, .4)):
    rows, columns = weight.shape
    if columns % group_size:
        raise ValueError(f"{columns=} must be divisible by {group_size=}")
    work = weight.float().clone()
    hessian = gram.float().clone()
    importance = hessian.diagonal().clone()
    dead = importance == 0
    work[:, dead] = 0
    hessian[dead, dead] = 1
    permutation = None
    if act_order:
        permutation = torch.argsort(importance, descending=True)
        work = work[:, permutation]
        hessian = hessian[permutation][:, permutation]
        importance = importance[permutation]
    diagonal = torch.arange(columns, device=weight.device)
    regularization = damp * hessian.diagonal().mean().clamp_min(1e-8)
    for attempt in range(5):
        candidate = hessian.clone()
        candidate[diagonal, diagonal] += regularization*(10**attempt)
        chol, info = torch.linalg.cholesky_ex(candidate)
        if info.item() == 0:
            inverse = torch.cholesky_inverse(chol)
            inverse_chol, info = torch.linalg.cholesky_ex(inverse, upper=True)
            if info.item() == 0:
                break
    else:
        raise RuntimeError("Cannot factor damped calibration Hessian")
    del hessian, inverse, chol, candidate
    codes = torch.empty((rows, columns), device=weight.device, dtype=torch.uint8)
    scales = torch.empty((rows, columns//group_size), device=weight.device, dtype=torch.float16)
    zeros = torch.empty_like(scales, dtype=torch.uint8)
    max_code = (1 << bits)-1
    for start in range(0, columns, group_size):
        end = start+group_size
        block = work[:, start:end].clone()
        scale, zero = affine_parameters(block, importance[None, start:end], bits, ratios)
        scales[:, start//group_size] = scale[:, 0].half()
        zeros[:, start//group_size] = zero[:, 0].byte()
        feedback = torch.empty_like(block)
        inv_block = inverse_chol[start:end, start:end]
        for col in range(group_size):
            current = block[:, col].clone()
            code = torch.round(current/scale[:, 0]+zero[:, 0]).clamp(0, max_code)
            reconstructed = (code-zero[:, 0])*scale[:, 0]
            codes[:, start+col] = code.byte()
            residual = (current-reconstructed)/inv_block[col, col]
            feedback[:, col] = residual
            block[:, col:] -= residual[:, None]*inv_block[None, col, col:]
        if end < columns:
            work[:, end:] -= feedback @ inverse_chol[start:end, end:]
    return PackedLowBitLinear(pack_codes(codes, bits), scales, zeros,
                              columns, rows, group_size, bits,
                              permutation=None if permutation is None else permutation.int())


@torch.no_grad()
def output_error(weight, candidate, inputs):
    reference = F.linear(inputs, weight).float()
    return ((candidate(inputs).float()-reference).square().sum()/reference.square().sum().clamp_min(1e-12)).item()


