"""Activation-weighted low-rank compression and real INT8/FP16 row storage.

For H=E[x x.T], the retained left singular vectors of W H^(1/2)
minimize output reconstruction error. We obtain them via the smaller Gram
eigendecomposition. Reference: https://arxiv.org/abs/2403.07378.
"""

from linear_accounting import original_bytes

import math

import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl


def tensor_bytes(tensor):
    return tensor.numel()*tensor.element_size()


def module_payload(module):
    return sum(tensor_bytes(t) for t in module.state_dict().values())


def target_linear_specs(model):
    return {name: dict(in_features=m.in_features, out_features=m.out_features,
                       bias=m.bias is not None)
            for name, m in model.named_modules() if isinstance(m, nn.Linear)}




def set_module(model, name, module):
    parent, _, attr = name.rpartition('.')
    setattr(model.get_submodule(parent) if parent else model, attr, module)


@triton.jit
def _int8_rows_kernel(Q, S, Tail, Out, COUNT: tl.constexpr, WIDTH: tl.constexpr,
                      QROWS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = i < COUNT
    quantized = i//WIDTH < QROWS
    code = tl.load(Q+i, valid & quantized, 0).to(tl.float32)
    scale = tl.load(S+i//WIDTH, valid & quantized, 0).to(tl.float32)
    tail = tl.load(Tail+i-QROWS*WIDTH, valid & ~quantized, 0)
    value = tl.where(quantized, code*scale, tail.to(tl.float32))
    tl.store(Out+i, value, valid)


class RowMixedLinear(nn.Module):
    def __init__(self, qweight, scales, weight16, bias=None):
        super().__init__()
        self.in_features = weight16.shape[1]
        self.out_features = qweight.shape[0]+weight16.shape[0]
        # Eigensolvers can produce column-major factors. The fused kernel reads
        # row-major buffers, including before the checkpoint normalizes layouts.
        self.register_buffer('qweight', qweight.contiguous())
        self.register_buffer('scales', scales.contiguous())
        self.register_buffer('weight16', weight16.contiguous())
        self.register_buffer('bias', bias)

    @classmethod
    def from_fp16(cls, weight, bias=None):
        return cls(torch.empty((0, weight.shape[1]), dtype=torch.int8, device=weight.device),
                   torch.empty((0, 1), dtype=torch.float16, device=weight.device),
                   weight.detach().clone(), None if bias is None else bias.detach().clone())

    def spec(self):
        return dict(kind='rowmixed', in_features=self.in_features, out_features=self.out_features,
                    qrows=self.qweight.shape[0], bias=self.bias is not None)

    def dequantize(self, dtype=torch.float16):
        if not self.qweight.shape[0]:
            return self.weight16.to(dtype)
        if not self.qweight.is_cuda or dtype != torch.float16:
            return torch.cat([self.qweight.to(dtype)*self.scales.to(dtype), self.weight16.to(dtype)])
        output = torch.empty((self.out_features, self.in_features), device=self.qweight.device, dtype=dtype)
        _int8_rows_kernel[(triton.cdiv(output.numel(), 1024),)](
            self.qweight, self.scales, self.weight16, output, output.numel(),
            self.in_features, self.qweight.shape[0], 1024)
        return output

    def forward(self, x):
        return F.linear(x, self.dequantize(x.dtype), self.bias)


class LowRankLinear(nn.Module):
    def __init__(self, left, right, bias=None):
        super().__init__()
        self.left = left
        self.right = right
        self.in_features = right.in_features
        self.out_features = left.out_features
        self.rank = right.out_features
        self.register_buffer('bias', bias)

    def forward(self, x):
        output = self.left(self.right(x))
        return output if self.bias is None else output+self.bias

    def spec(self):
        return dict(kind='lowrank', in_features=self.in_features, out_features=self.out_features,
                    rank=self.rank, left=self.left.spec(), right=self.right.spec(),
                    bias=self.bias is not None)


@torch.no_grad()
def int8_candidate(weight):
    """Per-output-row symmetric INT8, with exactly the stored FP16 scale."""
    scale = (weight.float().abs().amax(1, keepdim=True)/127).clamp_min(1e-7).half()
    codes = torch.round(weight.float()/scale.float()).clamp(-127, 127).to(torch.int8)
    return codes, scale


@torch.no_grad()
def apply_int8_rows(module, qweight, scales, rows):
    if module.qweight.shape[0]:
        raise ValueError('Expected unquantized factor')
    return RowMixedLinear(qweight[:rows].to(module.weight16.device).clone(),
                           scales[:rows].to(module.weight16.device).clone(),
                           module.weight16[rows:].clone(), module.bias)


@torch.no_grad()
def weighted_svd_basis(weight, gram, damp=1e-4, maximum_rank=None):
    """Return balanced factors for the maximum useful rank, plus output spectrum.

    A @ B = U @ U.T @ W, where U contains the leading left singular vectors
    of W @ chol(H). No gradient training and no calibration transform at runtime.
    """
    w = weight.float()
    m, n = w.shape
    h = gram.float().clone()
    h = (h+h.T)*.5
    h.diagonal().add_(damp*h.diagonal().mean().clamp_min(1e-8))
    k = min(m, n)
    if m <= n:
        covariance = (w @ h) @ w.T
        covariance = (covariance+covariance.T)*.5
        eigenvalues, vectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.flip(0).clamp_min(0)
        u = vectors.flip(1)
    else:
        chol = torch.linalg.cholesky(h)
        whitened = w @ chol
        covariance = whitened.T @ whitened
        covariance = (covariance+covariance.T)*.5
        eigenvalues, vectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.flip(0).clamp_min(0)
        keep = k if maximum_rank is None else min(k, maximum_rank)
        v = vectors.flip(1)[:, :keep]
        sigma = eigenvalues[:keep].sqrt().clamp_min(1e-12)
        u = (whitened @ v)/sigma[None]
        # Squaring the condition number can lose small components in FP32.
        # Fall back to direct SVD when Gram-derived vectors lose normalization.
        normalization_error = (u.square().sum(0)-1).abs().max().item()
        if not math.isfinite(normalization_error) or normalization_error > .05:
            u, singular, _ = torch.linalg.svd(whitened, full_matrices=False,
                                              driver='gesvd' if w.is_cuda else None)
            eigenvalues = singular.square()
    keep = k if maximum_rank is None else min(k, maximum_rank)
    u = u[:, :keep]
    # Balanced factors reduce intermediate dynamic range before INT8 storage.
    unscaled_right = u.T @ w
    balance = (unscaled_right.norm(dim=1)/u.norm(dim=0).clamp_min(1e-12)).clamp_min(1e-16).sqrt()
    left = (u*balance[None]).half()
    right = (unscaled_right/balance[:, None]).half()
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError('Non-finite low-rank factors')
    return left, right, eigenvalues


def lowrank_from_basis(payload, rank, device, bias=None):
    left = payload['left'][:, :rank].to(device).clone()
    right = payload['right'][:rank].to(device).clone()
    return LowRankLinear(RowMixedLinear.from_fp16(left), RowMixedLinear.from_fp16(right), bias)


def bounded_two_coin(a, b, target, max_a, max_b):
    """Solve a*x+b*y=target exactly, with bounded nonnegative row counts."""
    g = math.gcd(a, b)
    if target % g:
        return None
    aa, bb, tt = a//g, b//g, target//g
    x0 = (tt*pow(aa, -1, bb)) % bb
    y0 = (tt-aa*x0)//bb
    low = max((-x0+bb-1)//bb, (y0-max_b+aa-1)//aa)
    high = min((max_a-x0)//bb, y0//aa)
    if low > high:
        return None
    # Centered solution avoids extreme concentrations in a reserve component.
    step = (low+high)//2
    return x0+step*bb, y0-step*aa


def allocate_int8_rows(components, saving_target, reserve_names):
    """Allocate actual INT8 rows; reserve two coprime gains for exact-byte closure.

    A converted row saves in_features bytes, then costs 2 bytes for its scale.
    Nothing is added merely to occupy space; all tensors participate in Linear.
    """
    by_name = {c['name']: c for c in components}
    reserves = [by_name[name] for name in reserve_names]
    a, b = [c['in_features']-2 for c in reserves]
    max_a, max_b = [c['out_features'] for c in reserves]
    reserve_capacity = a*max_a+b*max_b
    desired_reserve = min(reserve_capacity//2, max(0, saving_target))
    assigned = {c['name']: 0 for c in components}
    remaining = saving_target
    for component in sorted((c for c in components if c['name'] not in reserve_names), key=lambda c: c['score']):
        gain = component['in_features']-2
        rows = min(component['out_features'], max(0, (remaining-desired_reserve)//gain))
        assigned[component['name']] = rows
        remaining -= rows*gain
    solution = bounded_two_coin(a, b, remaining, max_a, max_b)
    if solution is None:
        raise ValueError(f'No exact byte allocation for reserve remainder {remaining}')
    assigned[reserve_names[0]], assigned[reserve_names[1]] = solution
    actual = sum(assigned[c['name']]*(c['in_features']-2) for c in components)
    if actual != saving_target:
        raise AssertionError(f'Exact-byte allocation failed: {actual} != {saving_target}')
    return assigned
