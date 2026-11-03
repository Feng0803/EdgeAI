"""Physical MLP-channel pruning and closed-form output error compensation.

No gradients or optimizer. Every removed channel deletes gate/up rows and a
down column. The compensation is a damped least-squares projection using TRAIN.
"""

from linear_accounting import pruning_counts

import torch
from torch import nn


def set_module(root, name, module):
    parent, _, attr = name.rpartition('.')
    setattr(root.get_submodule(parent) if parent else root, attr, module)


def linear_from_weight(weight, bias=None):
    out_features, in_features = weight.shape
    module = nn.Linear(in_features, out_features, bias=bias is not None, device='meta', dtype=weight.dtype)
    module.weight = nn.Parameter(weight.detach().contiguous().clone(), requires_grad=False)
    if bias is not None:
        module.bias = nn.Parameter(bias.detach().to(weight.device).contiguous().clone(), requires_grad=False)
    return module


@torch.no_grad()
def channel_importance(down_weight, gram):
    return gram.diagonal().clamp_min(0)*down_weight.float().square().sum(0)


def keep_channels(scores, width):
    if not 0 < width <= scores.numel():
        raise ValueError('Invalid retained MLP width')
    return torch.topk(scores, width, sorted=False).indices.sort().values


@torch.no_grad()
def compensated_down_weight(weight, gram, keep, damp=.01):
    """D_keep + D_drop H_drop,keep (H_keep,keep + regularizer)^-1.

    Solve in unit-variance coordinates so a large activation outlier does not
    set the damping level for every other channel. The penalty applies to the
    update relative to the retained original weights, rather than to D itself.
    """
    keep = keep.long()
    remaining = torch.ones(weight.shape[1], device=weight.device, dtype=torch.bool)
    remaining[keep] = False
    drop = remaining.nonzero().flatten()
    retained = weight[:, keep].float()
    if not drop.numel():
        return retained.to(weight.dtype).contiguous()
    hkk = gram[keep][:, keep].float()
    hkk = (hkk+hkk.T)*.5
    std = hkk.diagonal().clamp_min(1e-12).sqrt()
    correlation = hkk/std[:, None]/std[None, :]
    cross = (weight[:, drop].float() @ gram[drop][:, keep].float())/std[None]
    for attempt in range(5):
        matrix = correlation.clone()
        matrix.diagonal().add_(damp*(10**attempt))
        chol, info = torch.linalg.cholesky_ex(matrix)
        if info.item() == 0:
            break
    else:
        raise RuntimeError('Cannot factor normalized pruning Gram matrix')
    delta = torch.cholesky_solve(cross.T.contiguous(), chol).T/std[None]
    result = (retained+delta).to(weight.dtype).contiguous()
    if not torch.isfinite(result).all():
        raise ValueError('Non-finite reconstructed down weights')
    return result


@torch.no_grad()
def cropped_mlp_modules(mlp, keep, down_weight=None):
    gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    gate_bias = None if gate.bias is None else gate.bias[keep]
    up_bias = None if up.bias is None else up.bias[keep]
    return dict(gate_proj=linear_from_weight(gate.weight[keep], gate_bias),
                up_proj=linear_from_weight(up.weight[keep], up_bias),
                down_proj=linear_from_weight(down.weight[:, keep] if down_weight is None else down_weight, down.bias))


def install_mlp_modules(mlp, modules):
    for name, module in modules.items():
        setattr(mlp, name, module)
    mlp.intermediate_size = mlp.gate_proj.out_features


