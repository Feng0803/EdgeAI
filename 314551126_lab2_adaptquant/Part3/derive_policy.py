"""Derive AdaptQuant thresholds once on a non-target reference model.

The default grading path never calls this program.  It is submitted to make
the frozen policy's origin reproducible, as requested by the TA.  The solver
follows AdaptQuant's offline procedure: build tile-statistic/error histograms,
then find per-operand/per-projection MXFP4/MXFP6/MXFP8 thresholds whose layer
errors are bounded relative to NVFP4.  A single global reference-model slack
is increased only until the lab's W+A effective-bit budget is met.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import torch
from torch import nn

from mx_formats import (FORMAT_BITS, STATISTIC_MAX,
                        mx_fake_quant, nvfp4_fake_quant, tile_relative_errors,
                        tile_statistic)


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")
OPERANDS = ("weight", "activation")
FORMAT_NAMES = ("mxfp4", "mxfp6", "mxfp8")
GRID_STEP = 0.02
GRID = [round(index * GRID_STEP, 2)
        for index in range(int(round(STATISTIC_MAX / GRID_STEP)) + 1)]
CAP_INDEX = len(GRID)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class Histogram:
    """Tile count and summed relative error at every statistic-grid position."""

    def __init__(self, device: torch.device):
        self.values = torch.zeros(5, len(GRID), dtype=torch.float64, device=device)

    @torch.no_grad()
    def add(self, operand: torch.Tensor, column_weights: torch.Tensor) -> None:
        bins = torch.floor(tile_statistic(operand) / GRID_STEP + 1e-6)
        bins = bins.clamp(0, len(GRID) - 1).long().reshape(-1)
        self.values[0].add_(torch.bincount(bins, minlength=len(GRID)).double())
        for row, format_name in enumerate(FORMAT_NAMES, start=1):
            errors = tile_relative_errors(operand, column_weights,
                                          mx_fake_quant(operand, format_name)).reshape(-1)
            self.values[row].add_(torch.bincount(bins, weights=errors.double(),
                                                 minlength=len(GRID)))
        errors = tile_relative_errors(operand, column_weights,
                                      nvfp4_fake_quant(operand)).reshape(-1)
        self.values[4].add_(torch.bincount(bins, weights=errors.double(),
                                          minlength=len(GRID)))


def _threshold_value(index: int):
    return GRID[index] if index < CAP_INDEX else "CAP"


def solve_thresholds(per_layer_histograms: List[torch.Tensor], slack_db: float) -> dict:
    """Exhaustively solve the two thresholds on the paper's 0.02 grid."""
    histogram = torch.stack(per_layer_histograms).double().cpu()
    zero = torch.zeros(histogram.shape[0], histogram.shape[1], 1, dtype=torch.float64)
    prefix = torch.cat((zero, histogram.cumsum(dim=2)), dim=2)
    counts = prefix[:, 0].sum(dim=0)
    total_tiles = counts[CAP_INDEX]
    positions = torch.arange(CAP_INDEX + 1)
    first = positions[:, None]
    second = positions[None, :]
    ordered = first <= second

    average_bits = (
        FORMAT_BITS["mxfp4"] * counts[first]
        + FORMAT_BITS["mxfp6"] * (counts[second] - counts[first])
        + FORMAT_BITS["mxfp8"] * (total_tiles - counts[second])
    ) / total_tiles

    error4, error6, error8, nv_error = (prefix[:, index] for index in range(1, 5))
    candidate_error = (error4[:, first]
                       + error6[:, second] - error6[:, first]
                       + error8[:, CAP_INDEX][:, None, None] - error8[:, second])
    reference_error = nv_error[:, CAP_INDEX][:, None, None]
    maximum_ratio = 10 ** (slack_db / 10)
    feasible = ordered & (candidate_error <= reference_error * maximum_ratio).all(dim=0)
    if not feasible.any():
        raise RuntimeError("No threshold pair satisfies the NVFP4 error constraint")
    cost = torch.where(feasible, average_bits,
                       torch.full_like(average_bits, float("inf")))
    best_cost = cost.min()
    candidates = (cost <= best_cost + 1e-12).nonzero().tolist()
    # Deterministic tie break: keep more tiles in lower precision.
    first_index, second_index = max(candidates, key=lambda pair: (pair[0], pair[1]))
    fp6_tiles = counts[second_index] - counts[first_index]
    fp8_tiles = total_tiles - counts[second_index]
    return {
        "t_mxfp4_to_mxfp6": _threshold_value(first_index),
        "t_mxfp6_to_mxfp8": _threshold_value(second_index),
        "element_and_scale_bits": round(best_cost.item(), 6),
        "fp6_tile_share": round((fp6_tiles / total_tiles).item(), 6),
        "fp8_tile_share": round((fp8_tiles / total_tiles).item(), 6),
        "reference_layers": histogram.shape[0],
        "reference_tiles": int(total_tiles.item()),
    }


def solve_all(histograms: Dict[Tuple[str, str], List[torch.Tensor]], slack_db: float) -> dict:
    return {
        operand: {
            projection: solve_thresholds(histograms[(operand, projection)], slack_db)
            for projection in PROJECTIONS
        }
        for operand in OPERANDS
    }


def reference_effective_bits(policy: dict) -> dict:
    def weighted_mean(entries: dict) -> float:
        numerator = sum(row["element_and_scale_bits"] * row["reference_tiles"]
                        for row in entries.values())
        denominator = sum(row["reference_tiles"] for row in entries.values())
        # TA clarification: include MX scale overhead (already in
        # element_and_scale_bits), but exclude five-byte tile metadata.
        return numerator / denominator

    weight_bits = weighted_mean(policy["weight"])
    activation_bits = weighted_mean(policy["activation"])
    return {
        "weight": weight_bits,
        "activation": activation_bits,
        "realized_effective_bit_rate": (weight_bits + activation_bits) / 2,
    }


def solve_to_budget(histograms, bit_limit: float, slack_step_db: float = 0.1) -> Tuple[dict, float, dict]:
    step = 0
    while True:
        slack_db = round(step * slack_step_db, 6)
        policy = solve_all(histograms, slack_db)
        bits = reference_effective_bits(policy)
        if bits["realized_effective_bit_rate"] <= bit_limit:
            return policy, slack_db, bits
        step += 1
        if slack_db >= 12:
            raise RuntimeError("Unable to satisfy the requested effective-bit budget")


class CaptureInput(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.attention_type = getattr(layer, "attention_type", None)
        self.hidden_states = None
        self.keyword_arguments = None

    def forward(self, hidden_states, **keyword_arguments):
        self.hidden_states = hidden_states
        self.keyword_arguments = keyword_arguments
        raise CaptureComplete


class CaptureComplete(Exception):
    pass


def projection_modules(layer) -> dict:
    return {
        name: getattr(layer.self_attn if name.endswith("_proj") and name in PROJECTIONS[:4]
                      else layer.mlp, name)
        for name in PROJECTIONS
    }


@torch.no_grad()
def collect_reference_histograms(model_id: str, revision: str | None,
                                 sequence_length: int, device: str):
    """One reference-model forward pass; target Qwen3-4B is never touched."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True)
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    corpus = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(corpus["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids[:, :sequence_length].to(device)

    decoder = model.model
    decoder.embed_tokens.to(device)
    decoder.rotary_emb.to(device)
    catcher = CaptureInput(decoder.layers[0])
    decoder.layers[0] = catcher
    try:
        model(tokens)
    except CaptureComplete:
        pass
    finally:
        decoder.layers[0] = catcher.layer
    hidden_states = catcher.hidden_states
    kwargs = {key: value for key, value in catcher.keyword_arguments.items()
              if key not in ("past_key_values", "use_cache")}
    histograms = defaultdict(list)

    for layer_index, layer in enumerate(decoder.layers):
        layer.to(device)
        linears = projection_modules(layer)
        input_second_moment = {}
        handles = []

        def make_hook(name, linear):
            output_sensitivity = linear.weight.float().square().sum(dim=0)

            def hook(unused_module, inputs):
                activation = inputs[0].reshape(-1, inputs[0].shape[-1])
                input_second_moment[name] = activation.float().square().mean(dim=0)
                histogram = Histogram(torch.device(device))
                histogram.add(activation, output_sensitivity)
                histograms[("activation", name)].append(histogram.values.cpu())
            return hook

        for name, linear in linears.items():
            handles.append(linear.register_forward_pre_hook(make_hook(name, linear)))
        result = layer(hidden_states, **kwargs)
        hidden_states = result[0] if isinstance(result, tuple) else result
        for handle in handles:
            handle.remove()
        for name, linear in linears.items():
            histogram = Histogram(torch.device(device))
            histogram.add(linear.weight.data, input_second_moment[name])
            histograms[("weight", name)].append(histogram.values.cpu())
        layer.to("cpu")
        torch.cuda.empty_cache()
        log(f"[derive] reference layer {layer_index + 1}/{len(decoder.layers)} complete")
    return dict(histograms)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-model", default="unsloth/Llama-3.1-8B")
    parser.add_argument("--reference-revision",
                        default="3f0d51f8e5640f98f1a96ea9044a0e55c0a83814")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bit-limit", type=float, default=5.17,
                        help="reference-model target below the assignment limit 5.3")
    parser.add_argument("--histogram-cache", type=Path, default=Path("reference_histograms.pt"))
    parser.add_argument("--output", type=Path, default=Path("frozen_policy.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.histogram_cache.exists():
        serialized = torch.load(args.histogram_cache, weights_only=True)
        histograms = {tuple(key.split("/", 1)): list(value) for key, value in serialized.items()}
        log(f"[derive] loaded {args.histogram_cache}")
    else:
        histograms = collect_reference_histograms(
            args.reference_model, args.reference_revision, args.sequence_length, args.device)
        serialized = {"/".join(key): torch.stack(value) for key, value in histograms.items()}
        torch.save(serialized, args.histogram_cache)
        log(f"[derive] saved {args.histogram_cache}")

    policy, slack_db, bits = solve_to_budget(histograms, args.bit_limit)
    output = {
        "method": "AdaptQuant reproduction with lab bit-budget adjustment",
        "paper": "https://openreview.net/forum?id=ZGeR4gkEZq",
        "reference_model": args.reference_model,
        "reference_revision": args.reference_revision,
        "reference_data": f"WikiText-2 train, first {args.sequence_length} tokens",
        "target_model_data_used": False,
        "statistic": "log2(tile_absmax / tile_rms)",
        "tile_shape": [16, 32],
        "grid_step": GRID_STEP,
        "reference_nvfp4_slack_db": slack_db,
        "reference_bit_target": args.bit_limit,
        "scale_overhead_counted": True,
        "five_byte_tile_metadata_counted": False,
        "bit_accounting_rule": "TA clarification: MX scale overhead included; five-byte tile metadata excluded",
        "reference_realized_bits": bits,
        "weight": policy["weight"],
        "activation": policy["activation"],
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    log(f"[derive] wrote frozen policy to {args.output}")
    log(f"[derive] reference effective bit rate: {bits['realized_effective_bit_rate']:.6f}")


if __name__ == "__main__":
    main()
