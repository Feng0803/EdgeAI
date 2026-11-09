"""Apply the frozen AdaptQuant policy to Qwen3-4B and run the official PPL test."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from mx_formats import adaptquant_operand, effective_bits


MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")
HERE = Path(__file__).resolve().parent
ASSIGNMENT_BIT_LIMIT = 5.3


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("adaptquant", "fp16"), default="adaptquant")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy", type=Path, default=HERE / "frozen_policy.json")
    parser.add_argument("--chunk-rows", type=int, default=256,
                        help="memory-only execution chunk; must be a multiple of 16")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "adaptquant_result.json")
    args = parser.parse_args()
    if args.chunk_rows < 16 or args.chunk_rows % 16:
        parser.error("--chunk-rows must be a positive multiple of 16")
    return args


def threshold(row: dict, key: str) -> float:
    value = row[key]
    return math.inf if value == "CAP" else float(value)


def policy_thresholds(policy: dict, operand: str, projection: str):
    row = policy[operand][projection]
    return (threshold(row, "t_mxfp4_to_mxfp6"),
            threshold(row, "t_mxfp6_to_mxfp8"))


def target_linears(model):
    rows = []
    for name, module in model.named_modules():
        if name.startswith("model.layers.") and isinstance(module, nn.Linear):
            projection = name.rsplit(".", 1)[-1]
            if projection not in PROJECTIONS:
                raise AssertionError(f"Unexpected decoder Linear: {name}")
            rows.append((name, projection, module))
    if len(rows) != 252:
        raise AssertionError(f"Expected 252 decoder Linear layers, found {len(rows)}")
    return rows


def tied_embedding_and_head(model) -> bool:
    return (model.get_input_embeddings().weight.data_ptr()
            == model.get_output_embeddings().weight.data_ptr())


@torch.no_grad()
def apply_frozen_policy(model, policy: dict, chunk_rows: int):
    """Quantize all target weights before any dataset is opened; install A hooks."""
    if not tied_embedding_and_head(model):
        raise AssertionError("Expected Qwen3-4B input embedding and lm_head to be tied")
    shared_weight = model.get_output_embeddings().weight
    shared_version = shared_weight._version
    device = next(model.parameters()).device
    counts = {
        "weight": torch.zeros(3, dtype=torch.int64),
        "activation": torch.zeros(3, dtype=torch.int64, device=device),
    }
    handles = []
    started = time.monotonic()

    for index, (name, projection, linear) in enumerate(target_linears(model), start=1):
        weight_thresholds = policy_thresholds(policy, "weight", projection)
        quantized, local_counts = adaptquant_operand(
            linear.weight.data, *weight_thresholds, chunk_rows=chunk_rows)
        linear.weight.data.copy_(quantized)
        counts["weight"].add_(local_counts.cpu())
        del quantized, local_counts

        activation_thresholds = policy_thresholds(policy, "activation", projection)

        def make_hook(thresholds):
            def hook(unused_module, inputs):
                if not inputs:
                    raise RuntimeError("Linear activation hook received no operand")
                source = inputs[0]
                operand = source.reshape(-1, source.shape[-1])
                reconstructed, format_counts = adaptquant_operand(
                    operand, *thresholds, chunk_rows=chunk_rows)
                counts["activation"].add_(format_counts)
                return (reconstructed.reshape_as(source), *inputs[1:])
            return hook

        handles.append(linear.register_forward_pre_hook(make_hook(activation_thresholds)))
        if index == 1 or index % 7 == 0:
            log(f"[quantize] {index}/252 {name} ({time.monotonic() - started:.1f}s)")

    if shared_weight._version != shared_version or not tied_embedding_and_head(model):
        raise AssertionError("Excluded tied embedding/lm_head was modified")
    model._adaptquant_handles = handles
    return counts, time.monotonic() - started


def performance_score(ppl: float) -> int:
    for upper, score in ((15, 15), (16, 12), (17, 9), (18, 6), (19, 3)):
        if ppl <= upper:
            return score
    return 0


def load_model(device: str):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float16,
        device_map={"": device})
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    return model


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the assignment evaluation")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.parent.mkdir(parents=True, exist_ok=True)

    policy = None
    if args.mode == "adaptquant":
        policy = json.loads(args.policy.read_text())
        required_keys = {"weight", "activation", "reference_model", "target_model_data_used"}
        if not required_keys.issubset(policy) or policy["target_model_data_used"]:
            raise ValueError("The supplied file is not a frozen target-calibration-free policy")
        log(f"[policy] frozen before target load; sha256={sha256(args.policy)}")
        log(f"[policy] source={policy['reference_model']}; target-model data used=false")

    # No target-model dataset or forward has occurred before this point.
    model = load_model(args.device)
    manifest = {
        "created_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "mode": args.mode,
        "actual_execution": "T4 fake quantization" if args.mode == "adaptquant" else "FP16",
        "dataset_access_during_policy_selection": False,
        "target_forward_during_policy_selection": False,
        "policy_sha256": sha256(args.policy) if policy else None,
    }

    if args.mode == "adaptquant":
        log("[quantize] AdaptQuant 16x32 per-tile MX selection; W+A; no target calibration")
        counts, quantization_seconds = apply_frozen_policy(model, policy, args.chunk_rows)
        weight_bits, weight_shares = effective_bits(counts["weight"])
        manifest.update(
            quantized_linear_count=252,
            quantization_seconds=quantization_seconds,
            weight_tile_counts=counts["weight"].tolist(),
            weight_tile_shares=weight_shares,
            weight_effective_bits=weight_bits,
            embedding_lm_head_norm_excluded=True,
            scale_overhead_included=True,
            format_index_overhead_included=False,
            format_index_overhead_rule="TA clarification: five-byte tile metadata is excluded",
        )
        log(f"[quantize] weight effective bits={weight_bits:.6f}; shares={weight_shares}")
    else:
        counts = None

    if args.skip_eval:
        args.output.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return

    # Importing the provided evaluator is deliberately delayed until all
    # policy choices and target weights are fixed.
    import evaluate

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    log("[eval] provided evaluate.py; WikiText-2 TEST; seqlen=2048")
    evaluation_started = time.monotonic()
    evaluated_segments = 0

    def evaluation_progress(unused_module, unused_inputs, unused_output):
        nonlocal evaluated_segments
        evaluated_segments += 1
        if evaluated_segments % 16 == 0:
            log(f"[eval] {evaluated_segments} segments "
                f"({time.monotonic() - evaluation_started:.1f}s)")

    progress_handle = model.register_forward_hook(evaluation_progress)
    try:
        ppl = evaluate.eval_ppl_wikitext2(model, tokenizer, seqlen=2048)
    finally:
        progress_handle.remove()
    manifest.update(ppl=ppl, evaluation_seconds=time.monotonic() - evaluation_started,
                    evaluation_segments=evaluated_segments,
                    evaluator_sha256=sha256(HERE / "evaluate.py"))

    if args.mode == "adaptquant":
        activation_bits, activation_shares = effective_bits(counts["activation"])
        realized = (weight_bits + activation_bits) / 2
        manifest.update(
            activation_tile_counts=counts["activation"].cpu().tolist(),
            activation_tile_shares=activation_shares,
            activation_effective_bits=activation_bits,
            realized_effective_bit_rate=realized,
            effective_bit_formula="(weight_effective_bits + activation_effective_bits) / 2",
            assignment_limit=ASSIGNMENT_BIT_LIMIT,
            requirement_met=realized <= ASSIGNMENT_BIT_LIMIT,
            performance_score=performance_score(ppl),
        )
        if realized > ASSIGNMENT_BIT_LIMIT:
            raise AssertionError(f"Effective bit rate {realized} exceeds {ASSIGNMENT_BIT_LIMIT}")
        print("Part3 AdaptQuant reproduction")
        print(f"Weight Effective Bits: {weight_bits:.6f}")
        print(f"Activation Effective Bits: {activation_bits:.6f}")
        print(f"Realized Effective Bit Rate: {realized:.6f}")
    else:
        print("Part3 FP16 baseline")
    print(f"PPL: {ppl:.4f}", flush=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
