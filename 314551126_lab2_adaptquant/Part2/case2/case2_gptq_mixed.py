"""Case 2: packed GPTQ 2/3-bit allocation, selected on WikiText TRAIN only."""

import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import runpy
import sys
import time

import bitsandbytes as bnb
from datasets import Dataset, load_dataset
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import init_empty_weights

import evaluate
from linear_accounting import (accounting_metadata, checkpoint_audit,
                               require_current_scope, target_payload)
from gptq_lowbit import (PackedLowBitLinear, SharedNF4Embedding, candidate_module,
                         install_shared_nf4, module_payload, original_bytes,
                         output_error, quantize_gptq, set_module, target_linear_specs)


MODEL_REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
FROZEN_MANIFEST = Path(__file__).resolve().parents[1] / 'frozen' / 'case2_manifest.json'

def log(message):
    print(message, file=sys.stderr, flush=True)


def load_frozen_manifest(args):
    manifest = json.loads(FROZEN_MANIFEST.read_text())
    if manifest['model'] != args.model or manifest['calibration']['revision'] != MODEL_REVISION:
        raise ValueError('Frozen Case 2 allocation does not match the requested model revision')
    return manifest


def checkpoint_dir(args):
    return args.work_dir / args.mode


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=["baseline3", "mixed"], default="mixed")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--holdout-samples", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=1024, help="calibration only; official PPL remains 2048")
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--damp", type=float, default=.01)
    parser.add_argument("--act-order", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-rate", type=float, default=.8187)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).parent / "artifacts/g256")
    parser.add_argument("--max-layers", type=int, default=None, help="smoke test only")
    parser.add_argument("--score", choices=["block", "projection"], default="block")
    parser.add_argument("--protect", nargs="*", default=[], help="decoder Linear names to retain at NF4")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--search", action="store_true",
                        help="rerun all INT2/INT3 candidates instead of the submitted allocation")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.holdout_samples < 1 or args.seqlen < 2:
        parser.error("positive fit/holdout samples and seqlen >= 2 required")
    if args.group_size < 1 or not 0 < args.target_rate < 1:
        parser.error("positive group size and target rate in (0,1) required")
    return args


def train_samples(tokenizer, args):
    cache = Path(os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache/huggingface/datasets"))
    arrow = sorted(cache.glob("wikitext/wikitext-2-raw-v1/*/*/wikitext-train.arrow"))
    dataset = Dataset.from_file(str(arrow[-1])) if arrow else load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids
    # Sample distinct, nonoverlapping blocks; reserve the last few for selection.
    rng = random.Random(args.seed)
    blocks = rng.sample(range(ids.shape[1] // args.seqlen), args.samples + args.holdout_samples)
    log(f"[calibration] WikiText-2 TRAIN: {args.samples} fit + {args.holdout_samples} holdout x {args.seqlen} tokens; blocks={blocks}")
    return [ids[:, i * args.seqlen:(i + 1) * args.seqlen].to(args.device) for i in blocks], blocks


class CaptureComplete(Exception):
    pass


@torch.no_grad()
def capture_first_inputs(model, batches):
    hidden = []
    kwargs = None

    def capture(module, inputs, named):
        nonlocal kwargs
        hidden.append(inputs[0].detach())
        kwargs = named
        raise CaptureComplete

    handle = model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for batch in batches:
            try:
                model(batch, use_cache=False)
            except CaptureComplete:
                pass
    finally:
        handle.remove()
    return hidden, kwargs


def candidate_path(args, name, bits):
    return args.work_dir / "candidates" / f"{name}.b{bits}.pt"


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: device_tree(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(device_tree(v, device) for v in value)
    return value


@torch.no_grad()
def block_error(layer, inputs, references, kwargs):
    numerator = 0.0
    denominator = 0.0
    for x, reference in zip(inputs, references):
        output = layer(x, **kwargs).float()
        numerator += (output-reference.float()).square().sum().item()
        denominator += reference.float().square().sum().item()
    return numerator/max(denominator, 1e-12)


@torch.no_grad()
def scan(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=MODEL_REVISION)
    batches, blocks = train_samples(tokenizer, args)
    settings = dict(model=args.model, revision=AutoConfig.from_pretrained(args.model, revision=MODEL_REVISION)._commit_hash,
                    samples=args.samples, holdout_samples=args.holdout_samples,
                    seqlen=args.seqlen, group_size=args.group_size, damp=args.damp,
                    act_order=args.act_order, seed=args.seed, blocks=blocks,
                    max_layers=args.max_layers, implementation="case2-affine-gptq-v1")
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    scan_dir = args.work_dir / "candidates"
    block_dir = args.work_dir / "blocks"
    scan_dir.mkdir(parents=True, exist_ok=True)
    block_dir.mkdir(parents=True, exist_ok=True)
    path = scan_dir / "settings.json"
    if path.exists() and json.loads(path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Calibration settings changed; use another --work-dir")
    path.write_text(json.dumps(dict(fingerprint=fingerprint, settings=settings), indent=2))
    registry_path = args.work_dir / "scan.json"
    if registry_path.exists() and not args.rebuild:
        registry = json.loads(registry_path.read_text())
        if registry["fingerprint"] != fingerprint:
            raise ValueError("Scan fingerprint mismatch")
        log("[scan] reusing completed candidate scan")
        return registry
    log("[scan] original FP16 model, one GPU; no inference parameter offloading")
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=settings["revision"],
                                                dtype=torch.float16, device_map={"": args.device})
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    specs = target_linear_specs(model)
    hidden, kwargs = capture_first_inputs(model, batches)
    del batches
    rows = []
    for index, layer in enumerate(model.model.layers):
        if args.max_layers is not None and index >= args.max_layers:
            break
        started = time.monotonic()
        modules = {n: m for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)}
        names = {n: f"model.layers.{index}.{n}" for n in modules}
        finished = block_dir / f"layer{index}.json"
        cached = finished.exists() and not args.rebuild
        missing = [n for n in modules if any(args.rebuild or not candidate_path(args, names[n], b).exists() for b in (2, 3))]
        cached = cached and not missing
        grams, heldout, handles = {}, {}, []

        def family(name):
            suffix = name.rsplit(".", 1)[-1]
            if suffix in ("gate_proj", "up_proj"):
                return "mlp_input"
            if suffix in ("q_proj", "k_proj", "v_proj"):
                return "attn_input"
            return name

        for name in missing:
            key = family(name)
            if key in grams:
                continue
            width = modules[name].in_features
            grams[key] = torch.zeros((width, width), device=args.device, dtype=torch.float32)
            heldout[key] = []

            def collect(module, inputs, key=key):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                if sample_index < args.samples:
                    xf = x.float()
                    grams[key].addmm_(xf.T, xf, beta=1, alpha=1/(args.samples*args.seqlen))
                else:
                    heldout[key].append(x[::max(1, len(x)//128)].clone())

            handles.append(modules[name].register_forward_pre_hook(collect))
        next_hidden = []
        try:
            for sample_index, value in enumerate(hidden):
                next_hidden.append(layer(value, **kwargs).detach())
        finally:
            for handle in handles:
                handle.remove()
        if cached:
            layer_rows = json.loads(finished.read_text())["rows"]
        else:
            layer_rows = []
            for name, module in modules.items():
                for bits in (3, 2):
                    path = candidate_path(args, names[name], bits)
                    if path.exists() and not args.rebuild:
                        payload = torch.load(path, map_location="cpu", weights_only=True)
                        metrics = payload["metrics"]
                        del payload
                    else:
                        tick = time.monotonic()
                        key = family(name)
                        candidate = quantize_gptq(module.weight, grams[key], bits,
                                                  args.group_size, args.damp, args.act_order)
                        error = output_error(module.weight, candidate, torch.cat(heldout[key]))
                        if module.bias is not None:
                            candidate.bias = module.bias.detach().clone()
                        metrics = dict(name=names[name], bits=bits, output_error=error,
                                       packed_bytes=module_payload(candidate))
                        torch.save(dict(spec=candidate.spec(), state=cpu_tree(candidate.state_dict()),
                                        metrics=metrics), path)
                        log(f"[candidate] {names[name]} b{bits}: E={error:.6g}, {time.monotonic()-tick:.1f}s")
                        del candidate
                    layer_rows.append(dict(metrics))
            # Compare downgrades in an otherwise all-3-bit decoder block.
            quantized3 = {}
            for name in modules:
                payload = torch.load(candidate_path(args, names[name], 3), map_location="cpu", weights_only=True)
                quantized3[name] = candidate_module(payload, args.device)
                set_module(layer, name, quantized3[name])
            hold_inputs = hidden[-args.holdout_samples:]
            hold_references = next_hidden[-args.holdout_samples:]
            base_error = block_error(layer, hold_inputs, hold_references, kwargs)
            for name in modules:
                payload = torch.load(candidate_path(args, names[name], 2), map_location="cpu", weights_only=True)
                candidate2 = candidate_module(payload, args.device)
                set_module(layer, name, candidate2)
                error = block_error(layer, hold_inputs, hold_references, kwargs)
                set_module(layer, name, quantized3[name])
                for row in layer_rows:
                    if row["name"] == names[name]:
                        row["block_error"] = base_error if row["bits"] == 3 else error
                        row["block_delta"] = 0.0 if row["bits"] == 3 else error-base_error
                log(f"[block] {names[name]} 3->2: delta={error-base_error:.6g}")
                del candidate2
            for name, module in modules.items():
                set_module(layer, name, module)
            torch.save(dict(inputs=cpu_tree(hold_inputs), references=cpu_tree(hold_references),
                            kwargs=cpu_tree(kwargs)), block_dir / f"layer{index}.pt")
            finished.write_text(json.dumps(dict(rows=layer_rows, baseline3_error=base_error), indent=2))
            del quantized3, hold_inputs, hold_references, payload
        rows.extend(layer_rows)
        hidden = next_hidden
        del next_hidden, grams, heldout
        gc.collect()
        torch.cuda.empty_cache()
        log(f"[scan] layer {index+1}/36 done ({time.monotonic()-started:.1f}s)")
    registry = dict(fingerprint=fingerprint, settings=settings, original_specs=specs, rows=rows)
    registry_path.write_text(json.dumps(registry, indent=2))
    del model, hidden, kwargs, modules
    gc.collect()
    torch.cuda.empty_cache()
    return registry


@torch.no_grad()
def scan_frozen(args):
    """Build only the submitted bit-width for each Linear; no candidate ranking."""
    frozen = load_frozen_manifest(args)
    int2 = {row['name'] for row in frozen['selected']}
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=MODEL_REVISION)
    batches, blocks = train_samples(tokenizer, args)
    settings = dict(model=args.model,
                    revision=AutoConfig.from_pretrained(args.model, revision=MODEL_REVISION)._commit_hash,
                    samples=args.samples, holdout_samples=args.holdout_samples,
                    seqlen=args.seqlen, group_size=args.group_size, damp=args.damp,
                    act_order=args.act_order, seed=args.seed, blocks=blocks,
                    max_layers=args.max_layers, implementation='case2-affine-gptq-v1')
    allocation_hash = hashlib.sha256(json.dumps(sorted(int2)).encode()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(dict(settings=settings, allocation=allocation_hash),
                                               sort_keys=True).encode()).hexdigest()
    scan_dir = args.work_dir/'candidates'
    scan_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.work_dir/'frozen_scan.json'
    if registry_path.exists() and not args.rebuild:
        registry = json.loads(registry_path.read_text())
        paths_ok = all(candidate_path(args, row['name'], row['bits']).exists()
                       for row in registry.get('rows', []))
        if registry.get('fingerprint') == fingerprint and paths_ok:
            log('[frozen] reusing completed fixed-bit GPTQ build scan')
            return registry
    log('[frozen] computing one predetermined GPTQ candidate per decoder Linear')
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=settings['revision'],
                                                dtype=torch.float16, device_map={'': args.device})
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    specs = target_linear_specs(model)
    hidden, kwargs = capture_first_inputs(model, batches[:args.samples])
    del batches
    rows = []
    for index, layer in enumerate(model.model.layers):
        started = time.monotonic()
        modules = {name: module for name, module in layer.named_modules()
                   if isinstance(module, torch.nn.Linear)}
        names = {name:f'model.layers.{index}.{name}' for name in modules}
        missing = [name for name in modules
                   if args.rebuild or not candidate_path(args, names[name], 2 if names[name] in int2 else 3).exists()]
        grams, handles = {}, []

        def family(name):
            suffix = name.rsplit('.', 1)[-1]
            if suffix in ('gate_proj', 'up_proj'):
                return 'mlp_input'
            if suffix in ('q_proj', 'k_proj', 'v_proj'):
                return 'attn_input'
            return name

        for name in missing:
            key = family(name)
            if key in grams:
                continue
            grams[key] = torch.zeros((modules[name].in_features, modules[name].in_features),
                                     device=args.device, dtype=torch.float32)

            def collect(module, inputs, key=key):
                value = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
                grams[key].addmm_(value.T, value, beta=1,
                                  alpha=1/(args.samples*args.seqlen))

            handles.append(modules[name].register_forward_pre_hook(collect))
        next_hidden = []
        try:
            for value in hidden:
                next_hidden.append(layer(value, **kwargs).detach())
        finally:
            for handle in handles:
                handle.remove()
        for name, module in modules.items():
            full_name = names[name]
            bits = 2 if full_name in int2 else 3
            path = candidate_path(args, full_name, bits)
            if path.exists() and not args.rebuild:
                payload = torch.load(path, map_location='cpu', weights_only=True)
                metrics = dict(payload['metrics'])
                del payload
            else:
                candidate = quantize_gptq(module.weight, grams[family(name)], bits,
                                          args.group_size, args.damp, args.act_order)
                if module.bias is not None:
                    candidate.bias = module.bias.detach().clone()
                metrics = dict(name=full_name, bits=bits,
                               packed_bytes=module_payload(candidate))
                torch.save(dict(spec=candidate.spec(), state=cpu_tree(candidate.state_dict()),
                                metrics=metrics), path)
                del candidate
            metrics['name'] = full_name
            metrics['bits'] = bits
            rows.append(metrics)
        hidden = next_hidden
        del next_hidden, grams
        gc.collect()
        torch.cuda.empty_cache()
        log(f'[frozen] layer {index+1}/36 built ({time.monotonic()-started:.1f}s)')
    registry = dict(fingerprint=fingerprint, settings=settings, original_specs=specs,
                    rows=rows, allocation_mode='frozen', int2_layers=sorted(int2))
    registry_path.write_text(json.dumps(registry, indent=2))
    del model, hidden, kwargs, modules
    gc.collect()
    torch.cuda.empty_cache()
    return registry


@torch.no_grad()
def build(args, registry):
    # Some names are assigned only while searching.  Initialise them so the
    # common checkpoint-reload cleanup also works with a frozen allocation.
    module = payload = layer = None
    frozen = None if args.search else load_frozen_manifest(args)
    output_dir = checkpoint_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=registry["settings"]["revision"],
                                                dtype=torch.float16, device_map={"": args.device}, quantization_config=cfg)
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    install_shared_nf4(model)
    specs = registry["original_specs"]
    before = original_bytes(specs)
    by_name = {}
    for row in registry["rows"]:
        by_name.setdefault(row["name"], {})[row["bits"]] = row
    if frozen is not None:
        for name, values in by_name.items():
            bits = next(iter(values))
            payload = torch.load(candidate_path(args, name, bits), map_location='cpu', weights_only=True)
            set_module(model, name, candidate_module(payload, args.device))
        after = target_payload(model, specs)
        ranking = frozen['ranking']
        selected = frozen['selected']
        block_errors = frozen['joint_block_errors']
        if after != frozen['compressed_linear_bytes']:
            raise AssertionError(f"Frozen Case 2 payload changed: {after} != {frozen['compressed_linear_bytes']}")
        log(f'[storage] frozen INT2/INT3 allocation: {before:,} -> {after:,} bytes ({1-after/before:.6%})')
    else:
        for name in by_name:
            if name in args.protect:
                continue
            payload = torch.load(candidate_path(args, name, 3), map_location="cpu", weights_only=True)
            set_module(model, name, candidate_module(payload, args.device))
        after = target_payload(model, specs)
        log(f"[storage] decoder-3bit baseline: {before:,} -> {after:,} bytes ({1-after/before:.6%})")
        ranking = []
        for name, values in by_name.items():
            if name in args.protect:
                continue
            saving = values[3]["packed_bytes"]-values[2]["packed_bytes"]
            delta = values[2]["block_delta"] if args.score == "block" else values[2]["output_error"]-values[3]["output_error"]
            ranking.append(dict(name=name, saved_bytes=saving, delta_error=delta,
                                score=delta/saving, error2=values[2]["output_error"], error3=values[3]["output_error"]))
        ranking.sort(key=lambda row: row["score"])
        selected = []
        if args.mode == "mixed":
            for row in ranking:
                if 1-after/before >= args.target_rate:
                    break
                payload = torch.load(candidate_path(args, row["name"], 2), map_location="cpu", weights_only=True)
                set_module(model, row["name"], candidate_module(payload, args.device))
                after -= row["saved_bytes"]
                selected.append(row)
                log(f"[select] {row['name']} b2: delta={row['delta_error']:.6g}, rate={1-after/before:.6%}")
    measured = target_payload(model, specs)
    if measured != after:
        raise AssertionError("Storage bookkeeping mismatch")
    # Validate combined choices on complete blocks, using TRAIN holdout only.
    if frozen is None:
        block_errors = {}
        for index, layer in enumerate(model.model.layers):
            path = args.work_dir / "blocks" / f"layer{index}.pt"
            if not path.exists():
                continue
            holdout = device_tree(torch.load(path, map_location="cpu", weights_only=True), args.device)
            error = block_error(layer, holdout["inputs"], holdout["references"], holdout["kwargs"])
            baseline = json.loads(path.with_suffix(".json").read_text())["baseline3_error"]
            block_errors[index] = dict(baseline3=baseline, selected=error, increase=error-baseline)
            log(f"[joint-block] {index}: E={error:.6g}, baseline3={baseline:.6g}")
            del holdout
    formats = {}
    for name in specs:
        module = model.get_submodule(name)
        formats[name] = module.spec() if isinstance(module, PackedLowBitLinear) else dict(kind="nf4", **specs[name])
    manifest = dict(model=args.model, mode=args.mode, original_specs=specs, formats=formats,
                    original_linear_bytes=before, compressed_linear_bytes=after,
                    compression_rate=1-after/before, target_rate=args.target_rate,
                    target_met=1-after/before >= args.target_rate,
                    **accounting_metadata(specs),
                    calibration=registry["settings"], score=args.score, protect=args.protect,
                    selected=selected, ranking=ranking, joint_block_errors=block_errors,
                    allocation_mode='search' if args.search else 'frozen',
                    frozen_source=FROZEN_MANIFEST.name if frozen is not None else None)
    checkpoint = output_dir / "model.safetensors"
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(checkpoint))
    manifest["checkpoint_file_bytes"] = checkpoint.stat().st_size
    manifest['storage_audit'] = checkpoint_audit(checkpoint, specs)
    if manifest['storage_audit']['compressed_linear_bytes'] != after:
        raise AssertionError('Serialized target payload differs from measured module state')
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"[checkpoint] {checkpoint}: {checkpoint.stat().st_size:,} bytes; {len(selected)} 2-bit layers")
    ids = torch.tensor([[1, 42, 314, 159, 265, 35, 897, 93]], device=args.device)
    reference = model(ids).logits.detach().cpu()
    del model, module, payload, layer
    gc.collect()
    torch.cuda.empty_cache()
    restored = load_checkpoint(args)
    actual = restored(ids).logits.detach().cpu()
    difference = (reference.float()-actual.float()).abs().max().item()
    if not torch.equal(reference, actual):
        raise AssertionError(f"Reload changed logits: max difference {difference}")
    manifest["reload_check"] = dict(bitwise_equal_logits=True, max_abs_difference=difference)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log("[checkpoint] reload passed: bitwise identical logits")
    return restored


@torch.no_grad()
def load_checkpoint(args):
    manifest = json.loads((checkpoint_dir(args) / "manifest.json").read_text())
    require_current_scope(manifest)
    state = load_file(str(checkpoint_dir(args) / "model.safetensors"))
    config = AutoConfig.from_pretrained(manifest["model"], revision=manifest["calibration"]["revision"])
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
    for name, spec in manifest["formats"].items():
        prefix = name + "."
        if spec["kind"] == "gptq":
            payload = dict(spec=spec, state={key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)})
            module = candidate_module(payload, args.device)
        else:
            module = bnb.nn.Linear4bit(spec["in_features"], spec["out_features"], bias=spec["bias"],
                                       compute_dtype=torch.float16, compress_statistics=True, quant_type="nf4")
            qprefix = prefix + "weight."
            stats = {key[len(qprefix):]: value for key, value in state.items() if key.startswith(qprefix)}
            module.weight = bnb.nn.Params4bit.from_prequantized(state[prefix+"weight"], stats,
                                                               requires_grad=False, device=args.device, module=module)
            if spec["bias"]:
                module.bias = torch.nn.Parameter(state[prefix+"bias"].to(args.device), requires_grad=False)
        set_module(model, name, module)
    model.set_input_embeddings(SharedNF4Embedding(model.get_output_embeddings()))
    # Only norms remain as ordinary parameters; all Linear weights were restored above.
    for name, parameter in list(model.named_parameters()):
        if parameter.is_meta:
            parent_name, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent_name), attr,
                    torch.nn.Parameter(state[name].to(args.device), requires_grad=False))
    for name, buffer in list(model.named_buffers()):
        if buffer.device != torch.device(args.device):
            parent_name, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent_name), attr, buffer.to(args.device))
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    measured = target_payload(model, manifest["original_specs"])
    if measured != manifest["compressed_linear_bytes"]:
        raise AssertionError("Restored model size does not match manifest")
    return model


def run_official_evaluation(model, args):
    manifest = json.loads((checkpoint_dir(args) / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"], revision=manifest["calibration"]["revision"])
    evaluate.load_model = lambda *unused, **kwargs: (model, tokenizer)
    evaluate.compression_rate = lambda model: manifest["compression_rate"]
    runner = Path(__file__).with_name("run_eval.py")
    sys.argv = [str(runner), "--case", "2", "--rate", str(args.target_rate),
                "--model", args.model, "--dtype", "float16", "--seed", str(args.seed)]
    log("[eval] unmodified instructor run_eval.py: WikiText-2 TEST, seqlen=2048")
    output = io.StringIO()
    started = time.monotonic()
    segments = 0

    def progress(module, inputs, result):
        nonlocal segments
        segments += 1
        if segments % 16 == 0:
            log(f"[eval] processed {segments} segments ({time.monotonic()-started:.1f}s)")

    handle = model.register_forward_hook(progress)
    try:
        with contextlib.redirect_stdout(output):
            runpy.run_path(str(runner), run_name="__main__")
    finally:
        handle.remove()
    print(output.getvalue(), end="", flush=True)
    ppl = float(re.search(r"^ppl: ([0-9.]+)$", output.getvalue(), re.MULTILINE).group(1))
    result = dict(case=2, ppl=ppl, compression_rate=manifest["compression_rate"],
                  original_linear_bytes=manifest["original_linear_bytes"],
                  compressed_linear_bytes=manifest["compressed_linear_bytes"],
                  case2_pass=manifest["compression_rate"] >= 0.81 and ppl < 20,
                  **accounting_metadata(manifest['original_specs']),
                  storage_audit=manifest['storage_audit'],
                  ppl_provenance='fresh official run_eval.py evaluation',
                  evaluation_seconds=time.monotonic()-started,
                  evaluator_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (runner, runner.with_name("evaluate.py"))})
    (checkpoint_dir(args) / "result.json").write_text(json.dumps(result, indent=2))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if not args.search and not args.eval_only and (args.mode != 'mixed' or args.protect or args.max_layers is not None):
        raise ValueError('Frozen Case 2 supports the submitted mixed mode only; use --search for custom experiments')
    if args.eval_only:
        model = load_checkpoint(args)
    else:
        registry = scan(args) if args.search else scan_frozen(args)
        model = build(args, registry)
    if not args.skip_eval:
        run_official_evaluation(model, args)


if __name__ == "__main__":
    main()

