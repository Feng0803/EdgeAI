"""Case 3: activation-weighted SVD >=20%, exact 50% overall payload, official PPL."""

import argparse
import contextlib
from fractions import Fraction
import gc
import hashlib
import heapq
import io
import json
import os
from pathlib import Path
import random
import re
import runpy
import sys
import time

from datasets import Dataset, load_dataset
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights

import evaluate
from linear_accounting import (accounting_metadata, checkpoint_audit,
                               require_current_scope, target_payload)
from svd_linear import (LowRankLinear, RowMixedLinear, allocate_int8_rows, apply_int8_rows,
                        int8_candidate, lowrank_from_basis, module_payload, original_bytes,
                        set_module, target_linear_specs, weighted_svd_basis)


MODEL_REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
FROZEN_MANIFEST = Path(__file__).resolve().parents[1] / 'frozen' / 'case3_manifest.json'

def log(message):
    print(message, file=sys.stderr, flush=True)


def load_frozen_manifest(args):
    manifest = json.loads(FROZEN_MANIFEST.read_text())
    if manifest['model'] != args.model or manifest['calibration']['revision'] != MODEL_REVISION:
        raise ValueError('Frozen Case 3 allocation does not match the requested model revision')
    return manifest


def checkpoint_dir(args):
    return args.work_dir / args.mode


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen3-4B')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mode', choices=['svd', 'mixed'], default='mixed')
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--holdout-samples', type=int, default=8)
    parser.add_argument('--seqlen', type=int, default=1024, help='calibration only; official PPL stays 2048')
    parser.add_argument('--svd-rate', type=float, default=.2227)
    parser.add_argument('--overall-rate', default='0.5', help='exact rational byte target; change only for smoke tests')
    parser.add_argument('--damp', type=float, default=1e-4)
    parser.add_argument('--min-rank', type=int, default=1024)
    parser.add_argument('--rank-step', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-layers', type=int, default=None, help='smoke test only')
    parser.add_argument('--work-dir', type=Path, default=Path(__file__).parent/'artifacts/weighted')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--search', action='store_true',
                        help='rerun SVD rank and INT8-row search instead of the submitted allocation')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--skip-eval', action='store_true')
    args = parser.parse_args()
    if args.samples < 1 or args.holdout_samples < 1 or args.seqlen < 2:
        parser.error('positive calibration counts and seqlen >=2 required')
    if not 0 < args.svd_rate < 1 or not 0 < Fraction(args.overall_rate) < 1:
        parser.error('compression rates must be in (0,1)')
    if args.min_rank < 1 or args.rank_step < 1:
        parser.error('positive ranks and steps required')
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
                    samples=args.samples, holdout_samples=args.holdout_samples, seqlen=args.seqlen,
                    seed=args.seed, blocks=blocks, damp=args.damp, min_rank=args.min_rank,
                    rank_step=args.rank_step, max_layers=args.max_layers, implementation='weighted-svd-v1')
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    basis_dir = args.work_dir/'basis'
    block_dir = args.work_dir/'blocks'
    basis_dir.mkdir(parents=True, exist_ok=True)
    block_dir.mkdir(parents=True, exist_ok=True)
    path = args.work_dir/'settings.json'
    if path.exists() and json.loads(path.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Calibration settings changed; choose another --work-dir')
    path.write_text(json.dumps(dict(fingerprint=fingerprint, settings=settings), indent=2))
    registry_path = args.work_dir/'scan.json'
    if registry_path.exists() and not args.rebuild:
        registry = json.loads(registry_path.read_text())
        if registry['fingerprint'] != fingerprint:
            raise ValueError('Scan fingerprint mismatch')
        log('[scan] reusing completed weighted SVD scan')
        return registry
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=settings['revision'],
                                                dtype=torch.float16, device_map={'': args.device})
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
        modules = {name: module for name, module in layer.named_modules() if isinstance(module, torch.nn.Linear)}
        targets = {name: module for name, module in modules.items() if name.startswith('mlp.')}
        paths = {name: basis_dir/f'model.layers.{index}.{name}.pt' for name in targets}
        missing = [name for name in targets if args.rebuild or not paths[name].exists()]
        block_path = block_dir/f'layer{index}.pt'
        collect_holdout = args.rebuild or not block_path.exists() or bool(missing)
        grams, heldout, handles = {}, {}, []

        def family(name):
            suffix = name.rsplit('.', 1)[-1]
            if suffix in ('q_proj', 'k_proj', 'v_proj'):
                return 'attention_input'
            if suffix in ('gate_proj', 'up_proj'):
                return 'mlp_input'
            return name

        required_grams = {family(name) for name in missing}
        for name, module in modules.items():
            key = family(name)
            if key in heldout:
                continue
            if collect_holdout:
                heldout[key] = []
            if key in required_grams:
                width = module.in_features
                grams[key] = torch.zeros((width, width), device=args.device, dtype=torch.float32)
            if not collect_holdout and key not in required_grams:
                continue

            def collect(module, inputs, key=key):
                value = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                if sample_index < args.samples and key in grams:
                    xf = value.float()
                    grams[key].addmm_(xf.T, xf, beta=1, alpha=1/(args.samples*args.seqlen))
                elif sample_index >= args.samples and collect_holdout:
                    heldout[key].append(value[::max(1, len(value)//128)].clone())

            handles.append(module.register_forward_pre_hook(collect))
        next_hidden = []
        try:
            for sample_index, value in enumerate(hidden):
                next_hidden.append(layer(value, **kwargs).detach())
        finally:
            for handle in handles:
                handle.remove()
        family_inputs = {key: torch.cat(values) for key, values in heldout.items() if values}
        for name, module in targets.items():
            full_name = f'model.layers.{index}.{name}'
            if name not in missing:
                payload = torch.load(paths[name], map_location='cpu', weights_only=True)
                rows.extend(payload['rows'])
                del payload
                continue
            tick = time.monotonic()
            m, n = module.weight.shape
            maximum_rank = (m*n-1)//(m+n)
            minimum_rank = min(args.min_rank, maximum_rank)
            ranks = sorted(set(range(minimum_rank, maximum_rank+1, args.rank_step)) | {maximum_rank}, reverse=True)
            left, right, spectrum = weighted_svd_basis(module.weight, grams[family(name)], args.damp, maximum_rank)
            inputs = family_inputs[family(name)]
            reference = torch.nn.functional.linear(inputs, module.weight).float()
            denominator = reference.square().sum().clamp_min(1e-12)
            candidate_rows = []
            for rank in ranks:
                output = torch.nn.functional.linear(torch.nn.functional.linear(inputs, right[:rank]), left[:, :rank]).float()
                error = ((output-reference).square().sum()/denominator).item()
                candidate_rows.append(dict(name=full_name, rank=rank, error=error,
                                           saved_bytes=2*(m*n-rank*(m+n))))
            torch.save(dict(left=left.cpu(), right=right.cpu(), spectrum=spectrum.cpu(), rows=candidate_rows), paths[name])
            rows.extend(candidate_rows)
            log(f'[svd] {full_name}: {len(ranks)} ranks, r={minimum_rank}..{maximum_rank}, {time.monotonic()-tick:.1f}s')
            del left, right, spectrum, reference, output, inputs
        if collect_holdout:
            torch.save(dict(inputs=cpu_tree(hidden[-args.holdout_samples:]),
                            references=cpu_tree(next_hidden[-args.holdout_samples:]),
                            kwargs=cpu_tree(kwargs), family_inputs=cpu_tree(family_inputs)), block_path)
        hidden = next_hidden
        del next_hidden, grams, heldout, family_inputs
        gc.collect()
        torch.cuda.empty_cache()
        log(f'[scan] layer {index+1}/36 finished ({time.monotonic()-started:.1f}s)')
    registry = dict(settings=settings, fingerprint=fingerprint, original_specs=specs, rows=rows)
    registry_path.write_text(json.dumps(registry, indent=2))
    del model, hidden, kwargs, modules, targets
    gc.collect()
    torch.cuda.empty_cache()
    return registry


@torch.no_grad()
def scan_frozen(args):
    """Compute only the 69 submitted weighted-SVD factors at their final ranks."""
    frozen = load_frozen_manifest(args)
    ranks = {name:row['rank'] for name,row in frozen['ranks'].items()}
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=MODEL_REVISION)
    batches, blocks = train_samples(tokenizer, args)
    settings = dict(model=args.model,
                    revision=AutoConfig.from_pretrained(args.model, revision=MODEL_REVISION)._commit_hash,
                    samples=args.samples, holdout_samples=args.holdout_samples, seqlen=args.seqlen,
                    seed=args.seed, blocks=blocks, damp=args.damp, min_rank=args.min_rank,
                    rank_step=args.rank_step, max_layers=args.max_layers,
                    implementation='weighted-svd-v1')
    allocation_hash = hashlib.sha256(json.dumps(ranks, sort_keys=True).encode()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(dict(settings=settings, allocation=allocation_hash),
                                               sort_keys=True).encode()).hexdigest()
    basis_dir = args.work_dir/'basis'
    basis_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.work_dir/'frozen_scan.json'
    expected_paths = [basis_dir/f'{name}.pt' for name in ranks]
    if registry_path.exists() and not args.rebuild and all(path.exists() for path in expected_paths):
        registry = json.loads(registry_path.read_text())
        if registry.get('fingerprint') == fingerprint:
            log('[frozen] reusing completed fixed-rank SVD scan')
            return registry
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=settings['revision'],
                                                dtype=torch.float16, device_map={'': args.device})
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    specs = target_linear_specs(model)
    hidden, kwargs = capture_first_inputs(model, batches[:args.samples])
    del batches
    for index, layer in enumerate(model.model.layers):
        started = time.monotonic()
        modules = {name:module for name,module in layer.named_modules()
                   if isinstance(module, torch.nn.Linear)}
        targets = {name:module for name,module in modules.items()
                   if f'model.layers.{index}.{name}' in ranks}
        paths = {name:basis_dir/f'model.layers.{index}.{name}.pt' for name in targets}
        missing = [name for name in targets if args.rebuild or not paths[name].exists()]
        grams, handles = {}, []

        def family(name):
            suffix = name.rsplit('.', 1)[-1]
            return 'mlp_input' if suffix in ('gate_proj', 'up_proj') else name

        for name in missing:
            key = family(name)
            if key in grams:
                continue
            grams[key] = torch.zeros((targets[name].in_features, targets[name].in_features),
                                     device=args.device, dtype=torch.float32)

            def collect(module, inputs, key=key):
                value = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
                grams[key].addmm_(value.T, value, beta=1,
                                  alpha=1/(args.samples*args.seqlen))

            handles.append(targets[name].register_forward_pre_hook(collect))
        next_hidden = []
        try:
            for value in hidden:
                next_hidden.append(layer(value, **kwargs).detach())
        finally:
            for handle in handles:
                handle.remove()
        for name in missing:
            full_name = f'model.layers.{index}.{name}'
            rank = ranks[full_name]
            left, right, spectrum = weighted_svd_basis(targets[name].weight, grams[family(name)],
                                                       args.damp, rank)
            row = dict(frozen['ranks'][full_name])
            torch.save(dict(left=left.cpu(), right=right.cpu(), spectrum=spectrum.cpu(), rows=[row]),
                       paths[name])
            del left, right, spectrum
        hidden = next_hidden
        del next_hidden, grams
        gc.collect()
        torch.cuda.empty_cache()
        log(f'[frozen] SVD layer {index+1}/36 built ({time.monotonic()-started:.1f}s)')
    registry = dict(settings=settings, fingerprint=fingerprint, original_specs=specs,
                    rows=list(frozen['ranks'].values()), allocation_mode='frozen')
    registry_path.write_text(json.dumps(registry, indent=2))
    del model, hidden, kwargs, modules, targets
    gc.collect()
    torch.cuda.empty_cache()
    return registry


def convex_options(rows):
    """Lower convex envelope of loss versus bytes saved, including the dense option."""
    points = [dict(rank=0, error=0.0, saved_bytes=0)]+sorted(rows, key=lambda r: r['saved_bytes'])
    hull = []
    for point in points:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            old = (b['error']-a['error'])/(b['saved_bytes']-a['saved_bytes'])
            new = (point['error']-b['error'])/(point['saved_bytes']-b['saved_bytes'])
            if old < new:
                break
            hull.pop()
        hull.append(point)
    return hull


def select_ranks(registry, args):
    grouped = {}
    for row in registry['rows']:
        grouped.setdefault(row['name'], []).append(row)
    hulls = {name: convex_options(rows) for name, rows in grouped.items()}
    current = {name: 0 for name in hulls}
    heap = []

    def push(name):
        index = current[name]
        if index+1 < len(hulls[name]):
            before, after = hulls[name][index:index+2]
            slope = (after['error']-before['error'])/(after['saved_bytes']-before['saved_bytes'])
            heapq.heappush(heap, (slope, name, index+1))

    for name in hulls:
        push(name)
    before = original_bytes(registry['original_specs'])
    saving = 0
    while saving/before < args.svd_rate and heap:
        _, name, index = heapq.heappop(heap)
        saving += hulls[name][index]['saved_bytes']-hulls[name][index-1]['saved_bytes']
        current[name] = index
        push(name)
    chosen = {name: hulls[name][index] for name, index in current.items() if index}
    log(f'[ranks] SVD saving={saving:,}/{before:,} bytes ({saving/before:.6%}), {len(chosen)} factored layers')
    if saving/before < args.svd_rate:
        log('[ranks] requested SVD rate unavailable with the scanned smoke-test candidates')
    return chosen, saving


def family_for_name(name):
    suffix = name.rsplit('.', 1)[-1]
    if suffix in ('q_proj', 'k_proj', 'v_proj'):
        return 'attention_input'
    if suffix in ('gate_proj', 'up_proj'):
        return 'mlp_input'
    return '.'.join(name.split('.')[3:])


@torch.no_grad()
def quantize_to_budget(model, specs, args, target_bytes):
    components = []
    cached = {}
    by_block = {}
    for name in specs:
        if name == 'lm_head':
            continue
        by_block.setdefault(int(name.split('.')[2]), []).append(name)
    for index, names in by_block.items():
        block_path = args.work_dir/'blocks'/f'layer{index}.pt'
        family_inputs = None
        if block_path.exists():
            data = torch.load(block_path, map_location='cpu', weights_only=True)
            family_inputs = device_tree(data['family_inputs'], args.device)
            del data
        for name in names:
            module = model.get_submodule(name)
            inputs = None if family_inputs is None else family_inputs[family_for_name(name)]
            pieces = [(name, module, inputs)]
            if isinstance(module, LowRankLinear):
                right_output = None if inputs is None else module.right(inputs)
                pieces = [(name+'.right', module.right, inputs), (name+'.left', module.left, right_output)]
            for component_name, piece, value in pieces:
                qweight, scales = int8_candidate(piece.weight16)
                reconstructed = qweight.half()*scales
                if value is None:
                    error = ((reconstructed.float()-piece.weight16.float()).square().sum()/piece.weight16.float().square().sum().clamp_min(1e-12)).item()
                else:
                    reference = torch.nn.functional.linear(value, piece.weight16).float()
                    output = torch.nn.functional.linear(value, reconstructed).float()
                    error = ((output-reference).square().sum()/reference.square().sum().clamp_min(1e-12)).item()
                    del reference, output
                capacity = piece.out_features*(piece.in_features-2)
                components.append(dict(name=component_name, in_features=piece.in_features,
                                       out_features=piece.out_features, error=error, score=error/capacity))
                cached[component_name] = (qweight.cpu(), scales.cpu())
                del qweight, scales, reconstructed
            del pieces
        del family_inputs
        log(f'[int8] block {index} factor/Linear output errors measured')
    current_bytes = target_payload(model, specs)
    saving_target = current_bytes-target_bytes
    reserves = ['model.layers.0.self_attn.q_proj', 'model.layers.0.self_attn.o_proj']
    assigned = allocate_int8_rows(components, saving_target, reserves)
    for component in components:
        name = component['name']
        module = model.get_submodule(name)
        rows = assigned[name]
        qweight, scales = cached.pop(name)
        set_module(model, name, apply_int8_rows(module, qweight, scales, rows))
    measured = target_payload(model, specs)
    if measured != target_bytes:
        raise AssertionError(f'Overall payload differs from exact target: {measured} != {target_bytes}')
    return dict(rows=assigned, components=components, reserve_components=reserves,
                saved_bytes=saving_target, target_bytes=target_bytes)


@torch.no_grad()
def apply_frozen_int8_rows(model, specs, frozen):
    """Apply the submitted INT8/FP16 row allocation without rescoring components."""
    assigned = frozen['quantization']['rows']
    for name in specs:
        if name == 'lm_head':
            continue
        module = model.get_submodule(name)
        pieces = [(name, module)]
        if isinstance(module, LowRankLinear):
            pieces = [(name+'.right', module.right), (name+'.left', module.left)]
        for component_name, piece in pieces:
            rows = assigned.get(component_name, 0)
            if not rows:
                continue
            if rows > piece.out_features:
                raise AssertionError(f'Frozen INT8 row count exceeds {component_name}')
            qweight, scales = int8_candidate(piece.weight16)
            set_module(model, component_name, apply_int8_rows(piece, qweight, scales, rows))
    after = target_payload(model, specs)
    if after != frozen['compressed_linear_bytes']:
        raise AssertionError(f"Frozen Case 3 payload changed: {after} != {frozen['compressed_linear_bytes']}")
    quantization = dict(frozen['quantization'])
    quantization['allocation_mode'] = 'frozen'
    return quantization


@torch.no_grad()
def build(args, registry):
    # Search and frozen allocation take different paths before the shared
    # checkpoint-reload cleanup.
    module = original = layer = None
    output_dir = checkpoint_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = None if args.search else load_frozen_manifest(args)
    if frozen is None:
        chosen, saving = select_ranks(registry, args)
    else:
        chosen = frozen['ranks']
        saving = sum(row['saved_bytes'] for row in chosen.values())
    specs = registry['original_specs']
    before = original_bytes(specs)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=registry['settings']['revision'],
                                                dtype=torch.float16, device_map={'': args.device})
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    for name in specs:
        if name == 'lm_head':
            continue
        original = model.get_submodule(name)
        if name in chosen:
            payload = torch.load(args.work_dir/'basis'/f'{name}.pt', map_location='cpu', weights_only=True)
            module = lowrank_from_basis(payload, chosen[name]['rank'], args.device, original.bias)
            del payload
        else:
            module = RowMixedLinear.from_fp16(original.weight, original.bias)
        set_module(model, name, module)
    svd_bytes = target_payload(model, specs)
    if svd_bytes != before-saving:
        raise AssertionError('SVD size does not match factor parameter count')
    if frozen is not None and svd_bytes != frozen['svd_linear_bytes']:
        raise AssertionError(f"Frozen SVD payload changed: {svd_bytes} != {frozen['svd_linear_bytes']}")
    quantization = None
    if args.mode == 'mixed':
        if frozen is not None:
            quantization = apply_frozen_int8_rows(model, specs, frozen)
        else:
            exact_target = Fraction(before)*(1-Fraction(args.overall_rate))
            if exact_target.denominator != 1:
                raise ValueError('Requested exact compression is not representable in whole bytes')
            quantization = quantize_to_budget(model, specs, args, int(exact_target))
    after = target_payload(model, specs)
    log(f'[storage] SVD={1-svd_bytes/before:.6%}; overall={1-after/before:.9%}, {after:,} bytes')
    if frozen is None:
        block_errors = {}
        for index, layer in enumerate(model.model.layers):
            path = args.work_dir/'blocks'/f'layer{index}.pt'
            if not path.exists():
                continue
            holdout = device_tree(torch.load(path, map_location='cpu', weights_only=True), args.device)
            error = block_error(layer, holdout['inputs'], holdout['references'], holdout['kwargs'])
            block_errors[index] = error
            log(f'[joint-block] {index}: normalized error={error:.6g}')
            del holdout
    else:
        block_errors = frozen['joint_block_errors']
    formats = {name: (dict(kind='dense', **specs[name]) if name == 'lm_head'
                      else model.get_submodule(name).spec()) for name in specs}
    manifest = dict(model=args.model, mode=args.mode, calibration=registry['settings'], original_specs=specs,
                    original_linear_bytes=before, svd_linear_bytes=svd_bytes,
                    svd_compression_rate=1-svd_bytes/before, compressed_linear_bytes=after,
                    compression_rate=1-after/before, exact_half=2*after == before,
                    ranks=chosen, formats=formats, quantization=quantization,
                    joint_block_errors=block_errors, **accounting_metadata(specs),
                    allocation_mode='search' if args.search else 'frozen',
                    frozen_source=FROZEN_MANIFEST.name if frozen is not None else None)
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()
             if key != 'model.embed_tokens.weight'}
    checkpoint = output_dir/'model.safetensors'
    save_file(state, str(checkpoint))
    del state
    manifest['checkpoint_file_bytes'] = checkpoint.stat().st_size
    manifest['storage_audit'] = checkpoint_audit(checkpoint, specs)
    if manifest['storage_audit']['compressed_linear_bytes'] != after:
        raise AssertionError('Serialized target payload differs from measured module state')
    manifest_path = output_dir/'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f'[checkpoint] saved {checkpoint.stat().st_size:,} bytes')
    ids = torch.tensor([[1, 42, 314, 159, 265, 35, 897, 93]], device=args.device)
    reference = model(ids).logits.detach().cpu()
    del model, module, original, layer
    gc.collect()
    torch.cuda.empty_cache()
    restored = load_checkpoint(args)
    actual = restored(ids).logits.detach().cpu()
    difference = (reference.float()-actual.float()).abs().max().item()
    if not torch.equal(reference, actual):
        raise AssertionError(f'Checkpoint reload changed logits by {difference}')
    manifest['reload_check'] = dict(bitwise_equal_logits=True, max_abs_difference=difference)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log('[checkpoint] reload passed: bitwise identical logits')
    return restored


def module_from_state(spec, state, prefix, device):
    if spec['kind'] == 'rowmixed':
        return RowMixedLinear(state[prefix+'qweight'].to(device), state[prefix+'scales'].to(device),
                              state[prefix+'weight16'].to(device),
                              state[prefix+'bias'].to(device) if spec['bias'] else None)
    if spec['kind'] == 'lowrank':
        return LowRankLinear(module_from_state(spec['left'], state, prefix+'left.', device),
                              module_from_state(spec['right'], state, prefix+'right.', device),
                              state[prefix+'bias'].to(device) if spec['bias'] else None)
    if spec['kind'] == 'dense':
        module = torch.nn.Linear(spec['in_features'], spec['out_features'], bias=spec['bias'],
                                  dtype=torch.float16, device='meta')
        module.weight = torch.nn.Parameter(state[prefix+'weight'].to(device), requires_grad=False)
        if spec['bias']:
            module.bias = torch.nn.Parameter(state[prefix+'bias'].to(device), requires_grad=False)
        return module
    raise ValueError(f'Unsupported module format: {spec}')


@torch.no_grad()
def load_checkpoint(args):
    output_dir = checkpoint_dir(args)
    manifest = json.loads((output_dir/'manifest.json').read_text())
    require_current_scope(manifest)
    state = load_file(str(output_dir/'model.safetensors'))
    config = AutoConfig.from_pretrained(manifest['model'], revision=manifest['calibration']['revision'])
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
    for name, spec in manifest['formats'].items():
        set_module(model, name, module_from_state(spec, state, name+'.', args.device))
    # The head owns the single saved tied matrix; embedding refers to it.
    model.get_input_embeddings().weight = model.get_output_embeddings().weight
    for name, parameter in list(model.named_parameters()):
        if parameter.is_meta:
            parent, _, attr = name.rpartition('.')
            setattr(model.get_submodule(parent), attr,
                    torch.nn.Parameter(state[name].to(args.device), requires_grad=False))
    for name, buffer in list(model.named_buffers()):
        if buffer.device != torch.device(args.device):
            parent, _, attr = name.rpartition('.')
            setattr(model.get_submodule(parent), attr, buffer.to(args.device))
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    measured = target_payload(model, manifest['original_specs'])
    if measured != manifest['compressed_linear_bytes']:
        raise AssertionError('Restored model payload mismatch')
    if model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise AssertionError('Original embedding/lm_head tying lost')
    return model


def run_official_evaluation(model, args):
    output_dir = checkpoint_dir(args)
    manifest = json.loads((output_dir/'manifest.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(manifest['model'], revision=manifest['calibration']['revision'])
    evaluate.load_model = lambda *unused, **kwargs: (model, tokenizer)
    evaluate.compression_rate = lambda unused: manifest['compression_rate']
    runner = Path(__file__).with_name('run_eval.py')
    sys.argv = [str(runner), '--case', '3', '--rate', args.overall_rate,
                '--model', manifest['model'], '--dtype', 'float16', '--seed', str(args.seed)]
    output = io.StringIO()
    started = time.monotonic()
    segments = 0

    def progress(module, inputs, result):
        nonlocal segments
        segments += 1
        if segments % 16 == 0:
            log(f'[eval] processed {segments} segments ({time.monotonic()-started:.1f}s)')

    log('[eval] unmodified official run_eval.py: WikiText-2 TEST, seqlen=2048')
    handle = model.register_forward_hook(progress)
    try:
        with contextlib.redirect_stdout(output):
            runpy.run_path(str(runner), run_name='__main__')
    finally:
        handle.remove()
    print(output.getvalue(), end='', flush=True)
    ppl = float(re.search(r'^ppl: (\S+)$', output.getvalue(), re.MULTILINE).group(1))
    result = dict(case=3, mode=args.mode, ppl=ppl, compression_rate=manifest['compression_rate'],
                  svd_compression_rate=manifest['svd_compression_rate'], exact_half=manifest['exact_half'],
                  original_linear_bytes=manifest['original_linear_bytes'],
                  svd_linear_bytes=manifest['svd_linear_bytes'],
                  compressed_linear_bytes=manifest['compressed_linear_bytes'],
                  case3_pass=manifest['svd_compression_rate'] >= .20 and .495 <= manifest['compression_rate'] <= .505 and ppl < 50,
                  **accounting_metadata(manifest['original_specs']),
                  storage_audit=manifest['storage_audit'],
                  ppl_provenance='fresh official run_eval.py evaluation',
                  evaluation_seconds=time.monotonic()-started,
                  evaluator_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in [runner, runner.with_name('evaluate.py')]})
    (output_dir/'result.json').write_text(json.dumps(result, indent=2)+'\n')


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if not args.search and not args.eval_only and (args.mode != 'mixed' or args.max_layers is not None or
                                                   args.overall_rate != '0.5'):
        raise ValueError('Frozen Case 3 supports the submitted exact-50% mixed mode only; use --search for custom experiments')
    if args.eval_only:
        model = load_checkpoint(args)
    else:
        registry = scan(args) if args.search else scan_frozen(args)
        model = build(args, registry)
    if not args.skip_eval:
        run_official_evaluation(model, args)


if __name__ == '__main__':
    main()


