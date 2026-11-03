from linear_accounting import SCOPE, accounting_metadata, checkpoint_audit, target_linear_names, target_payload

import gc
import hashlib
import heapq
import json
import time

from safetensors.torch import save_file
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from calibration import train_samples, capture_first_inputs, cpu_tree, block_error, log
from gptq_lowbit import module_payload, original_bytes, set_module, target_linear_specs
from pruning import (channel_importance, compensated_down_weight, cropped_mlp_modules,
                     install_mlp_modules, keep_channels, linear_from_weight, pruning_counts)


MODEL_REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'

def candidate_path(args, index, width):
    return args.work_dir/'prune_candidates'/f'layer{index}.w{width}.pt'


@torch.no_grad()
def scan_pruning(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=MODEL_REVISION)
    batches, blocks = train_samples(tokenizer, args)
    settings = dict(model=args.model, revision=AutoConfig.from_pretrained(args.model, revision=MODEL_REVISION)._commit_hash,
                    samples=args.samples, holdout_samples=args.holdout_samples, seqlen=args.seqlen,
                    widths=sorted(args.prune_widths), pruning_rate=args.pruning_rate,
                    prune_damp=args.prune_damp, seed=args.seed, blocks=blocks,
                    max_layers=args.max_layers, implementation='structured-prune-v1', scope=SCOPE)
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    registry_path = args.work_dir/'pruning_scan.json'
    if registry_path.exists() and not args.rebuild:
        registry = json.loads(registry_path.read_text())
        if registry['fingerprint'] != fingerprint:
            raise ValueError('Pruning settings changed; choose another --work-dir')
        log('[prune] reusing completed candidate scan')
        return registry
    (args.work_dir/'prune_candidates').mkdir(parents=True, exist_ok=True)
    block_dir = args.work_dir/'prune_blocks'
    block_dir.mkdir(parents=True, exist_ok=True)
    settings_path = args.work_dir/'pruning_settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Pruning settings changed; choose another --work-dir')
    settings_path.write_text(json.dumps(dict(fingerprint=fingerprint, settings=settings), indent=2))
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=settings['revision'],
                                                dtype=torch.float16, device_map={'': args.device})
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    original_specs = target_linear_specs(model)
    hidden, kwargs = capture_first_inputs(model, batches)
    del batches
    rows = []
    for index, layer in enumerate(model.model.layers):
        if args.max_layers is not None and index >= args.max_layers:
            break
        started = time.monotonic()
        mlp = layer.mlp
        original = {name: getattr(mlp, name) for name in ['gate_proj', 'up_proj', 'down_proj']}
        old_width = mlp.gate_proj.out_features
        widths = [w for w in sorted(set(args.prune_widths), reverse=True) if w < old_width]
        done_path = block_dir/f'layer{index}.json'
        cached = done_path.exists() and not args.rebuild
        gram = None if cached else torch.zeros((old_width, old_width), device=args.device, dtype=torch.float32)

        def collect(module, inputs):
            if sample_index < args.samples:
                z = inputs[0].detach().reshape(-1, old_width).float()
                gram.addmm_(z.T, z, beta=1, alpha=1/(args.samples*args.seqlen))

        handle = None if cached else mlp.down_proj.register_forward_pre_hook(collect)
        next_hidden = []
        try:
            for sample_index, value in enumerate(hidden):
                next_hidden.append(layer(value, **kwargs).detach())
        finally:
            if handle is not None:
                handle.remove()
        if cached:
            layer_rows = json.loads(done_path.read_text())['rows']
        else:
            scores = channel_importance(original['down_proj'].weight, gram)
            layer_rows = []
            for width in widths:
                tick = time.monotonic()
                keep = keep_channels(scores, width)
                naive = cropped_mlp_modules(mlp, keep)
                install_mlp_modules(mlp, naive)
                naive_error = block_error(layer, hidden[-args.holdout_samples:], next_hidden[-args.holdout_samples:], kwargs)
                corrected_weight = compensated_down_weight(original['down_proj'].weight, gram, keep, args.prune_damp)
                corrected = linear_from_weight(corrected_weight, original['down_proj'].bias)
                mlp.down_proj = corrected
                corrected_error = block_error(layer, hidden[-args.holdout_samples:], next_hidden[-args.holdout_samples:], kwargs)
                use_correction = corrected_error <= naive_error
                if not use_correction:
                    mlp.down_proj = naive['down_proj']
                removed = 3*mlp.gate_proj.in_features*(old_width-width)
                row = dict(index=index, width=width, removed_weights=removed,
                           error=min(naive_error, corrected_error), naive_error=naive_error,
                           corrected_error=corrected_error, compensated=use_correction)
                state = {name: cpu_tree(getattr(mlp, name).state_dict()) for name in original}
                torch.save(dict(state=state, keep_indices=keep.int().cpu(), metrics=row), candidate_path(args,index,width))
                layer_rows.append(row)
                install_mlp_modules(mlp, original)
                log(f'[prune-candidate] block {index} width {width}: naive={naive_error:.6g}, compensated={corrected_error:.6g}, {time.monotonic()-tick:.1f}s')
                del naive, corrected, corrected_weight, state
            torch.save(dict(inputs=cpu_tree(hidden[-args.holdout_samples:]),
                            references=cpu_tree(next_hidden[-args.holdout_samples:]), kwargs=cpu_tree(kwargs)),
                       block_dir/f'layer{index}.pt')
            done_path.write_text(json.dumps(dict(rows=layer_rows), indent=2))
        rows.extend(layer_rows)
        hidden = next_hidden
        del next_hidden, gram
        gc.collect()
        torch.cuda.empty_cache()
        log(f'[prune-scan] block {index+1}/36 done ({time.monotonic()-started:.1f}s)')
    registry = dict(fingerprint=fingerprint, settings=settings, original_specs=original_specs, rows=rows)
    registry_path.write_text(json.dumps(registry, indent=2))
    del model, hidden, kwargs, original, mlp, layer
    gc.collect()
    torch.cuda.empty_cache()
    return registry


def choose_widths(registry, args):
    grouped = {}
    for row in registry['rows']:
        grouped.setdefault(row['index'], []).append(row)
    hulls = {}
    for index, rows in grouped.items():
        old_width = registry['original_specs'][f'model.layers.{index}.mlp.gate_proj']['out_features']
        points = [dict(index=index, width=old_width, removed_weights=0, error=0.0, compensated=False)]
        points += sorted(rows, key=lambda r:r['removed_weights'])
        hull = []
        for point in points:
            while len(hull) >= 2:
                a,b = hull[-2:]
                previous = (b['error']-a['error'])/(b['removed_weights']-a['removed_weights'])
                following = (point['error']-b['error'])/(point['removed_weights']-b['removed_weights'])
                if previous < following:
                    break
                hull.pop()
            hull.append(point)
        hulls[index] = hull
    current = {index:0 for index in hulls}
    queue = []

    def push(index):
        pos = current[index]
        if pos+1 < len(hulls[index]):
            a,b = hulls[index][pos:pos+2]
            heapq.heappush(queue, ((b['error']-a['error'])/(b['removed_weights']-a['removed_weights']),index,pos+1))

    for index in hulls:
        push(index)
    total = sum(registry['original_specs'][n]['in_features']*registry['original_specs'][n]['out_features']
                for n in target_linear_names(registry['original_specs']))
    removed = 0
    while removed/total < args.pruning_rate and queue:
        _, index, pos = heapq.heappop(queue)
        removed += hulls[index][pos]['removed_weights']-hulls[index][pos-1]['removed_weights']
        current[index] = pos
        push(index)
    chosen = {index:hulls[index][pos] for index,pos in current.items() if pos}
    log(f'[prune-select] removed {removed:,}/{total:,} weights ({removed/total:.6%}), {len(chosen)} cropped blocks')
    if removed/total < args.pruning_rate:
        log('[prune-select] requested rate unavailable with smoke-test candidates')
    return chosen


@torch.no_grad()
def ensure_pruned(args, loader):
    registry = scan_pruning(args)
    output_dir = args.work_dir/'pruned'
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir/'manifest.json'
    if manifest_path.exists() and not args.rebuild:
        saved = json.loads(manifest_path.read_text())
        if (saved.get('scope') == SCOPE and
                saved.get('pruning_fingerprint') == registry['fingerprint'] and
                saved.get('reload_check',{}).get('bitwise_equal_logits')):
            log('[prune] using verified saved FP16 pruned checkpoint')
            return saved
    chosen = choose_widths(registry,args)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=registry['settings']['revision'],
                                                dtype=torch.float16, device_map={'': args.device})
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    for index,row in chosen.items():
        payload = torch.load(candidate_path(args,index,row['width']), map_location='cpu', weights_only=True)
        for name,state in payload['state'].items():
            module = linear_from_weight(state['weight'].to(args.device), state.get('bias'))
            set_module(model,f'model.layers.{index}.mlp.{name}',module)
        model.model.layers[index].mlp.intermediate_size = row['width']
    pruned_specs = target_linear_specs(model)
    counts = pruning_counts(registry['original_specs'],pruned_specs)
    before = original_bytes(registry['original_specs'])
    after = target_payload(model, registry['original_specs'])
    formats = {name:dict(kind='dense',**spec) for name,spec in pruned_specs.items()}
    manifest = dict(model=args.model,mode='pruned',calibration=registry['settings'],
                    pruning_fingerprint=registry['fingerprint'], original_specs=registry['original_specs'],
                    pruned_specs=pruned_specs,original_linear_bytes=before,compressed_linear_bytes=after,
                    compression_rate=1-after/before, **accounting_metadata(registry['original_specs']), formats=formats,widths={str(i):r for i,r in chosen.items()},**counts)
    save_file({key:value.detach().cpu().contiguous() for key,value in model.state_dict().items()
               if key!='model.embed_tokens.weight'},str(output_dir/'model.safetensors'))
    manifest['checkpoint_file_bytes'] = (output_dir/'model.safetensors').stat().st_size
    manifest['storage_audit'] = checkpoint_audit(output_dir/'model.safetensors', registry['original_specs'])
    if manifest['storage_audit']['compressed_linear_bytes'] != after:
        raise AssertionError('Serialized pruned payload differs from measured module state')
    manifest_path.write_text(json.dumps(manifest,indent=2))
    log(f'[prune] physical pruning rate {counts["pruning_rate"]:.6%}, {after:,} bytes')
    ids = torch.tensor([[1,42,314,159,265,35,897,93]],device=args.device)
    reference = model(ids).logits.detach().cpu()
    del model,module,payload
    gc.collect()
    torch.cuda.empty_cache()
    restored = loader(args, mode='pruned')
    actual = restored(ids).logits.detach().cpu()
    if not torch.equal(reference,actual):
        raise AssertionError(f'Pruned checkpoint reload changed logits by {(reference.float()-actual.float()).abs().max().item()}')
    manifest['reload_check'] = dict(bitwise_equal_logits=True,max_abs_difference=0.0)
    manifest_path.write_text(json.dumps(manifest,indent=2))
    log('[prune] checkpoint reload passed: bitwise identical logits')
    del restored
    gc.collect()
    torch.cuda.empty_cache()
    return manifest
