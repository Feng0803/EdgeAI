"""Case 1: NF4 baseline + calibration-selected, genuinely packed GPTQ INT3.

The instructor's run_eval.py and evaluate.eval_ppl_wikitext2 are unchanged.
Only their model loader / compression callback are supplied by this wrapper.
Calibration and layer selection use WikiText-2 TRAIN, never TEST.
"""

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
from gptq_mixed import (Packed3BitLinear, SharedNF4Embedding, install_shared_nf4,
                        module_payload, original_bytes, quantize_gptq3,
                        reconstruction_errors, set_module, target_linear_specs)


MODEL_REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'

def log(message):
    print(message, file=sys.stderr, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--holdout-samples", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=512, help="calibration only; official PPL remains 2048")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--targets", nargs="+", default=["gate_proj", "up_proj", "down_proj"],
                        choices=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--target-rate", type=float, default=0.752,
                        help="decoder Linear tensor payload target (lm_head excluded), with margin above 75%%")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).parent / "artifacts/case1_gptq")
    parser.add_argument("--max-layers", type=int, default=None, help="limit scan for a smoke test only")
    parser.add_argument("--rebuild", action="store_true", help="ignore existing scan candidates")
    parser.add_argument("--eval-only", action="store_true", help="reload saved packed model and run official evaluation")
    parser.add_argument("--skip-eval", action="store_true", help="build and verify checkpoint only")
    args = parser.parse_args()
    if args.samples < 1 or args.holdout_samples < 1 or args.seqlen < 2:
        parser.error("positive calibration samples, holdout samples, and seqlen >= 2 required")
    if not 0 < args.target_rate < 1:
        parser.error("--target-rate must be in (0,1)")
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


@torch.no_grad()
def scan(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=MODEL_REVISION)
    batches, blocks = train_samples(tokenizer, args)
    fingerprint_data = dict(model=args.model, revision=AutoConfig.from_pretrained(args.model, revision=MODEL_REVISION)._commit_hash,
                            samples=args.samples, holdout=args.holdout_samples,
                            seqlen=args.seqlen, group_size=args.group_size,
                            damp=args.damp, targets=args.targets, seed=args.seed, blocks=blocks,
                            implementation="affine-gptq-v1")
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    scan_dir = args.work_dir / "candidates"
    scan_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = scan_dir / "settings.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Calibration settings changed: choose another --work-dir")
    metadata_path.write_text(json.dumps(dict(fingerprint=fingerprint, settings=fingerprint_data), indent=2))
    log("[scan] loading original FP16 model on a single GPU; CPU is not used for inference offloading")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map={"": args.device}, revision=MODEL_REVISION)
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    specs = target_linear_specs(model)
    hidden, layer_kwargs = capture_first_inputs(model, batches)
    del batches
    rows = []
    for index, layer in enumerate(model.model.layers):
        if args.max_layers is not None and index >= args.max_layers:
            break
        t0 = time.monotonic()
        modules = {name: module for name, module in layer.named_modules()
                   if name.rsplit(".", 1)[-1] in args.targets}
        paths = {name: scan_dir / (f"model.layers.{index}.{name}.pt") for name in modules}
        missing = [name for name in modules if args.rebuild or not paths[name].exists()]
        grams, heldout = {}, {}
        # gate/up and q/k/v share inputs; one Gram matrix per shared input suffices.
        def family(name):
            suffix = name.rsplit(".", 1)[-1]
            return "mlp_input" if suffix in ("gate_proj", "up_proj") else "attn_input" if suffix in ("q_proj", "k_proj", "v_proj") else name

        handles = []
        families = {}
        for name in missing:
            key = family(name)
            if key in families:
                continue
            families[key] = name
            width = modules[name].in_features
            grams[key] = torch.zeros((width, width), dtype=torch.float32, device=args.device)
            heldout[key] = []

            def collect(module, inputs, key=key):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                if sample_index < args.samples:
                    xf = x.float()
                    grams[key].addmm_(xf.T, xf, beta=1, alpha=1 / (args.samples * args.seqlen))
                else:
                    # Uniformly subsample fit-independent tokens for output reconstruction.
                    heldout[key].append(x[::max(1, x.shape[0] // 128)].clone())

            handles.append(modules[name].register_forward_pre_hook(collect))
        next_hidden = []
        try:
            for sample_index, value in enumerate(hidden):
                next_hidden.append(layer(value, **layer_kwargs).detach())
        finally:
            for handle in handles:
                handle.remove()
        del hidden
        hidden = next_hidden
        for name, module in modules.items():
            full_name = f"model.layers.{index}.{name}"
            if name not in missing:
                payload = torch.load(paths[name], map_location="cpu", weights_only=True)
                rows.append(payload["metrics"])
                del payload
                continue
            start = time.monotonic()
            key = family(name)
            candidate = quantize_gptq3(module.weight, grams[key], args.group_size, args.damp)
            error3, error4 = reconstruction_errors(module.weight, candidate, torch.cat(heldout[key]))
            if module.bias is not None:
                candidate.bias = module.bias.detach().clone()
            metrics = dict(name=full_name, error3=error3, error4=error4,
                           delta_error=error3-error4, packed_bytes=module_payload(candidate))
            torch.save(dict(spec=candidate.spec(), state={k: v.cpu() for k, v in candidate.state_dict().items()},
                            metrics=metrics), paths[name])
            rows.append(metrics)
            log(f"[candidate] {full_name}: E3={error3:.6g} E4={error4:.6g} delta={error3-error4:.6g}, {time.monotonic()-start:.1f}s")
            del candidate
        log(f"[scan] layer {index+1}/{len(model.model.layers)} done in {time.monotonic()-t0:.1f}s")
        del grams, heldout, next_hidden
        torch.cuda.empty_cache()
    del model, hidden, layer_kwargs
    gc.collect()
    torch.cuda.empty_cache()
    return specs, rows, fingerprint_data


def candidate_module(payload, device):
    s = payload["spec"]
    state = {k: v.to(device) for k, v in payload["state"].items()}
    return Packed3BitLinear(state["packed"], state["scales"], state["zeros"],
                            s["in_features"], s["out_features"], s["group_size"], state.get("bias"))


@torch.no_grad()
def build(args, specs, rows, settings):
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
                                                device_map={"": args.device}, quantization_config=cfg, revision=MODEL_REVISION)
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    install_shared_nf4(model)
    baseline_bytes = target_payload(model, specs)
    before = original_bytes(specs)
    after = baseline_bytes
    ranking = []
    for row in rows:
        saving = module_payload(model.get_submodule(row["name"])) - row["packed_bytes"]
        if saving > 0:
            ranking.append(dict(**row, saved_bytes=saving, score=row["delta_error"] / saving))
    ranking.sort(key=lambda row: row["score"])
    selected = []
    log(f"[storage] baseline decoder Linear packed NF4: {before:,} -> {after:,} bytes ({1-after/before:.6%})")
    for row in ranking:
        if 1 - after / before >= args.target_rate:
            break
        payload = torch.load(args.work_dir / "candidates" / (row["name"] + ".pt"), map_location="cpu", weights_only=True)
        set_module(model, row["name"], candidate_module(payload, args.device))
        after -= row["saved_bytes"]
        selected.append(row)
        log(f"[select] {row['name']}: delta={row['delta_error']:.6g} saving={row['saved_bytes']:,}, rate={1-after/before:.6%}")
    measured = target_payload(model, specs)
    if measured != after:
        raise AssertionError(f"Storage accounting mismatch: {measured} != {after}")
    formats = {}
    for name in specs:
        module = model.get_submodule(name)
        formats[name] = module.spec() if isinstance(module, Packed3BitLinear) else dict(kind="nf4", **specs[name])
    manifest = dict(model=args.model, original_specs=specs, formats=formats,
                    original_linear_bytes=before, compressed_linear_bytes=after,
                    compression_rate=1-after/before, target_rate=args.target_rate,
                    target_met=1-after/before >= args.target_rate,
                    **accounting_metadata(specs),
                    calibration=settings, selected=selected, ranking=ranking)
    checkpoint = args.work_dir / "model.safetensors"
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(checkpoint))
    manifest["checkpoint_file_bytes"] = checkpoint.stat().st_size
    manifest['storage_audit'] = checkpoint_audit(checkpoint, specs)
    if manifest['storage_audit']['compressed_linear_bytes'] != after:
        raise AssertionError('Serialized target payload differs from measured module state')
    (args.work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"[checkpoint] {checkpoint}, {checkpoint.stat().st_size:,} bytes; selected {len(selected)} layers")
    if not manifest["target_met"]:
        log("[storage] WARNING: available candidates did not meet requested compression target")
    # Keep one fixed input/output as an independent serialization correctness check.
    probe_ids = torch.tensor([[1, 42, 314, 159, 265, 35, 897, 93]], device=args.device)
    probe_logits = model(probe_ids).logits.detach().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    restored = load_checkpoint(args)
    actual = restored(probe_ids).logits.detach().cpu()
    difference = (probe_logits.float()-actual.float()).abs().max().item()
    if not torch.equal(probe_logits, actual):
        raise AssertionError(f"Checkpoint reload changed logits: max difference={difference}")
    manifest["reload_check"] = dict(bitwise_equal_logits=True, max_abs_difference=difference)
    (args.work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log("[checkpoint] reload check passed: bitwise identical logits")
    return restored


@torch.no_grad()
def load_checkpoint(args):
    manifest = json.loads((args.work_dir / "manifest.json").read_text())
    require_current_scope(manifest)
    state = load_file(str(args.work_dir / "model.safetensors"))
    config = AutoConfig.from_pretrained(manifest["model"], revision=MODEL_REVISION)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    for name, spec in manifest["formats"].items():
        prefix = name + "."
        if spec["kind"] == "gptq3":
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=MODEL_REVISION)
    manifest = json.loads((args.work_dir / "manifest.json").read_text())
    evaluate.load_model = lambda *unused, **kwargs: (model, tokenizer)
    evaluate.compression_rate = lambda model: manifest["compression_rate"]
    runner = Path(__file__).with_name("run_eval.py")
    sys.argv = [str(runner), "--case", "1", "--rate", str(args.target_rate),
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
    result = dict(case=1, ppl=ppl, compression_rate=manifest["compression_rate"],
                  original_linear_bytes=manifest["original_linear_bytes"],
                  compressed_linear_bytes=manifest["compressed_linear_bytes"],
                  case1_pass=manifest["compression_rate"] > 0.75 and ppl < 15,
                  **accounting_metadata(manifest['original_specs']),
                  storage_audit=manifest['storage_audit'],
                  ppl_provenance='fresh official run_eval.py evaluation',
                  evaluation_seconds=time.monotonic()-started,
                  evaluator_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (runner, runner.with_name("evaluate.py"))})
    (args.work_dir / "result.json").write_text(json.dumps(result, indent=2))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        model = load_checkpoint(args)
    else:
        specs, rows, settings = scan(args)
        model = build(args, specs, rows, settings)
    if not args.skip_eval:
        run_official_evaluation(model, args)


if __name__ == "__main__":
    main()
