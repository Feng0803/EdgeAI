"""Part 3: fixed calibration-free NF4 fake PTQ, followed by official evaluation."""

import argparse
import contextlib
from datetime import datetime
import gc
import hashlib
import io
import json
from pathlib import Path
import re
import runpy
import sys
import time
from zoneinfo import ZoneInfo

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from nf4 import POLICY, fake_quantize_decoder


MODEL_ID = 'Qwen/Qwen3-4B'
MODEL_REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
HERE = Path(__file__).resolve().parent


def log(message):
    print(message, file=sys.stderr, flush=True)


def policy_hash():
    return hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['fp16', 'nf4'], default='nf4')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rows-per-chunk', type=int, default=256,
                        help='execution memory only; does not change block size or NF4 mapping')
    parser.add_argument('--work-dir', type=Path, default=HERE/'artifacts')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--skip-eval', action='store_true')
    args = parser.parse_args()
    if args.rows_per_chunk < 1:
        parser.error('positive execution chunk size required')
    return args


def load_original(args, revision):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=revision,
                                                dtype=torch.float16, device_map={'': args.device})
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    return model


def shared_matrix_is_tied(model):
    return model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()


@torch.no_grad()
def build_model(args, revision):
    output_dir = args.work_dir/args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(model=MODEL_ID, revision=revision, mode=args.mode, dtype='float16',
                    created_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
                    policy=POLICY if args.mode == 'nf4' else None,
                    policy_sha256=policy_hash() if args.mode == 'nf4' else None,
                    quantization_locked_before_dataset_access=True,
                    datasets_read_during_quantization=[], calibration_forward_passes=0,
                    actual_storage='FP16', compression_claim=False)
    # Persist the frozen mapping before original model weights or evaluation data.
    (output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    model = load_original(args, revision)
    if not shared_matrix_is_tied(model):
        raise AssertionError('Expected original tied embedding/lm_head')
    if args.mode == 'nf4':
        log('[quantize] frozen policy: NF4 W4A16, block64, FP32 absmax; no calibration data/forwards')

        def forbidden_forward(module, inputs):
            raise RuntimeError('Model forwards are forbidden during weight-only quantization')

        hooks = [module.register_forward_pre_hook(forbidden_forward) for module in model.modules()]
        started = time.monotonic()

        def progress(row, count):
            if count == 1 or count % 7 == 0:
                log(f'[quantize] {count}/252 {row["name"]}: weight MSE={row["mse"]:.6g}')

        try:
            statistics = fake_quantize_decoder(model, args.rows_per_chunk, progress)
        finally:
            for hook in hooks:
                hook.remove()
        if len(statistics) != 252:
            raise AssertionError(f'Expected all 252 decoder Linear layers, got {len(statistics)}')
        if policy_hash() != manifest['policy_sha256']:
            raise AssertionError('Quantization policy changed')
        manifest.update(quantized_linear_count=len(statistics),
                        quantized_weights=sum(row['weights'] for row in statistics),
                        weight_mse=sum(row['sse'] for row in statistics)/sum(row['weights'] for row in statistics),
                        quantization_seconds=time.monotonic()-started, layer_statistics=statistics)
        log('[quantize] complete; all forwards and dataset access so far were unnecessary for quantization')
        # Officially allowed fake quantization: original architecture and FP16 storage.
        checkpoint = output_dir/'model'
        model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size='3GB')
        manifest['checkpoint_file_bytes'] = sum(p.stat().st_size for p in checkpoint.glob('*.safetensors'))
        probe = torch.tensor([[1, 42, 314, 159, 265, 35, 897, 93]], device=args.device)
        reference = model(probe).logits.detach().cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float16, device_map={'': args.device})
        model.eval().requires_grad_(False)
        model.config.use_cache = False
        actual = model(probe).logits.detach().cpu()
        if not torch.equal(reference, actual):
            raise AssertionError(f'Reload changed logits by {(reference.float()-actual.float()).abs().max().item()}')
        if not shared_matrix_is_tied(model):
            raise AssertionError('Original embedding/lm_head tying lost on reload')
        manifest['reload_check'] = dict(bitwise_equal_logits=True, tied_embedding_head=True,
                                        max_abs_difference=0.0)
        log('[checkpoint] fake-quantized FP16 checkpoint reload passed: bitwise identical logits')
    (output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if args.mode == 'nf4':
        (HERE/'results/quantization_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return model, manifest


@torch.no_grad()
def load_saved(args):
    output_dir = args.work_dir/args.mode
    manifest = json.loads((output_dir/'manifest.json').read_text())
    if args.mode == 'fp16':
        return load_original(args, manifest['revision']), manifest
    if manifest['policy_sha256'] != policy_hash():
        raise ValueError('Saved model policy differs from this frozen implementation')
    model = AutoModelForCausalLM.from_pretrained(output_dir/'model', dtype=torch.float16,
                                                device_map={'': args.device})
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    if not shared_matrix_is_tied(model):
        raise AssertionError('Embedding/head sharing lost')
    return model, manifest


def performance_score(ppl):
    for threshold, score in [(15, 15), (16, 12), (17, 9), (18, 6), (19, 3)]:
        if ppl <= threshold:
            return score
    return 0


def run_official_evaluation(model, manifest, args):
    # The evaluation module (and its datasets import) is reached only after the
    # quantizer has finished and its policy/checkpoint have been fixed.
    import evaluate
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=manifest['revision'])
    if args.mode == 'nf4':
        tokenizer.save_pretrained(args.work_dir/'nf4/model')
    evaluate.load_model = lambda *unused, **kwargs: (model, tokenizer)
    # A fake-quantized FP16 model has no real storage compression to report.
    evaluate.compression_rate = lambda unused: 0.0
    runner = HERE/'run_eval.py'
    sys.argv = [str(runner), '--model', MODEL_ID, '--dtype', 'float16', '--seed', '0']
    log('[eval] original run_eval.py; WikiText-2 TEST, seqlen=2048; quantization already locked')
    started = time.monotonic()
    segments = 0

    def progress(module, inputs, output):
        nonlocal segments
        segments += 1
        if segments % 16 == 0:
            log(f'[eval] {segments} segments ({time.monotonic()-started:.1f}s)')

    hook = model.register_forward_hook(progress)
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            runpy.run_path(str(runner), run_name='__main__')
    finally:
        hook.remove()
    ppl = float(re.search(r'^ppl: (\S+)$', captured.getvalue(), re.MULTILINE).group(1))
    print(f'Part3 {args.mode}\nPPL: {ppl:.4f}', flush=True)
    result = dict(mode=args.mode, model=MODEL_ID, revision=manifest['revision'], ppl=ppl,
                  adaptquant_assignment_reference_ppl=13.98, ppl_difference_vs_reference=ppl-13.98,
                  performance_score=performance_score(ppl) if args.mode == 'nf4' else None,
                  evaluation_seconds=time.monotonic()-started, evaluation_segments=segments,
                  official_stdout=captured.getvalue(), actual_weight_storage='FP16',
                  policy_sha256=manifest['policy_sha256'],
                  evaluator_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [runner, HERE/'evaluate.py']})
    (args.work_dir/args.mode/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (HERE/'results'/f'{args.mode}_result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Single CUDA GPU required for the assignment evaluation')
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (HERE/'results').mkdir(exist_ok=True)
    source_path = args.work_dir/'source.json'
    if not source_path.exists():
        source = dict(model=MODEL_ID, revision=AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)._commit_hash,
                      original_dtype='float16')
        source_path.write_text(json.dumps(source, indent=2)+'\n')
    source = json.loads(source_path.read_text())
    if args.eval_only:
        model, manifest = load_saved(args)
    else:
        model, manifest = build_model(args, source['revision'])
    if not args.skip_eval:
        run_official_evaluation(model, manifest, args)


if __name__ == '__main__':
    main()
