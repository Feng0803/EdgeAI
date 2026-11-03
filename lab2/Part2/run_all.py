"""Run all four real compression procedures and the supplied PPL evaluator."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch
from linear_accounting import checkpoint_audit, require_current_scope

HERE = Path(__file__).resolve().parent
SCRIPTS = ['case1_gptq_mixed.py', 'case2_gptq_mixed.py', 'case3_svd.py', 'case4_prune_gptq.py']


def storage_audit(checkpoint, manifest):
    require_current_scope(manifest)
    audit = checkpoint_audit(checkpoint, manifest['original_specs'])
    if (audit['original_linear_bytes'] != manifest['original_linear_bytes'] or
            audit['compressed_linear_bytes'] != manifest['compressed_linear_bytes']):
        raise AssertionError('Actual checkpoint tensor bytes differ from compression accounting')
    if audit['compression_rate'] != manifest['compression_rate']:
        raise AssertionError('Incorrect compression ratio')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--eval-only', action='store_true', help='requires previously generated checkpoints for all cases')
    args = parser.parse_args()
    if sys.version_info[:2] != (3,11) or torch.__version__ != '2.5.1+cu121':
        raise RuntimeError('Required base environment: Python 3.11, torch 2.5.1+cu121')
    if not torch.cuda.is_available():
        raise RuntimeError('One CUDA GPU is required')
    logs = HERE/'logs'
    results_dir = HERE/'results'
    logs.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    hashes = {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['evaluate.py','run_eval.py']}
    results = []
    started = time.monotonic()
    run = dict(started_at=datetime.now(timezone.utc).isoformat(), python=sys.version,
               torch=torch.__version__, gpu=torch.cuda.get_device_name(torch.device(args.device)),
               eval_only=args.eval_only, evaluator_sha256=hashes)
    for case,script in enumerate(SCRIPTS,1):
        cwd = HERE/f'case{case}'
        work = HERE/'artifacts'/f'case{case}'
        output = work if case == 1 else work/'mixed'
        result_path = output/'result.json'
        for name in hashes:
            if (cwd/name).read_bytes() != (HERE/name).read_bytes():
                raise AssertionError(f'Case {case} evaluator changed')
        command = [sys.executable,'-u',str(cwd/script),'--work-dir',str(work),'--device',args.device]
        if case != 1:
            command += ['--mode','mixed']
        if args.eval_only:
            command += ['--eval-only']
        result_path.unlink(missing_ok=True)
        print(f'[Part2] Case {case}: {"evaluate" if args.eval_only else "compress and evaluate"}; log: logs/case{case}.log',file=sys.stderr,flush=True)
        with (logs/f'case{case}.log').open('w') as log:
            subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
        result = json.loads(result_path.read_text())
        manifest = json.loads((output/'manifest.json').read_text())
        if not result[f'case{case}_pass'] or not math.isfinite(result['ppl']):
            raise AssertionError(f'Case {case} failed its assignment thresholds: {result}')
        if result['evaluator_sha256'] != hashes:
            raise AssertionError('PPL was not produced by the supplied evaluator')
        audit = storage_audit(output/'model.safetensors',manifest)
        if case == 3 and not .495 <= audit['compression_rate'] <= .505:
            raise AssertionError('Case 3 must be within the TA-defined 49.5%-50.5% range')
        result['storage_audit'] = audit
        result['reload_check'] = manifest['reload_check']
        result['revision'] = manifest['calibration']['revision']
        (results_dir/f'case{case}_result.json').write_text(json.dumps(result,indent=2)+'\n')
        (results_dir/f'case{case}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        results.append(result)
        print(f'[Part2] Case {case}: passed',file=sys.stderr,flush=True)
    if len({r['revision'] for r in results}) != 1:
        raise AssertionError('All cases must use the same original model revision')
    run.update(elapsed_seconds=time.monotonic()-started,cases=results,
               scope=results[0]['scope'], complete=True, all_cases_pass=True)
    (results_dir/'all_cases.json').write_text(json.dumps(run,indent=2)+'\n')
    for result in results:
        print(f'Case {result["case"]}\nCompression Ratio: {result["compression_rate"]:.8%}\nPPL: {result["ppl"]:.4f}\n')


if __name__ == '__main__':
    main()
