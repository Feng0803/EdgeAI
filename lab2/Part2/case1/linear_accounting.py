"""TA-defined Linear payload accounting; lm_head is outside the target scope.

original_specs/formats still include lm_head so checkpoints reconstruct the
complete model. Only accounting excludes it. Never deduplicate distinct Linear
operations merely because they share parameter storage.
"""

import json
from pathlib import Path
import struct


SCOPE = 'original nn.Linear excluding lm_head; embedding and normalization excluded'


def target_linear_names(specs):
    return tuple(name for name in specs if name != 'lm_head')


def original_bytes(specs):
    return sum(2 * (specs[n]['in_features'] * specs[n]['out_features'] +
                    (specs[n]['out_features'] if specs[n].get('bias', False) else 0))
               for n in target_linear_names(specs))


def target_payload(model, specs):
    # All factor weights, uncompressed rows, scales, zeros, permutations,
    # codebooks and serialized quantization state belong to their parent Linear.
    return sum(t.numel() * t.element_size()
               for n in target_linear_names(specs)
               for t in model.get_submodule(n).state_dict().values())


def accounting_metadata(specs):
    return dict(scope=SCOPE, target_linear_count=len(target_linear_names(specs)),
                excluded_linear_names=[n for n in specs if n == 'lm_head'])


def require_current_scope(manifest):
    if manifest.get('scope') != SCOPE:
        raise ValueError('Old accounting scope: run Part2/recount_results.py before --eval-only')


def checkpoint_audit(checkpoint, specs):
    checkpoint = Path(checkpoint)
    with checkpoint.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError('Invalid safetensors header')
        header = json.loads(stream.read(struct.unpack('<Q', raw)[0]))
    prefixes = tuple(n + '.' for n in target_linear_names(specs))
    compressed = 0
    excluded_head = 0
    for name, value in header.items():
        if name == '__metadata__':
            continue
        size = value['data_offsets'][1] - value['data_offsets'][0]
        if name.startswith(prefixes):
            compressed += size
        elif name.startswith('lm_head.'):
            excluded_head += size
    before = original_bytes(specs)
    return dict(**accounting_metadata(specs), original_linear_bytes=before,
                compressed_linear_bytes=compressed, compression_rate=1-compressed/before,
                excluded_lm_head_bytes=excluded_head,
                checkpoint_file_bytes=checkpoint.stat().st_size,
                verified_from_safetensors_header=True)


def pruning_counts(original_specs, pruned_specs):
    names = target_linear_names(original_specs)
    original = sum(original_specs[n]['in_features'] * original_specs[n]['out_features'] for n in names)
    remaining = sum(pruned_specs[n]['in_features'] * pruned_specs[n]['out_features'] for n in names)
    return dict(original_linear_weights=original, remaining_linear_weights=remaining,
                removed_linear_weights=original-remaining, pruning_rate=1-remaining/original)
