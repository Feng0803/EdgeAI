# Lab 2 Part 3 — AdaptQuant reproduction (314551126)

This folder contains a target-model calibration-free reproduction of
[AdaptQuant](https://openreview.net/forum?id=ZGeR4gkEZq) for `Qwen/Qwen3-4B`.
It quantizes all 252 decoder projections (`q/k/v/o/gate/up/down_proj`) and their
input activations.  Embedding, tied `lm_head`, and normalization are excluded,
as specified by the TA.

## Final result

| Model / policy | WikiText-2 PPL | Weight bits | Activation bits | (W+A)/2 |
|---|---:|---:|---:|---:|
| Original FP16 baseline | 13.6378 | 16 | 16 | 16 |
| AdaptQuant reproduction | **14.6440** | **5.045517** | **5.145720** | **5.095619** |

The effective-bit calculation includes the MX element payload, one 8-bit E8M0
scale per 32 values.  Per the TA's latest clarification, AdaptQuant's 5-byte
per-tile format-tag/offset metadata is **not** included.  The T4 does not
implement Blackwell MX MMA, so this submission uses the fake-quantization
option explicitly allowed by the assignment: values are quantized according
to MX and dequantized before the ordinary FP16 Linear operation.  It verifies
model quality and logical bit rate; it does not claim actual packed checkpoint
size or T4 speedup.

## Reproduce the submitted result

From this directory:

```bash
bash run.sh
```

`run.sh` creates/reuses `.venv`, installs `requirements.txt`, runs the fast CPU
checks, loads the already frozen `frozen_policy.json`, quantizes Qwen3-4B, and
uses the provided `evaluate.py` function on WikiText-2 test with sequence
length 2048.  A single T4 16 GB is sufficient.  Set a different device if
needed:

```bash
bash run.sh --device cuda:1
```

The default path does **not** search for layers, tune thresholds, read a
calibration dataset, or derive the policy again.  The expected run time is
longer than uniform fake quantization because every activation tile is
classified and fake-quantized during every Linear call.

## Method and calibration-free boundary

For each 16x32 operand tile, the implementation computes one statistic:

```text
stat = log2(tile_absmax / tile_RMS)
```

Two frozen thresholds select MXFP4, MXFP6, or MXFP8 for that tile.  Each row's
32 values share an E8M0 scale.  The thresholds differ by operand
(weight/activation) and projection type, but are unchanged across the 36
Qwen3 decoder blocks.

The thresholds were derived once, offline, on the non-target reference model
`unsloth/Llama-3.1-8B` (revision
`3f0d51f8e5640f98f1a96ea9044a0e55c0a83814`) using the first 2048
WikiText-2 train tokens.  The
derivation compares reconstruction error with NVFP4 and solves the paper's
threshold grid.  To satisfy this lab's new combined `(W bits + A bits)/2 <=
5.3` rule, the solver uses a single 0.7 dB reference-model error slack and a
5.17-bit reference budget.  This is the only assignment-specific adjustment
to the paper procedure.  It is decided solely on the reference model.

Qwen3-4B data, activations, PPL, or layer errors are never used to choose or
alter the thresholds.  The target model is only quantized with the frozen
file and then evaluated.  Therefore the target procedure is calibration-free,
while the origin of the reusable offline rule is explicit.

## Effective-bit calculation

For either weights or activations:

```text
MXFP4 = 4 + 8/32 = 4.25 bits/value
MXFP6 = 6 + 8/32 = 6.25 bits/value
MXFP8 = 8 + 8/32 = 8.25 bits/value

operand bits = sum(tile_share[format] x format_bits)
final rate   = (weight bits + activation bits) / 2
```

The five-byte tile format-tag/offset metadata is explicitly excluded according
to the TA clarification; the 8-bit E8M0 scale overhead remains included.

The verified Qwen3-4B run measured:

```text
weights:     60.224130% MXFP4, 39.775870% MXFP6, 0% MXFP8
             0.60224130*4.25 + 0.39775870*6.25
             = 5.045517 bits/value

activations: 55.213991% MXFP4, 44.786009% MXFP6, 0% MXFP8
             0.55213991*4.25 + 0.44786009*6.25
             = 5.145720 bits/value

final:       (5.045517 + 5.145720) / 2 = 5.095619 <= 5.3
```

The submitted run measures weight tile shares during weight quantization and
activation tile shares across the full official evaluation.  It writes all
counts, shares, hashes, and results to `results/adaptquant_result.json`.

## Optional: reproduce the offline policy derivation

This step is submitted for provenance but is not run by `run.sh`:

```bash
python derive_policy.py \
  --reference-model unsloth/Llama-3.1-8B \
  --reference-revision 3f0d51f8e5640f98f1a96ea9044a0e55c0a83814 \
  --sequence-length 2048 \
  --bit-limit 5.17 \
  --output frozen_policy.json
```

The TA stated that the reference model may be assumed available.  A first run
requires enough host RAM for the reference model and writes
`reference_histograms.pt`; later threshold solving can reuse that cache.  The
reference model is never required for the normal grading command.

## Files

- `run.sh`: one-command submitted-result reproduction; uses frozen policy.
- `run_adaptquant.py`: Qwen3 weight quantization, activation hooks, bit
  accounting, and official PPL evaluation.
- `mx_formats.py`: MXFP4/6/8, NVFP4, tile statistic, selection, and fake quant.
- `frozen_policy.json`: fixed non-target thresholds and derivation metadata.
- `derive_policy.py`: optional reference-model threshold derivation.
- `test_adaptquant.py`: fast CPU format/padding/accounting checks.
- `evaluate.py`, `run_eval.py`: course-provided evaluation code.
- `results/adaptquant_result.json`: verified submitted run.
- `314551126_report.pdf`: Part 3 report.
