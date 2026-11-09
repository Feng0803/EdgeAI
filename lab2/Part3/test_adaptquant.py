"""Fast CPU checks for the frozen AdaptQuant implementation."""

import json
import math
from pathlib import Path

import torch

from mx_formats import (adaptquant_operand, effective_bits, mx_fake_quant,
                        tile_statistic)


HERE = Path(__file__).resolve().parent


def main():
    fp4_source = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 4])
    assert torch.equal(mx_fake_quant(fp4_source, "mxfp4"), fp4_source)

    source = torch.zeros(16, 32)
    source[0, 0] = 1.0
    statistic = tile_statistic(source)
    assert statistic.shape == (1, 1)
    assert abs(statistic.item() - 0.5 * math.log2(512)) < 1e-6

    torch.manual_seed(0)
    source = torch.randn(17, 64, dtype=torch.float16)
    reconstructed, counts = adaptquant_operand(source, 1.9, math.inf, chunk_rows=16)
    assert reconstructed.shape == source.shape
    assert counts.sum().item() == 4  # two padded 16x32 tile rows by two K tiles
    assert torch.isfinite(reconstructed).all()

    bits, shares = effective_bits(torch.tensor([1, 1, 0]))
    # Scale overhead is included in 4.25/6.25; five-byte tile metadata is not.
    expected = (4.25 + 6.25) / 2
    assert abs(bits - expected) < 1e-12
    assert shares == {"mxfp4": 0.5, "mxfp6": 0.5, "mxfp8": 0.0}

    policy = json.loads((HERE / "frozen_policy.json").read_text())
    assert policy["target_model_data_used"] is False
    assert set(policy["weight"]) == set(policy["activation"])
    assert set(policy["weight"]) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }
    print("AdaptQuant implementation checks passed.")


if __name__ == "__main__":
    main()
