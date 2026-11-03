import os
from pathlib import Path
import random
import sys
import torch
from datasets import Dataset, load_dataset


def log(message):
    print(message,file=sys.stderr,flush=True)


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


