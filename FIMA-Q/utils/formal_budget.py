"""Fixed-sum DPLR reconstruction schedules and hard-path probes."""

import hashlib
import math
from pathlib import Path
import random


BLOCKS = tuple(f'blocks.{i}' for i in range(12))
NOMINAL_STEPS = 10000
PILOT_STEPS = 1000
PROBE_COUNT = 128
TIERS = (7000,) * 4 + (10000,) * 4 + (13000,) * 4
VARIANTS = ('uniform', 'residual', 'shuffled')


def make_schedule(variant, scores=None, seed=3408):
    if variant == 'uniform':
        return dict.fromkeys(BLOCKS, NOMINAL_STEPS)
    if variant not in VARIANTS or not isinstance(scores, dict) or set(scores) != set(BLOCKS):
        raise ValueError('A nonuniform plan requires all 12 uniform-run pilot scores')
    if any(not math.isfinite(v) or v < 0 for v in scores.values()):
        raise ValueError('Pilot scores must be finite and nonnegative')
    order = sorted(BLOCKS, key=lambda name: (scores[name], BLOCKS.index(name)))
    assigned = dict(zip(order, TIERS))
    budgets = [assigned[name] for name in BLOCKS]
    if variant == 'shuffled':
        rng = random.Random(91173 + seed)
        original = budgets.copy()
        while budgets == original:
            rng.shuffle(budgets)
    return dict(zip(BLOCKS, budgets))


def validate_plan(plan, *, seed, steps, k):
    if plan['seed'] != seed or steps != NOMINAL_STEPS or k != 5:
        raise ValueError('Formal budget runs require the matching seed, nominal 10,000 steps and DPLR k=5')
    if plan['budgets'] != make_schedule(plan['variant'], plan.get('uniform_pilot_nmse'), seed):
        raise ValueError('Formal budget does not match the specified assignment rule')


def fresh_probe_batches(generator, data_root, seed, original_hashes):
    import numpy as np
    import torch

    dataset = generator.train_set
    permutation = np.random.RandomState(seed).permutation(len(dataset))
    for count, index_hash, _ in original_hashes:
        indices = permutation[:int(count)].astype('<i8', copy=False)
        if hashlib.sha256(indices.tobytes()).hexdigest() != index_hash:
            raise ValueError('Cannot verify the original training-image selection')
    last = max(int(row[0]) for row in original_hashes)
    # Reserve the preceding 512 positions for endpoint diagnosis and scale screening.
    indices = permutation[last + 512:last + 512 + PROBE_COUNT].tolist()
    if len(indices) != PROBE_COUNT:
        raise ValueError('Not enough disjoint ImageNet train images for the hard-path probe')
    images, names = [], []
    digest = hashlib.sha256()
    for i in indices:
        path = dataset.samples[i][0]
        tensor = generator.val_transform(dataset.loader(path)).contiguous()
        name = Path(path).relative_to(data_root).as_posix()
        digest.update(name.encode() + b'\0')
        digest.update(tensor.numpy().tobytes())
        images.append(tensor)
        names.append(name)
    batches = [torch.stack(images[i:i + 8]) for i in range(0, len(images), 8)]
    return batches, dict(indices=indices, paths=names, tensor_sha256=digest.hexdigest())


def hard_probe_nmse(reconstructor, block, images, device):
    import torch
    from utils.hard_scale_repair import quantization_path

    modes = [(module, module.mode) for module in reconstructor.model.modules() if hasattr(module, 'mode')]
    prefix = [reconstructor.blocks[name] for name in reconstructor.quanted_blocks]
    captured = None

    def capture(_, args, output):
        nonlocal captured
        captured = output.detach()

    handle = block.register_forward_hook(capture)
    try:
        with torch.no_grad():
            for module, _ in modes:
                module.mode = 'raw'
            references = []
            for batch in images:
                reconstructor.model(batch.to(device))
                references.append(captured.cpu())
            for previous in prefix:
                reconstructor.set_block_mode(previous, 'quant_forward')
            err, energy = 0., 0.
            with quantization_path(block, soft=False, probability=1., training=False):
                for batch, reference in zip(images, references):
                    reconstructor.model(batch.to(device))
                    reference = reference.to(device)
                    err += float((captured - reference).square().sum())
                    energy += float(reference.square().sum())
    finally:
        handle.remove()
        for module, mode in modes:
            module.mode = mode
    value = err / max(energy, 1e-12)
    if not math.isfinite(value):
        raise RuntimeError('Non-finite formal reconstruction probe')
    return value
