"""Measure the soft/hard handoff and repair scales on the hard inference path.

No labels, weight updates, new Fisher queries, or inference branches are used.
The last images are held out from THIS repair, not from the earlier reconstruction.
"""

from contextlib import contextmanager
import hashlib
import math
import time
import logging

import torch
from torch import nn
from torch.nn.utils import parametrize


@contextmanager
def isolated_rng(device, seed):
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def activation_quantizers(block):
    found, seen = [], set()
    for module in block.modules():
        for attr in ('a_quantizer', 'A_quantizer', 'B_quantizer'):
            q = getattr(module, attr, None)
            if q is not None and id(q) not in seen and q.n_bits != 32:
                seen.add(id(q))
                found.append(q)
    return found


@contextmanager
def quantization_path(block, soft, probability):
    saved = []
    for module in block.modules():
        state = {key: getattr(module, key) for key in ('mode', 'training_mode', 'drop_prob', 'soft_targets')
                 if hasattr(module, key)}
        saved.append((module, state))
        if 'mode' in state:
            module.mode = 'quant_forward'
        if 'training_mode' in state:
            module.training_mode = True  # STE forward values equal inference when p=1.
        if 'drop_prob' in state:
            module.drop_prob = probability
        if 'soft_targets' in state:
            module.soft_targets = soft
    try:
        yield
    finally:
        for module, state in saved:
            for key, value in state.items():
                setattr(module, key, value)


def _batch(block, indices, device, probability):
    quant = block.quanted_input if block.quanted_input is not None else block.raw_input
    x = quant[indices].to(device)
    if probability < 1:
        raw = block.raw_input[indices].to(device)
        x = torch.where(torch.rand_like(x) < probability, x, raw)
    return x, block.raw_out[indices].to(device)


@torch.no_grad()
def path_mse(block, indices, device, batch_size, soft=False, probability=1.0, seed=0):
    total, count = 0.0, 0
    with isolated_rng(device, seed), quantization_path(block, soft, probability):
        for start in range(0, len(indices), batch_size):
            x, target = _batch(block, indices[start:start + batch_size], device, probability)
            error = block(x) - target
            total += float(error.square().sum())
            count += error.numel()
    value = total / max(count, 1)
    if not math.isfinite(value):
        raise RuntimeError('Non-finite hard-path diagnostic')
    return value


def path_diagnostics(block, indices, device, batch_size, seed, probability=0.5):
    result = {}
    for soft, label in ((True, 'soft'), (False, 'hard')):
        result[label + '_full_mse'] = path_mse(block, indices, device, batch_size, soft, 1.0, seed)
        # Paired masks for soft versus hard; two draws reduce mask noise.
        result[label + '_qdrop_mse'] = sum(path_mse(block, indices, device, batch_size, soft,
                                                  probability, seed + j) for j in range(2)) / 2
    result['hardening_gap_full'] = result['hard_full_mse'] - result['soft_full_mse']
    result['qdrop_gap_hard'] = result['hard_full_mse'] - result['hard_qdrop_mse']
    result['training_qdrop_probability'] = probability
    return result


class BoundedScale(nn.Module):
    def __init__(self, scale, bound):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros_like(scale))
        self.limit = math.log(bound)

    def forward(self, original):
        return original.detach() * self.delta.clamp(-self.limit, self.limit).exp()


def _fixed_digest(block, scale_ids):
    digest = hashlib.sha256()
    for name, p in block.named_parameters():
        if id(p) not in scale_ids:
            digest.update(name.encode())
            digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def repair_block_scales(block, name, device, options, drop_prob=0.5):
    started = time.perf_counter()
    count = len(block.raw_input)
    holdout = int(options['holdout'])
    if not 0 < holdout < count:
        raise ValueError('Repair holdout must be smaller than cached calibration size')
    train_count = count - holdout
    validation = list(range(train_count, count))
    seed = int(options['seed']) + 17001 + int(name.split('.')[-1])
    batch_size = int(options['batch_size'])
    diagnostics = path_diagnostics(block, validation, device, batch_size, seed, drop_prob)
    quantizers = activation_quantizers(block)
    if not quantizers:
        raise ValueError('No activation scales to repair')
    for q in quantizers:
        if not bool(torch.isfinite(q.scale).all() and (q.scale > 0).all()):
            raise ValueError('Repair requires finite positive activation scales')
    initial = [q.scale.detach().clone() for q in quantizers]
    flags = [(p, p.requires_grad) for p in block.parameters()]
    fixed_before = _fixed_digest(block, {id(q.scale) for q in quantizers})
    baseline = best_score = diagnostics['hard_full_mse']
    best = [torch.zeros_like(scale) for scale in initial]
    selected_step, active_steps = 0, 0
    last_loss = None
    attached = []
    try:
        for p, _ in flags:
            p.requires_grad_(False)
        for q in quantizers:
            parametrize.register_parametrization(q, 'scale', BoundedScale(q.scale, options['bound']))
            attached.append(q)
        deltas = [q.parametrizations.scale[0].delta for q in quantizers]
        optimizer = torch.optim.Adam(deltas, lr=options['lr'])
        generator = torch.Generator().manual_seed(seed)
        normalizer = max(baseline, 1e-12)
        with isolated_rng(device, seed), quantization_path(block, False, 1.0):
            for step in range(int(options['steps'])):
                indices = torch.randperm(train_count, generator=generator)[:batch_size].tolist()
                x, target = _batch(block, indices, device, 1.0)
                optimizer.zero_grad(set_to_none=True)
                mse = (block(x) - target).square().mean()
                loss = mse / normalizer
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Non-finite scale repair loss')
                loss.backward()
                if not all(d.grad is not None and bool(torch.isfinite(d.grad).all()) for d in deltas):
                    raise RuntimeError('Missing/non-finite scale repair gradients')
                optimizer.step()
                with torch.no_grad():
                    for q, delta in zip(quantizers, deltas):
                        limit = q.parametrizations.scale[0].limit
                        delta.clamp_(-limit, limit)
                active_steps += 1
                last_loss = float(mse.detach())
                if (step + 1) % options['eval_every'] == 0 or step + 1 == options['steps']:
                    score = path_mse(block, validation, device, batch_size)
                    logging.info('Hard-scale repair %s step %d/%d: holdout MSE %.8g (baseline %.8g)',
                                 name, step + 1, options['steps'], score, baseline)
                    if score < best_score - max(1e-12, abs(baseline) * 1e-8):
                        best_score = score
                        best = [d.detach().clone() for d in deltas]
                        selected_step = step + 1
            with torch.no_grad():
                for d, value in zip(deltas, best):
                    d.copy_(value)
    except BaseException:
        for q in attached:
            parametrize.remove_parametrizations(q, 'scale', leave_parametrized=False)
        with torch.no_grad():
            for q, value in zip(quantizers, initial):
                q.scale.copy_(value)
        raise
    else:
        for q in attached:
            parametrize.remove_parametrizations(q, 'scale', leave_parametrized=True)
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
    fixed_after = _fixed_digest(block, {id(q.scale) for q in quantizers})
    if fixed_before != fixed_after:
        raise RuntimeError('Scale-only repair changed a fixed parameter')
    final_score = path_mse(block, validation, device, batch_size)
    if final_score > baseline + max(1e-10, abs(baseline) * 1e-5):
        raise RuntimeError('Selected repair disagrees with the saved inference scales')
    ratios = torch.cat([(q.scale.detach() / value).flatten().cpu() for q, value in zip(quantizers, initial)])
    return dict(block=name, mode='hard_scale', train_samples=train_count,
                repair_holdout_samples=holdout, holdout_used_in_main_reconstruction=True,
                steps=options['steps'], active_steps=active_steps, selected_step=selected_step,
                accepted=selected_step > 0, diagnostics=diagnostics,
                hard_mse_before=baseline, hard_mse_after=final_score,
                last_training_mse=last_loss, ratio_min=float(ratios.min()), ratio_max=float(ratios.max()),
                fixed_parameters_unchanged=True, elapsed_seconds=time.perf_counter() - started)


def continue_soft_block(block, name, device, options, loss_func, w_optimizer, a_optimizer,
                        initial_lr, drop_prob, end_b):
    """Extra-step control: same sampled images/batch size as hard repair.

    Retains the main DPLR normalizers, Fisher estimate, Adam states and QDrop.
    Matching update counts does not imply matching FLOPs or wall-clock time.
    """
    started = time.perf_counter()
    train_count = len(block.raw_input) - options['holdout']
    validation = list(range(train_count, len(block.raw_input)))
    seed = int(options['seed']) + 17001 + int(name.split('.')[-1])
    before = path_diagnostics(block, validation, device, options['batch_size'], seed, drop_prob)
    generator = torch.Generator().manual_seed(seed)
    for group in a_optimizer.param_groups:
        group['lr'] = initial_lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_optimizer, options['steps'], eta_min=0.)
    old_decay = loss_func.temp_decay
    loss_func.temp_decay = lambda count: end_b
    try:
        with isolated_rng(device, seed), quantization_path(block, True, drop_prob):
            for step in range(options['steps']):
                indices = torch.randperm(train_count, generator=generator)[:options['batch_size']].tolist()
                x, target = _batch(block, indices, device, drop_prob)
                w_optimizer.zero_grad(set_to_none=True)
                a_optimizer.zero_grad(set_to_none=True)
                loss_func.update_fisher = False
                gradient = block.raw_grad.to(device) if loss_func.rec_loss == 'fisher_dplr' else None
                loss = loss_func(block(x), target, gradient)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Non-finite soft continuation loss')
                loss.backward()
                w_optimizer.step()
                a_optimizer.step()
                scheduler.step()
    finally:
        loss_func.temp_decay = old_decay
    after = path_diagnostics(block, validation, device, options['batch_size'], seed, drop_prob)
    return dict(block=name, mode='soft_extra', extra_steps=options['steps'],
                train_samples=train_count, batch_size=options['batch_size'],
                diagnostics_before=before, diagnostics_after=after,
                fisher_refreshes=0, elapsed_seconds=time.perf_counter() - started)
