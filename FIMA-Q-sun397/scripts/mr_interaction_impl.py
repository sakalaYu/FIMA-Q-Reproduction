"""Residual-budget MR with detached, harmful operator-interaction weights.

ReLU MR/plain-clipped supervision: APHQ-ViT / GLF-Q. Fake quantization and
STE are established tools. The positive four-path excess-loss weighting is
an experimental mechanism, not a claim of established novelty or accuracy.
"""

import logging

import torch
from torch.nn import functional as F

from quantizers.uniform import UniformQuantizer
from utils import lsvit_mr as mr
from utils.lsvit_pair import fingerprint


@torch.no_grad()
def make_quantizer(x, per_row=False):
    """Small MSE grid search, affine 3-bit, per-output-row weights/global acts.

    Uses the repository's UniformQuantizer for both hard and STE forwards.
    Formal PTQ later performs its full calibration and LN reparameterization.
    """
    flat = x.detach() if per_row else x.detach().reshape(1, -1)
    low = flat.amin(-1, keepdim=True).clamp_max(0.)
    high = flat.amax(-1, keepdim=True).clamp_min(0.)
    best_error = torch.full_like(low, float('inf'))
    best_scale, best_zero = torch.ones_like(low), torch.zeros_like(low)
    for factor in (.7, .85, 1., 1.1):
        scale = ((high - low) * factor / 7).clamp_min(1e-8)
        zero = (-low * factor / scale).round().clamp(0, 7)
        sim = ((flat / scale).round() + zero).clamp(0, 7).sub(zero).mul(scale)
        error = (sim - flat).square().mean(-1, keepdim=True)
        update = error < best_error
        best_error = torch.minimum(error, best_error)
        best_scale = torch.where(update, scale, best_scale)
        best_zero = torch.where(update, zero, best_zero)
    q = UniformQuantizer(n_bits=3, symmetric=False, channel_wise=per_row)
    q.scale = best_scale if per_row else best_scale.reshape(())
    q.zero_point = best_zero if per_row else best_zero.reshape(())
    q.inited = True
    q.training_mode = True
    return q


@torch.no_grad()
def grids(student, inputs, device):
    # Only the fitting split is used; collect all activations for its first 32 images.
    x = inputs[:32].to(device)
    h = student.act(student.fc1(x))
    return (make_quantizer(x), make_quantizer(student.fc1.weight, True),
            make_quantizer(h), make_quantizer(student.fc2.weight, True))


def four_paths(student, x, quantizers):
    qa1, qw1, qa2, qw2 = quantizers
    h0 = student.act(student.fc1(x))
    h1 = student.act(F.linear(qa1(x), qw1(student.fc1.weight), student.fc1.bias))
    w2 = qw2(student.fc2.weight)
    return (F.linear(h0, student.fc2.weight, student.fc2.bias),
            F.linear(h1, student.fc2.weight, student.fc2.bias),
            F.linear(qa2(h0), w2, student.fc2.bias),
            F.linear(qa2(h1), w2, student.fc2.bias))


def token_errors(paths, target):
    return tuple((out - target).square().mean(-1) for out in paths)


def interaction_weights(errors):
    e00, e10, e01, e11 = (e.detach() for e in errors)
    delta = e11 - e10 - e01 + e00
    ratio = delta.clamp_min(0) / (e00 + e10 + e01 + e11).clamp_min(1e-12)
    return 1 + ratio, delta


def continue_fit(original_train, fit, inputs, targets, device, steps, clip, variant):
    warmup = steps // 2
    original_train(fit, inputs, targets, device, warmup, clip)
    student, optimizer, scheduler, generator = fit
    shared_sha = fingerprint((student.state_dict(), optimizer.state_dict(),
                              scheduler.state_dict(), generator.get_state()))
    if variant == 'baseline':
        original_train(fit, inputs, targets, device, steps - warmup, clip)
        return shared_sha
    quantizers = grids(student, inputs, device)
    remaining = steps - warmup
    logging.info('Interaction MR %s: %d quant-aware updates', variant, remaining)
    for count in range(remaining):
        if count and count % 250 == 0:
            quantizers = grids(student, inputs, device)
        ids = torch.randint(len(inputs), (4,), generator=generator)
        x, target = inputs[ids].to(device), targets[ids].to(device)
        optimizer.zero_grad(set_to_none=True)
        with mr.capture_fc2_input(student) as captured:
            out = student(x)
            h = captured.pop()
        clipped = F.linear(h.clamp_max(clip), student.fc2.weight, student.fc2.bias)
        base = F.mse_loss(out, target) + 2 * F.mse_loss(clipped, target)
        paths = four_paths(student, x, quantizers)
        errors = token_errors(paths, target)
        weights, delta = interaction_weights(errors)
        # Both arms compute identical paths; only the detached weights differ.
        effective = weights if variant == 'interaction' else torch.ones_like(weights)
        auxiliary = (effective * errors[-1]).mean()
        ramp = min((count + 1) / max(remaining // 2, 1), 1.)
        loss = base + .25 * ramp * auxiliary
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite interaction MR loss')
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (count + 1) % 250 == 0 or count + 1 == remaining:
            logging.info('Interaction MR %s %d/%d base=%.6g joint=%.6g harmful=%.3f mean_w=%.4f',
                         variant, count + 1, remaining, float(base.detach()),
                         float(errors[-1].mean().detach()), float((delta > 0).float().mean()),
                         float(weights.mean()))
    return shared_sha


@torch.no_grad()
def diagnose(student, inputs, targets, device, quantizers):
    sums = torch.zeros(7, dtype=torch.float64)
    count, energy = 0, 0.
    for x, target in zip(inputs.split(4), targets.split(4)):
        x, target = x.to(device), target.to(device)
        paths = four_paths(student, x, quantizers)
        errors = token_errors(paths, target)
        weights, delta = interaction_weights(errors)
        vals = [*(e.sum() for e in errors), delta.clamp_min(0).sum(),
                (delta > 0).sum(), weights.sum()]
        sums += torch.tensor([float(v) for v in vals], dtype=torch.float64)
        count += errors[0].numel()
        energy += float(target.square().mean(-1).sum())
    normalized = (sums[:5] / max(energy, 1e-12)).tolist()
    return dict(fp_nmse=normalized[0], fc1_w3a3_nmse=normalized[1],
                fc2_w3a3_nmse=normalized[2], joint_w3a3_nmse=normalized[3],
                positive_interaction_nmse=normalized[4],
                harmful_token_fraction=float(sums[5] / max(count, 1)),
                mean_interaction_weight=float(sums[6] / max(count, 1)))


def fit_residual(model, loader, device, seed, holdout, budget_policy, variant, original_fit):
    if budget_policy != 'residual' or len(model.blocks) != 12:
        raise ValueError('Interaction MR requires residual allocation over 12 ViT blocks')
    original_train = mr.train_fit
    calls, anchors = [], []

    def train(fit, inputs, targets, device, steps, clip):
        calls.append(steps)
        if len(calls) <= 12:
            if steps != 1000:
                raise RuntimeError('Unexpected MR pilot layout')
            return original_train(fit, inputs, targets, device, steps, clip)
        sha = continue_fit(original_train, fit, inputs, targets, device, steps, clip, variant)
        anchors.append(sha)

    # Intercept only the existing trainer; restore it even after interruption/error.
    mr.train_fit = train
    try:
        states, reports = original_fit(model, loader, device, seed, holdout, budget_policy)
    finally:
        mr.train_fit = original_train
    if len(calls) != 24 or len(anchors) != 12:
        raise RuntimeError('Unexpected residual MR call layout')
    import copy
    for index, (block, state, report) in enumerate(zip(model.blocks, states, reports)):
        inputs, targets = mr.teacher_features(model, block.mlp, loader, device)
        student = copy.deepcopy(block.mlp).to(device).eval()
        student.load_state_dict(state)
        student.act = torch.nn.ReLU()
        n = len(inputs) - holdout
        quantizers = grids(student, inputs[:n], device)
        report['interaction_diagnostics'] = diagnose(student, inputs[n:], targets[n:], device, quantizers)
        report['shared_start_sha256'] = anchors[index]
        logging.info('Interaction diagnostics blocks.%d: %s', index, report['interaction_diagnostics'])
        del student, inputs, targets, quantizers
    return states, reports
