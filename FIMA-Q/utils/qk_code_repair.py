"""Experimental final-hard-path Q/K code search; not a claimed new algorithm.

Uses existing quantization grids and logits KD. No LS estimator is introduced
here. Selection uses calibration images only; zero-change is always retained.
"""
import logging

import torch
from torch.nn import functional as F

from utils.hard_scale_repair import isolated_rng, quantization_path
from utils.activation_codebook import teacher_kl


def propose_codes(weight, gradient, scale, zero, width, budget):
    """One-step integer moves, restricted to Q/K rows in fused QKV."""
    scale = scale.reshape(-1, 1).expand_as(weight)
    zero = zero.reshape(-1, 1).expand_as(weight).round()
    if weight.shape[0] != 3 * width or weight.shape[1] != width:
        raise ValueError('Expected fused QKV with shape (3C, C)')
    if not bool(torch.isfinite(scale).all() and (scale != 0).all()):
        raise ValueError('Weight scales must be finite and nonzero')
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError('Non-finite Q/K gradient')
    codes = ((weight / scale).round() + zero).clamp(0, 7)
    direction = -(gradient * scale).sign()
    proposed = codes + direction
    score = (gradient * scale).abs()
    eligible = (proposed >= 0) & (proposed <= 7) & (direction != 0)
    eligible[2 * width:] = False
    score = torch.where(eligible, score, torch.zeros_like(score))
    count = min(budget, int((score > 0).sum()))
    result = weight.detach().clone()
    if count:
        indices = score.flatten().topk(count).indices
        values = ((proposed - zero) * scale).flatten()
        result.flatten()[indices] = values[indices]
    return result, count


def fit_kl(model, images, logits, device, gradient=False):
    value = 0.0
    for x, target in zip(images.split(4), logits.split(4)):
        loss = F.kl_div((model(x.to(device)) / 2).log_softmax(-1),
                        (target.to(device) / 2).softmax(-1), reduction='batchmean') * 4
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Non-finite code-search KD')
        factor = len(x) / len(images)
        if gradient:
            (loss * factor).backward()
        value += float(loss.detach()) * factor
    return value


def refine_qk_codes(model, images, logits, gate, targets, device):
    """One pass, <=128 changed entries/block; candidates preselected on fit KL."""
    flags = [(p, p.requires_grad) for p in model.parameters()]
    baseline = score = teacher_kl(model, gate, targets, device)
    reports = []
    # 64 fit images, disjoint from the last 128 gate images. These images may
    # have been used by earlier C/LS reconstruction; no independent-test claim.
    fit_x, fit_y = images[:64], logits[:64]
    try:
        model.requires_grad_(False)
        with isolated_rng(device, 193407), quantization_path(model, False, 1.0):
            for index, block in enumerate(model.blocks):
                layer = block.attn.qkv
                q = layer.w_quantizer
                if q.n_bits != 3 or q.sym:
                    raise ValueError('Q/K code search requires asymmetric 3-bit weights')
                original = layer.weight.detach().clone()
                layer.weight.requires_grad_(True)
                model.zero_grad(set_to_none=True)
                try:
                    before_fit = fit_kl(model, fit_x, fit_y, device, gradient=True)
                    if layer.weight.grad is None:
                        raise RuntimeError('QKV received no gradient')
                    gradient = layer.weight.grad.detach().clone()
                finally:
                    layer.weight.requires_grad_(False)
                    model.zero_grad(set_to_none=True)
                best_fit, best, selected = before_fit, original, 0
                trials = []
                committed = False
                before_gate = score
                trial_gate = None
                try:
                    for budget in (8, 32, 128):
                        candidate, count = propose_codes(original, gradient, q.scale.detach(),
                                                         q.zero_point.detach(), layer.in_features, budget)
                        with torch.no_grad():
                            layer.weight.copy_(candidate)
                            candidate_fit = fit_kl(model, fit_x, fit_y, device)
                        trials.append(dict(budget=budget, count=count, fit_kl=candidate_fit))
                        if candidate_fit < best_fit - 1e-8:
                            best_fit, best, selected = candidate_fit, candidate, count
                    with torch.no_grad():
                        layer.weight.copy_(best)
                        if selected:
                            trial_gate = teacher_kl(model, gate, targets, device)
                            committed = trial_gate < score - max(1e-7, score * 0.001)
                    if committed:
                        score = trial_gate
                finally:
                    if not committed:
                        with torch.no_grad():
                            layer.weight.copy_(original)
                report = dict(block=index, accepted=committed, changed_entries=selected if committed else 0,
                              fit_kl_before=before_fit, fit_kl_candidate=best_fit,
                              gate_kl_before=before_gate, gate_kl_trial=trial_gate, trials=trials)
                reports.append(report)
                logging.info('Q/K integer-code repair %d/%d: %s', index + 1, len(model.blocks), report)
    finally:
        model.zero_grad(set_to_none=True)
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
    return dict(method='qk_integer_code_repair_v1', blocks=reports, fit_images=64,
                gate_kl_before=baseline, gate_kl_after=score,
                changed_entries=sum(r['changed_entries'] for r in reports))
