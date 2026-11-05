"""Independent C-MSE adaptations of KD, AdaLog and ERQ.

References and deviations are specified in METHOD_ADAPTATIONS_RUN.md.
All fits use unlabeled training images. Validation labels are used only by the
caller to report the paired final accuracies.
"""

import copy
import logging
import math
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from utils.activation_codebook import capture_inputs, teacher_kl
from utils.hard_scale_repair import BoundedScale, activation_quantizers, isolated_rng, quantization_path


class HeadAdaLog(nn.Module):
    """r=37 AdaLog with per-head q, token scales, and code 7 reserved for zero."""
    n_bits = 3
    sym = False
    n_levels = 4

    def __init__(self, scale, q):
        super().__init__()
        self.scale = nn.Parameter(scale.detach().clone(), requires_grad=False)
        self.register_buffer('q', q.detach().clone())
        self.inited = True
        self.training_mode = False
        self.drop_prob = 1.0

    def forward(self, x):
        index = torch.round(-torch.log2((x / self.scale).clamp(1e-30, 1.0)) * 37 / self.q)
        exponent = index.clamp(0, 6) * self.q / 37
        integer = exponent.floor()
        fraction = (torch.exp2(-(exponent - integer)) * 14).round() / 14
        value = torch.exp2(-integer) * fraction * self.scale
        return torch.where((index < 7) & (x > 0), value, torch.zeros_like(value))


def attach_method_quantizers(model, state):
    for key, value in state.items():
        if key.endswith('.adalog_q'):
            name = key[:-len('.adalog_q')]
            parent, attr = name.rsplit('.', 1)
            quantizer = HeadAdaLog(state[name + '.scale'], value)
            # On disk the distinctive buffer name identifies this quantizer.
            quantizer.register_buffer('adalog_q', quantizer.q)
            setattr(model.get_submodule(parent), attr, quantizer)


def prepare_data(model, teacher, loader):
    device = next(model.parameters()).device
    images = torch.cat([x for x, _ in loader])
    if len(images) != 1024:
        raise ValueError('Method adaptations require 1024 cached calibration images')
    fit = images[:896].split(4)
    gate = images[896:].split(8)
    teacher.to(device)
    with torch.no_grad():
        logits = [teacher(x.to(device)).detach().cpu() for x in images.split(8)]
    teacher.cpu()
    logits = torch.cat(logits)
    targets = [x.to(device).softmax(-1) for x in logits[896:].split(8)]
    return images, logits, fit, gate, targets


def kd_refine(model, images, logits, gate, targets, steps, device):
    """End-to-end logits KD on a hard-quantized network, learning scales only."""
    quantizers = activation_quantizers(model)
    if any(not bool(torch.isfinite(q.scale).all() and (q.scale != 0).all()) for q in quantizers):
        raise ValueError('KD requires finite nonzero scales')
    flags = [(p, p.requires_grad) for p in model.parameters()]
    baseline = best_score = teacher_kl(model, gate, targets, device)
    initial = [q.scale.detach().clone() for q in quantizers]
    attached = []
    best = [torch.zeros_like(s) for s in initial]
    selected = 0
    try:
        model.requires_grad_(False)
        for q in quantizers:
            parametrize.register_parametrization(q, 'scale', BoundedScale(q.scale, 1.1))
            attached.append(q)
        deltas = [q.parametrizations.scale[0].delta for q in quantizers]
        optimizer = torch.optim.Adam(deltas, lr=0.0003)
        generator = torch.Generator().manual_seed(93407)
        temperature = 2.0
        with isolated_rng(device, 93407), quantization_path(model, False, 1.0):
            for step in range(steps):
                indices = torch.randperm(896, generator=generator)[:4]
                target = (logits[indices].to(device) / temperature).softmax(-1)
                prediction = model(images[indices].to(device))
                kd = F.kl_div((prediction / temperature).log_softmax(-1), target,
                              reduction='batchmean') * temperature ** 2
                regularizer = sum(d.square().mean() for d in deltas) / len(deltas)
                loss = kd + 0.001 * regularizer
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Non-finite end-to-end KD loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(d.grad is None or not bool(torch.isfinite(d.grad).all()) for d in deltas):
                    raise RuntimeError('Missing or non-finite KD scale gradients')
                optimizer.step()
                with torch.no_grad():
                    for delta in deltas:
                        delta.clamp_(-math.log(1.1), math.log(1.1))
                if (step + 1) % 100 == 0 or step + 1 == steps:
                    with torch.no_grad():
                        score = teacher_kl(model, gate, targets, device)
                    logging.info('End-to-end KD step %d/%d loss %.6g gate KL %.6g', step + 1, steps, float(kd), score)
                    if score < best_score:
                        best_score, selected = score, step + 1
                        best = [d.detach().clone() for d in deltas]
            with torch.no_grad():
                for delta, value in zip(deltas, best):
                    delta.copy_(value)
    except BaseException:
        for q in attached:
            parametrize.remove_parametrizations(q, 'scale', leave_parametrized=False)
        raise
    else:
        for q in attached:
            parametrize.remove_parametrizations(q, 'scale', leave_parametrized=True)
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
    return dict(method='end_to_end_hard_scale_KD', steps=steps, temperature=2, selected_step=selected,
                scale_bound=1.1, lr=0.0003, gate_kl_before=baseline, gate_kl_after=best_score)


@torch.no_grad()
def adalog_refine(model, fit, gate, targets, device):
    score = initial_score = teacher_kl(model, gate, targets, device)
    reports = []
    # A fixed 64-image fit budget makes q search practical on one shared GPU.
    calibration = fit[:16]
    for index, block in enumerate(model.blocks):
        layer = block.attn.matmul2
        original = layer.A_quantizer
        inputs = capture_inputs(model, layer, calibration, device)
        heads = layer.num_heads
        # Preserve the original per-query-token granularity. Use observed
        # maxima as the positive log-grid scale, then search q per head.
        scale = None
        for x, _ in inputs:
            maxima = x.amax(dim=(0, 3), keepdim=True)
            scale = maxima if scale is None else torch.maximum(scale, maxima)
        scale = scale.clamp_min(1e-12).to(device)
        qs = list(range(10, 139, 8))
        errors = torch.zeros(len(qs), heads, device=device)
        baseline_error = torch.zeros(heads, device=device)
        for x, value in inputs:
            x, value = x.to(device), layer.quant_input_B(value.to(device))
            target = x @ value
            baseline_error += ((original(x) @ value - target).square()).sum(dim=(0, 2, 3))
            for j, q in enumerate(qs):
                quantizer = HeadAdaLog(scale, torch.full((1, heads, 1, 1), q, device=device))
                errors[j] += ((quantizer(x) @ value - target).square()).sum(dim=(0, 2, 3))
        choices = errors.argmin(0)
        chosen_q = torch.tensor(qs, device=device)[choices].reshape(1, heads, 1, 1)
        candidate = HeadAdaLog(scale, chosen_q)
        candidate.register_buffer('adalog_q', candidate.q)
        layer.A_quantizer = candidate
        accepted = False
        before = score
        try:
            trial = teacher_kl(model, gate, targets, device)
            accepted = trial < score - max(1e-7, 0.001 * score)
        finally:
            if not accepted:
                layer.A_quantizer = original
        if accepted:
            score = trial
        report = dict(block=index, accepted=accepted, q_per_head=chosen_q.flatten().tolist(),
                      scale_shape=list(scale.shape), gate_kl_before=before, gate_kl_trial=trial,
                      local_sse_before=float(baseline_error.sum()), local_sse_after=float(errors.min(0).values.sum()))
        reports.append(report)
        logging.info('AdaLog adaptation %s', report)
    return dict(method='adalog_head_token_zero_reserved', blocks=reports,
                gate_kl_before=initial_score, gate_kl_after=score)


def erq_quantize(weight, gram, scale, zero, iterations=8):
    """Progressive half partition, rounding flips and ridge correction.

    Uses the existing 3-bit row grid, so the resulting weights remain exactly
    representable by the current inference quantizer.
    """
    work = weight.double().cpu().clone()
    gram = gram.double().cpu()
    scale, zero = scale.double().cpu(), zero.double().cpu().round()
    output = torch.empty_like(work)
    start, width = 0, work.shape[1]
    ridge = max(float(gram.diagonal().mean()) * 0.01, 1e-12)
    while start < width:
        end = start + max(1, (width - start) // 2)
        raw = work[:, start:end]
        lower = ((raw / scale).floor() + zero).clamp(0, 7) - zero
        upper = ((raw / scale).ceil() + zero).clamp(0, 7) - zero
        quant = ((raw / scale).round() + zero).clamp(0, 7) - zero
        quant = quant * scale
        hessian = gram[start:end, start:end]
        error = quant - raw
        for _ in range(iterations):
            alternate = torch.where((quant - lower * scale).abs() < 1e-9, upper * scale, lower * scale)
            change = alternate - quant
            gains = 2 * change * (error @ hessian) + change.square() * hessian.diagonal()
            best_gain, positions = gains.min(1)
            rows = torch.where(best_gain < -1e-12)[0]
            if not len(rows):
                break
            columns = positions[rows]
            quant[rows, columns] = alternate[rows, columns]
            error = quant - raw
        output[:, start:end] = quant
        if end < width:
            rr = gram[end:, end:] + ridge * torch.eye(width - end, dtype=gram.dtype)
            cross = error @ gram[start:end, end:]
            work[:, end:] -= torch.linalg.solve(rr, cross.T).T
        start = end
    return output.to(weight)


@torch.no_grad()
def erq_refine(model, teacher, fit, gate, targets, device):
    score = initial_score = teacher_kl(model, gate, targets, device)
    reports = []
    original_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        for index, block in enumerate(model.blocks):
            for suffix in ('attn.proj', 'mlp.fc2'):
                name = f'blocks.{index}.{suffix}'
                layer = model.get_submodule(name)
                # These two projections are outside Q/K's learned transforms.
                # The FP teacher has the same fixed Hadamard basis as C-MSE.
                reference = teacher.get_submodule(name).weight.detach().double().cpu()
                width = layer.in_features
                xx = torch.zeros(width, width, dtype=torch.float64)
                dx = torch.zeros_like(xx)
                count = 0
                inputs = capture_inputs(model, layer, fit[:16], device)
                for (x,) in inputs:
                    xq = layer.quant_input(x.to(device)).cpu().reshape(-1, width).double()
                    x = x.reshape(-1, width).double()
                    xx += xq.T @ xq
                    dx += (xq - x).T @ xq
                    count += len(x)
                del inputs
                xx /= count
                dx /= count
                ridge = max(float(xx.diagonal().mean()) * 0.01, 1e-12)
                corrected = reference - torch.linalg.solve(xx + ridge * torch.eye(width, dtype=xx.dtype),
                                                           (reference @ dx).T).T
                scale = layer.w_quantizer.scale.detach().reshape(layer.out_features, 1)
                zero = layer.w_quantizer.zero_point.detach().reshape(layer.out_features, 1)
                candidate = erq_quantize(corrected.to(layer.weight), xx, scale, zero)
                original = layer.weight.detach().clone()
                layer.weight.copy_(candidate)
                accepted = False
                before = score
                try:
                    trial = teacher_kl(model, gate, targets, device)
                    accepted = trial < score - max(1e-7, 0.001 * score)
                finally:
                    if not accepted:
                        layer.weight.copy_(original)
                if accepted:
                    score = trial
                report = dict(layer=name, accepted=accepted, gate_kl_before=before, gate_kl_trial=trial)
                reports.append(report)
                logging.info('ERQ adaptation %s', report)
    finally:
        torch.set_num_threads(original_threads)
    return dict(method='erq_projection_grid_preserving', layers=reports,
                gate_kl_before=initial_score, gate_kl_after=score)


def adapt_model(model, teacher, loader, method, steps):
    started = time.perf_counter()
    images, logits, fit, gate, targets = prepare_data(model, teacher, loader)
    device = next(model.parameters()).device
    if method == 'kd':
        result = kd_refine(model, images, logits, gate, targets, steps, device)
    elif method in ('qkcode', 'qkcode_kd'):
        from utils.qk_code_repair import refine_qk_codes
        result = refine_qk_codes(model, images, logits, gate, targets, device)
        if method == 'qkcode_kd':
            result = dict(method='qk_integer_code_then_scale_KD_v1', codes=result,
                          kd=kd_refine(model, images, logits, gate, targets, steps, device))
    elif method == 'adalog':
        result = adalog_refine(model, fit, gate, targets, device)
    elif method == 'erq':
        result = erq_refine(model, teacher, fit, gate, targets, device)
    else:
        raise ValueError(method)
    result.update(fit_images=896 if method in ('kd', 'qkcode_kd') else 64, gate_images=128,
                  elapsed_seconds=time.perf_counter() - started)
    return result
