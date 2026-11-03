"""Fit one eight-value activation codebook per selected ViT layer."""

import logging
import math

import torch
from torch import nn
from torch.nn import functional as F


class ActivationCodebook(nn.Module):
    n_bits = 3

    def __init__(self, levels):
        super().__init__()
        self.register_buffer('codebook', levels.detach().clone())

    def forward(self, x):
        levels = self.codebook.to(x)
        boundaries = (levels[:-1] + levels[1:]) * 0.5
        return levels[torch.bucketize(x.contiguous(), boundaries)]


def attach_saved_codebooks(model, state):
    for key, levels in state.items():
        if key.endswith('.codebook'):
            parent, attr, _ = key.rsplit('.', 2)
            setattr(model.get_submodule(parent), attr, ActivationCodebook(levels))


class _Captured(Exception):
    pass


def capture_inputs(model, layer, batches, device):
    inputs = []

    def capture(module, args):
        inputs.append(tuple(x.detach().cpu() for x in args))
        raise _Captured()

    hook = layer.register_forward_pre_hook(capture)
    try:
        for images in batches:
            try:
                model(images.to(device))
            except _Captured:
                continue
            raise RuntimeError('Expected activation layer was not reached')
    finally:
        hook.remove()
    return inputs


def candidate_levels(inputs, site):
    # Bound the percentile sample size independently of the number of tokens.
    values = torch.cat([x.flatten()[::max(1, x.numel() // 4096)] for x, *_ in inputs])
    positive = values[values > 0]
    if not positive.numel():
        return []
    caps = torch.quantile(positive.float(), torch.tensor([0.99, 0.999, 0.9999])).unique()
    levels = []
    if site == 'softmax':
        for cap in caps:
            for spacing in (0.5, 1.0, 2.0):
                levels.append(torch.cat((torch.zeros(1), cap * torch.exp2(-spacing * torch.arange(6, -1, -1)))))
    else:
        negative = -values[values < 0]
        if not negative.numel():
            return []
        neg_cap = torch.quantile(negative.float(), 0.999)
        for cap in caps:
            for n_negative in (1, 2, 3):
                neg = torch.linspace(-1, 0, n_negative + 1)[:-1] * neg_cap
                for exponent in (1, 2):
                    pos = torch.linspace(0, 1, 8 - n_negative).pow(exponent) * cap
                    levels.append(torch.cat((neg, pos)))
    return levels


def local_fit(layer, attr, inputs, site, device):
    baseline = getattr(layer, attr)
    candidates = [baseline] + [ActivationCodebook(x.to(device)) for x in candidate_levels(inputs, site)]
    errors = torch.zeros(len(candidates), device=device)
    weight = layer.quant_weight_bias()[0] if site == 'gelu' else None
    elements = 0
    for x, *other in inputs:
        x = x.to(device)
        if site == 'softmax':
            value = layer.quant_input_B(other[0].to(device))
            target = x @ value
        else:
            target = F.linear(x, weight)
        elements += target.numel()
        for index, candidate in enumerate(candidates):
            prediction = candidate(x) @ value if site == 'softmax' else F.linear(candidate(x), weight)
            errors[index] += (prediction - target).square().sum()
    errors = errors.cpu() / elements
    if not bool(torch.isfinite(errors).all()):
        raise RuntimeError('Non-finite codebook calibration error')
    index = int(errors.argmin())
    return candidates[index], float(errors[0]), float(errors[index])


def teacher_kl(model, batches, targets, device):
    total = 0.0
    for images, target in zip(batches, targets):
        logits = model(images.to(device)).float()
        total += float(F.kl_div(logits.log_softmax(-1), target, reduction='sum'))
    score = total / sum(len(images) for images in batches)
    if not math.isfinite(score):
        raise RuntimeError('Non-finite teacher KL during activation codebook selection')
    return score


@torch.no_grad()
def fit_activation_codebooks(model, teacher, loader, site):
    device = next(model.parameters()).device
    images = torch.cat([batch for batch, _ in loader])
    if len(images) != 128:
        raise ValueError('Codebook fitting requires 64 fit and 64 gate images')
    fit = images[:64].split(8)
    gate = images[64:].split(8)
    teacher.to(device)
    targets = [teacher(batch.to(device)).float().softmax(-1) for batch in gate]
    teacher.cpu()
    score = initial_kl = teacher_kl(model, gate, targets, device)
    reports = []
    for index, block in enumerate(model.blocks):
        if site == 'softmax':
            layer, attr = block.attn.matmul2, 'A_quantizer'
        else:
            layer, attr = block.mlp.fc2, 'a_quantizer'
        baseline = getattr(layer, attr)
        inputs = capture_inputs(model, layer, fit, device)
        candidate, mse_before, mse_after = local_fit(layer, attr, inputs, site, device)
        del inputs
        trial = None
        accepted = False
        before = score
        if candidate is not baseline:
            setattr(layer, attr, candidate)
            try:
                trial = teacher_kl(model, gate, targets, device)
                accepted = trial < score - max(1e-7, 0.001 * score)
            finally:
                if not accepted:
                    setattr(layer, attr, baseline)
            if accepted:
                score = trial
        report = dict(block=index, accepted=accepted, local_mse_before=mse_before, local_mse_best=mse_after,
                      gate_kl_before=before, gate_kl_trial=trial, gate_kl_after=score,
                      levels=candidate.codebook.tolist() if candidate is not baseline else None)
        reports.append(report)
        logging.info('Activation codebook %s %s', site, report)
    return dict(site=site, bits=3, fit_images=64, gate_images=64, gate_kl_before=initial_kl,
                gate_kl_after=score, blocks=reports)
