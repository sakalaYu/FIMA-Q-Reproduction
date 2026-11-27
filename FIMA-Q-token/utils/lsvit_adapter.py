"""Streaming, damped LS-ViT curvature statistics.

Equations follow hhh7748/LS-ViT utils/block_recon.py cal_hessian.
Relative denominator floors and streaming accumulation are integration changes.
"""
from contextlib import ExitStack

import torch
import torch.nn.functional as F

from utils.hard_scale_repair import isolated_rng, quantization_path


MIX_CANDIDATES = (0.25, 0.375, 0.5, 0.625, 0.75)


def ls_statistics(gradient, delta):
    g, d = gradient.abs().flatten(1).double(), delta.abs().flatten(1).double()
    product = (g * d).sum(1)
    return (g * d).sum(0), d.square().sum(0), (g * product.sqrt().unsqueeze(1)).sum(0), product.sum()


def finish_statistics(gd, dd, gs, total):
    floor = dd.mean().clamp_min(1e-30) * 1e-6
    diagonal = gd / dd.clamp_min(floor)
    root = gs / total.clamp_min(1e-30)
    if not bool(torch.isfinite(diagonal).all() and torch.isfinite(root).all()):
        raise RuntimeError('Non-finite LS-ViT estimate')
    return diagonal.float().unsqueeze(0), root.float().unsqueeze(0)


def estimate_lsvit(reconstructor, block, device):
    captured = {}
    def forward(module, inputs, output):
        captured['output'] = output.detach()
    def backward(module, grad_inputs, grad_outputs):
        captured['gradient'] = grad_outputs[0].detach()
    hooks = [block.register_forward_hook(forward), block.register_full_backward_hook(backward)]
    accumulated = None
    try:
        for index, (images, _) in enumerate(reconstructor.calib_loader):
            reconstructor.model.zero_grad(set_to_none=True)
            prediction = reconstructor.model(images.to(device)) / reconstructor.temperature
            loss = torch.nn.functional.kl_div(prediction.log_softmax(-1),
                reconstructor.raw_pred_softmaxs[index].to(device), reduction='batchmean')
            loss.backward()
            start = index * reconstructor.calib_loader.batch_size
            target = block.raw_out[start:start + len(images)].to(device)
            stats = ls_statistics(captured['gradient'], captured['output'] - target)
            if accumulated is None:
                accumulated = [x.clone() for x in stats]
            else:
                for total, value in zip(accumulated, stats):
                    total.add_(value)
            captured.clear()
    finally:
        for hook in hooks:
            hook.remove()
    block.ls_diagonal, block.ls_root = finish_statistics(*accumulated)


def ls_terms_per_image(prediction, target, diagonal, root):
    error = (prediction - target).abs().flatten(1)
    return (error.square() * diagonal).sum(1), (error * root).sum(1).square()


def ls_terms(prediction, target, diagonal, root):
    diag, rank_one = ls_terms_per_image(prediction, target, diagonal, root)
    return diag.mean(), rank_one.mean()


def fit_ls_mix(diag, rank_one, target):
    """Fit a positive KL scale on each half and score it on the other half."""
    diag, rank_one, target = (x.double() for x in (diag, rank_one, target))
    if not all(bool(torch.isfinite(x).all()) for x in (diag, rank_one, target)):
        raise RuntimeError('Non-finite hard-path LS probe')
    halves = (slice(None, len(target) // 2), slice(len(target) // 2, None))
    scores = []
    for weight in MIX_CANDIDATES:
        prediction = (1 - weight) * diag + weight * rank_one
        errors = []
        for train, validation in (halves, halves[::-1]):
            x = prediction[train]
            scale = (x @ target[train] / (x @ x).clamp_min(1e-30)).clamp_min(0)
            errors.append(float((scale * prediction[validation] - target[validation]).square().mean()))
        scores.append(dict(weight=weight, fold_mse=errors))
    equal = scores[MIX_CANDIDATES.index(0.5)]
    eligible = [s for s in scores if all(err < ref * (1 - 1e-6)
                for err, ref in zip(s['fold_mse'], equal['fold_mse']))]
    selected = min(eligible, key=lambda s: (sum(s['fold_mse']), abs(s['weight'] - 0.5)), default=equal)
    return selected['weight'], scores


@torch.no_grad()
def probe_ls_mix(reconstructor, block, device, loss, seed, fingerprint_probe=False):
    """Compare FP and hard current block with the same quantized prefix and FP suffix."""
    dataset = reconstructor.calib_loader.dataset
    count = min(128, len(dataset))
    if count < 4:
        raise ValueError('LS mixture probe requires at least four calibration images')
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    diag, rank_one, kl = [], [], []
    captured = []
    hook = block.register_forward_hook(lambda _module, _inputs, output: captured.append(output.detach()))
    try:
        with isolated_rng(device, seed), ExitStack() as paths:
            for name in reconstructor.quanted_blocks:
                paths.enter_context(quantization_path(reconstructor.blocks[name], False, 1.0, training=False))
            paths.enter_context(quantization_path(block, False, 1.0, training=False))
            for start in range(0, count, reconstructor.batch_size):
                chosen = indices[start:start + reconstructor.batch_size]
                images = torch.stack([dataset[i][0] for i in chosen]).to(device)
                reconstructor.set_block_mode(block, 'raw')
                fp_logits = reconstructor.model(images).double().log_softmax(-1)
                reconstructor.set_block_mode(block, 'quant_forward')
                quant_logits = reconstructor.model(images).double().log_softmax(-1)
                fp_out, quant_out = captured
                captured.clear()
                d, r = ls_terms_per_image(quant_out, fp_out, block.ls_diagonal, block.ls_root)
                diag.append((d / loss.init_loss_1).cpu())
                rank_one.append((r / loss.init_loss_2).cpu())
                kl.append(F.kl_div(quant_logits, fp_logits, reduction='none', log_target=True)
                          .sum(-1).clamp_min(0).cpu())
    finally:
        hook.remove()
    diag, rank_one, target = torch.cat(diag), torch.cat(rank_one), torch.cat(kl)
    weight, scores = fit_ls_mix(diag, rank_one, target)
    report = dict(probe_samples=count, probe_seed=seed, mean_kl=float(target.mean()),
                  suggested_weight=weight, candidates=scores)
    if fingerprint_probe:
        from utils.lsvit_pair import fingerprint
        report['probe_sha256'] = fingerprint((indices, diag, rank_one, target))
    return report
