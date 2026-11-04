"""Streaming, damped LS-ViT curvature statistics.

Equations follow hhh7748/LS-ViT utils/block_recon.py cal_hessian.
Relative denominator floors and streaming accumulation are integration changes.
"""
import torch


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


def ls_terms(prediction, target, diagonal, root):
    error = (prediction - target).abs().flatten(1)
    return (error.square() * diagonal).sum(1).mean(), (error * root).sum(1).square().mean()
