"""Deployment-path KD primitives; no timm dependency, no inference adapters."""
import contextlib

import torch
from torch import nn
from torch.nn import functional as F

from quantizers.uniform import UniformQuantizer


ARMS = {
    "E1": (None, None),
    "E2": (None, "kl"),
    "E3": (None, "standardized"),
    "E4": ("fp", None),
    "E5": ("quant", None),
    "E6": ("quant", "standardized"),
}


def distillation(student, teacher, temperature=2.0, standardized=False):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher = teacher.detach().float()
    student = student.float()
    if standardized:
        # Population std, per image, over classes (Sun et al., CVPR 2024).
        def normalize(z):
            return (z - z.mean(-1, keepdim=True)) / z.std(
                -1, unbiased=False, keepdim=True).clamp_min(1e-6)
        student, teacher = normalize(student), normalize(teacher)
    return F.kl_div(F.log_softmax(student / temperature, -1),
                    F.softmax(teacher / temperature, -1),
                    reduction="batchmean") * temperature ** 2


def quant_mode(model, ste=False, activation_drop=1.0):
    model.eval()
    for name, module in model.named_modules():
        if hasattr(module, "mode"):
            module.mode = "quant_forward"
            module.calibrated = True
        if isinstance(module, UniformQuantizer):
            module.training_mode = ste
            module.drop_prob = 1.0 if name.endswith("w_quantizer") else activation_drop


@contextlib.contextmanager
def quant_state(model, ste=False, activation_drop=1.0):
    saved = [(m, {a: getattr(m, a) for a in ("mode", "training_mode", "drop_prob")
                   if hasattr(m, a)}) for m in model.modules()]
    quant_mode(model, ste, activation_drop)
    try:
        yield
    finally:
        for module, values in saved:
            for name, value in values.items():
                setattr(module, name, value)


def vit_prefix(model, images, block_index):
    x = model.patch_embed(images)
    if not hasattr(model, "_pos_embed"):
        raise TypeError("This implementation requires timm ViT._pos_embed")
    x = model._pos_embed(x)
    if hasattr(model, "patch_drop"):
        x = model.patch_drop(x)
    if hasattr(model, "norm_pre"):
        x = model.norm_pre(x)
    for block in model.blocks[:block_index]:
        x = block(x)
    return x


def vit_suffix(model, value, block_index):
    x = value
    for block in model.blocks[block_index + 1:]:
        x = block(x)
    x = model.norm(x)
    return model.forward_head(x)


def activation_scales(block):
    """Only existing activation scales; weight codes/scales and zero points fixed."""
    selected = {}
    for name, module in block.named_modules():
        if isinstance(module, UniformQuantizer) and not name.endswith("w_quantizer"):
            if module.n_bits < 32:
                if not isinstance(module.scale, nn.Parameter):
                    raise TypeError("Expected an existing Parameter for " + name)
                selected[name + ".scale"] = module.scale
    if not selected:
        raise ValueError("No activation scales in selected block")
    return selected


def clamp_scales(parameters, centers, radius):
    with torch.no_grad():
        for name, parameter in parameters.items():
            base = centers[name].to(parameter)
            parameter.copy_(torch.maximum(base * (1 - radius),
                            torch.minimum(base * (1 + radius), parameter)))


def dplr_terms(prediction, target, gradients, inverse):
    """Same abs-error/abs-gradient convention as FIMA-Q's current implementation."""
    error = (prediction - target).abs().flatten(1)
    projected = error @ gradients.t()
    low_rank = ((projected @ inverse) * projected).sum(-1).mean().clamp_min(0)
    diagonal = (error.square() * gradients.mean(0)).mean()
    return low_rank, diagonal


def dplr_loss(prediction, target, anchor):
    low, diag = dplr_terms(prediction, target, anchor["gradient"], anchor["inverse"])
    return low / anchor["low_norm"] + diag / anchor["diag_norm"]


def estimate_anchor(student, teacher, block_index, inputs, targets, global_logits,
                    count=128, rank=5, batch_size=8, temperature=20.0, seed=3407):
    """Fixed FIMA-style anchor for a NEW scale-only second pass.

    Float suffix, 0.5 activation QDrop probes, abs batch-aggregated secants,
    rank-5 Gram with ridge. This is NOT replaying original 20k progressive
    AdaRound. All arms use this identical regularizer. Holdout is never used.
    """
    device = next(student.parameters()).device
    count = min(count, len(inputs))
    gradients, directions = [], []
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        with quant_state(student.blocks[block_index], ste=True, activation_drop=0.5):
            for _ in range(rank):
                gradient_sum = direction_sum = None
                for start in range(0, count, batch_size):
                    stop = min(start + batch_size, count)
                    with torch.no_grad():
                        output = student.blocks[block_index](inputs[start:stop].to(device))
                    output = output.detach().requires_grad_(True)
                    logits = vit_suffix(teacher, output, block_index)
                    loss = distillation(logits, global_logits[start:stop].to(device), temperature)
                    gradient = torch.autograd.grad(loss, output)[0].detach().abs().sum(0)
                    # Remove batchmean scaling. T^2 is common and removed by normalization.
                    gradient = gradient * (stop - start)
                    direction = (output.detach() - targets[start:stop].to(device)).abs().sum(0)
                    gradient_sum = gradient if gradient_sum is None else gradient_sum + gradient
                    direction_sum = direction if direction_sum is None else direction_sum + direction
                gradients.append((gradient_sum / count).flatten())
                directions.append((direction_sum / count).flatten())
    gradient = torch.stack(gradients)
    direction = torch.stack(directions)
    gram = direction.double() @ direction.double().t()
    ridge = gram.diagonal().mean().clamp_min(1e-20) * 1e-6
    inverse = torch.linalg.inv(gram + torch.eye(rank, device=device, dtype=gram.dtype) * ridge).float()
    # Fixed full-quantized normalization, independent of optimization minibatches.
    low_sum = diag_sum = 0.0
    with quant_state(student.blocks[block_index], ste=False), torch.no_grad():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            output = student.blocks[block_index](inputs[start:stop].to(device))
            low, diag = dplr_terms(output, targets[start:stop].to(device), gradient, inverse)
            low_sum += float(low) * (stop - start)
            diag_sum += float(diag) * (stop - start)
    values = dict(gradient=gradient.detach(), inverse=inverse.detach(),
                  low_norm=max(low_sum / count, 1e-12),
                  diag_norm=max(diag_sum / count, 1e-12),
                  gram_condition=float(torch.linalg.cond(gram + torch.eye(
                      rank, device=device, dtype=gram.dtype) * ridge)))
    if not all(torch.isfinite(v).all() for v in (gradient, inverse)):
        raise FloatingPointError("Non-finite fixed DPLR anchor")
    return values


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def anchor_to(anchor, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in anchor.items()}
