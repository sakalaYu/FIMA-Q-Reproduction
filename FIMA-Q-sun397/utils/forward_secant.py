"""Per-sample forward-secant reconstruction losses.

This module is independent of timm so the defining loss can be tested without
building a vision model.
"""
import torch


def forward_secant_normalizers(anchor_direction, anchor_kl, eps=1e-12):
    """Return fixed anchor scales for the projection and MSE residual."""
    if anchor_direction.shape[0] != anchor_kl.numel():
        raise ValueError('anchor_kl must contain one value per sample')
    secant_scale = anchor_kl.detach().reshape(-1).clamp_min(0).mean().clamp_min(eps)
    residual_scale = anchor_direction.detach().square().mean().clamp_min(eps)
    return secant_scale, residual_scale


def forward_secant_terms(prediction, target, anchor_direction, anchor_kl, eps=1e-12):
    """Return the secant projection loss and an isotropic MSE residual.

    Every tensor uses the same leading sample dimension. ``anchor_direction``
    is the signed quantization error observed during a forward-only refresh and
    ``anchor_kl`` contains one output KL value per sample.
    """
    if prediction.shape != target.shape or prediction.shape != anchor_direction.shape:
        raise ValueError('prediction, target and anchor_direction must have identical shapes')
    if prediction.shape[0] != anchor_kl.numel():
        raise ValueError('anchor_kl must contain one value per sample')
    error = (prediction - target).reshape(prediction.shape[0], -1)
    direction = anchor_direction.reshape(anchor_direction.shape[0], -1).detach()
    weights = anchor_kl.reshape(-1).detach().clamp_min(0)
    denominator = direction.square().sum(dim=1).clamp_min(eps)
    projection = (error * direction).sum(dim=1) / denominator
    secant = (weights * projection.square()).mean()
    residual = error.square().mean()
    return secant, residual
