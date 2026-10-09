"""Small residual branch correction for quantized ViT sublayers."""

import torch
from torch import nn


class LowRankCorrection(nn.Module):
    """A zero-initialized rank-r correction; the initial model is unchanged."""

    def __init__(self, width, rank):
        super().__init__()
        if rank < 1 or rank > width:
            raise ValueError('rank must be in [1, width]')
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x))


class CorrectedBranch(nn.Module):
    """Apply a correction after Attention or MLP and before residual addition."""

    def __init__(self, base, width, rank):
        super().__init__()
        self.base = base
        self.adapter = LowRankCorrection(width, rank)

    def forward(self, *args, **kwargs):
        output = self.base(*args, **kwargs)
        return output + self.adapter(output)


def normalized_feature_loss(student_features, teacher_features, importance=None):
    """Compare branch outputs with optional per-channel Fisher importance."""

    losses = []
    for name, target in teacher_features.items():
        prediction = student_features[name]
        if prediction.shape != target.shape:
            raise ValueError('feature shapes differ for {}'.format(name))
        squared_error = (prediction - target).square()
        if importance is not None:
            channel_weight = importance[name].to(
                device=squared_error.device, dtype=squared_error.dtype)
            if channel_weight.numel() != squared_error.shape[-1]:
                raise ValueError('Fisher weight shape differs for {}'.format(name))
            squared_error = squared_error * channel_weight
        losses.append(squared_error.mean() / target.square().mean().clamp_min(1e-6))
    if not losses:
        raise ValueError('no branch features were captured')
    return torch.stack(losses).mean()
