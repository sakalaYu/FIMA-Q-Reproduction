"""Unlabeled strata and group-robust loss for Fisher block reconstruction."""
import math

import torch


def make_groups(teacher_prob, quant_error):
    """Split samples at the rank median of entropy and block error.

    Rank splits avoid empty halves when confidence values are tied. Group IDs
    are fixed for a block after its first Fisher refresh.
    """
    if teacher_prob.ndim != 2 or quant_error.ndim != 1:
        raise ValueError('Expected probabilities [N,C] and errors [N]')
    if teacher_prob.shape[0] != quant_error.numel() or quant_error.numel() < 4:
        raise ValueError('Group construction needs at least four matched samples')
    if not torch.isfinite(teacher_prob).all() or not torch.isfinite(quant_error).all():
        raise ValueError('Non-finite group inputs')
    prob = teacher_prob.detach().float().cpu().clamp_min(1e-12)
    entropy = -(prob * prob.log()).sum(dim=1)
    error = quant_error.detach().float().cpu()
    n = error.numel()
    high_entropy = torch.zeros(n, dtype=torch.long)
    high_error = torch.zeros(n, dtype=torch.long)
    high_entropy[torch.argsort(entropy, stable=True)[n // 2:]] = 1
    high_error[torch.argsort(error, stable=True)[n // 2:]] = 1
    return 2 * high_entropy + high_error


def balanced_indices(groups, batch_size):
    """Draw an equal-size minibatch from each occupied stratum, with replacement."""
    groups = groups.detach().long().cpu()
    unique = torch.unique(groups, sorted=True)
    if batch_size < unique.numel():
        raise ValueError('Batch must hold at least one sample per occupied group')
    base, remainder = divmod(batch_size, unique.numel())
    pieces = []
    for j, group in enumerate(unique):
        members = torch.where(groups == group)[0]
        count = base + (j < remainder)
        pieces.append(members[torch.randint(members.numel(), (count,))])
    index = torch.cat(pieces)
    return index[torch.randperm(index.numel())]


def fisher_sample_terms(pred, target, grad, inverse_b):
    """The two per-sample DPLR terms, using FIMA-Q's absolute error convention."""
    error = (pred - target).abs().reshape(pred.shape[0], -1)
    gradient = grad.abs()
    projection = error @ gradient.transpose(0, 1)
    low_rank = torch.einsum('bi,ij,bj->b', projection, inverse_b, projection)
    diagonal = (error.square() * gradient.mean(dim=0)).mean(dim=1)
    return low_rank, diagonal


def group_robust_loss(sample_loss, groups, beta=0.5, temperature=0.5):
    """Mixture of ERM and a smooth maximum over occupied group means."""
    if sample_loss.ndim != 1 or sample_loss.numel() != groups.numel():
        raise ValueError('Loss and group lengths must match')
    if not 0 <= beta <= 1 or temperature <= 0:
        raise ValueError('Expected beta in [0,1] and temperature > 0')
    group_means = torch.stack([
        sample_loss[groups == group].mean() for group in torch.unique(groups, sorted=True)
    ])
    smooth_max = temperature * (torch.logsumexp(group_means / temperature, dim=0)
                                - math.log(group_means.numel()))
    return (1 - beta) * sample_loss.mean() + beta * smooth_max
