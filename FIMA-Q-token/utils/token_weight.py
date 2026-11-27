"""Token weights for matched CLS and Fisher patch reweighting in DPLR."""

PROTOCOL = 'dplr_matched_cls_sqrt_v1'
CLS_SCALE = 16.0
SMOKE_STEPS = 2
MODES = ('cls', 'gradient')


def matched_token_weights(tokens, cls_scale, device, patch_scores=None):
    """Return quadratic weights with the same CLS and total patch weight in both arms."""
    import torch

    weights = torch.ones(tokens, device=device, dtype=torch.float32)
    source = 'uniform_patch'
    if patch_scores is not None:
        if patch_scores.shape != (tokens - 1,):
            raise ValueError('Expected one Fisher sensitivity per patch position')
        scores = patch_scores.to(device=device, dtype=weights.dtype)
        mean = scores.mean()
        if bool(torch.isfinite(scores).all() and (scores >= 0).all() and mean > 1e-12):
            # Shrink to uniform weights and preserve the sum across patches.
            weights[1:] = 0.5 + 0.5 * (scores / mean)
            source = 'fisher_patch'
        else:
            source = 'uniform_patch_fallback'
    weights[0] = cls_scale
    return weights * (tokens / (cls_scale + tokens - 1)), source
