"""Low-dimensional Fisher diagnostics, independent of timm and training code.

Directions are ROWS of an orthonormal basis. All reported Fisher matrices live
in this basis, not in the full token/channel coordinate system.
"""
import torch


def error_basis(errors, max_rank, tolerance=1e-8):
    """Uncentered signed error SVD via a small sample Gram matrix."""
    x = errors.reshape(errors.shape[0], -1).double()
    gram = x @ x.T
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    values, vectors = values[order].clamp_min(0), vectors[:, order]
    if values[0] <= 0:
        raise ValueError('All quantization errors are zero; no error subspace to estimate')
    numerical_rank = int((values > values[0] * tolerance).sum())
    rank = min(max_rank, numerical_rank)
    basis = (vectors[:, :rank].T @ x) / values[:rank].sqrt()[:, None]
    return basis.float(), values, numerical_rank


def projected_fisher(responses, probabilities):
    """responses = d(logits / T)/d(coeff), shape [classes, directions]."""
    v, p = responses.double(), probabilities.double()
    mean = p @ v
    result = v.T @ (p[:, None] * v) - mean[:, None] * mean[None, :]
    return (result + result.T) * 0.5


def response_jacobian(fn, rank, device, method, step=None):
    """Autograd JVP reference or forward-only central differences.

    fn maps subspace coefficients to temperature-scaled logits for ONE image.
    Finite differences never construct a gradient graph.
    """
    if method not in ('autograd', 'finite_difference'):
        raise ValueError('Unknown response method')
    if method == 'finite_difference' and (step is None or step <= 0):
        raise ValueError('A positive finite-difference step is required')
    zero = torch.zeros(rank, device=device)
    columns = []
    for j in range(rank):
        unit = torch.zeros_like(zero)
        unit[j] = 1
        if method == 'autograd':
            with torch.enable_grad():
                _, response = torch.autograd.functional.jvp(fn, zero, unit, create_graph=False)
        else:
            with torch.no_grad():
                # Subtract in double precision, but network evaluation remains FP32.
                response = (fn(unit * step).double() - fn(-unit * step).double()) / (2 * step)
        columns.append(response.detach().double())
    return torch.stack(columns, dim=-1)


def matrix_summary(matrix):
    matrix = matrix.double()
    values = torch.linalg.eigvalsh(matrix).flip(0).clamp_min(0)
    total = float(values.sum())
    rank95 = int(torch.searchsorted(values.cumsum(0), values.sum() * 0.95)) + 1 if total > 0 else 0
    positive = values[values > max(float(values[0]) * 1e-8, 1e-30)]
    norm = matrix.norm().clamp_min(1e-30)
    return dict(eigenvalues=values.cpu().tolist(), trace=total, rank95=rank95,
                numerical_rank=len(positive),
                condition_positive=float(positive[0] / positive[-1]) if len(positive) else None,
                full_condition=float(values[0] / values[-1]) if len(positive) == len(values) else None,
                projected_offdiag_ratio=float((matrix - matrix.diag().diag()).norm() / norm))


def quadratic_predictions(matrix, coeff):
    """Full/diagonal/rank-half predictions in the SAME projected coordinates."""
    f, a = matrix.double(), coeff.double()
    values, vectors = torch.linalg.eigh(f)
    keep = max(1, len(a) // 2)
    low = (vectors[:, -keep:] * values[-keep:].clamp_min(0)) @ vectors[:, -keep:].T
    return dict(full=float(0.5 * a @ f @ a),
                diagonal=float(0.5 * (a.square() * f.diag()).sum()),
                low_rank=float(0.5 * a @ low @ a))
