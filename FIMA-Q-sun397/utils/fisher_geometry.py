"""Directional Fisher diagnostics, independent of timm and training code.

Directions are rows of an orthonormal basis. Reported Fisher matrices live in
the supplied direction coordinates, not in the full token/channel space.
"""
import torch


def fixed_error_directions(error, groups=1):
    """Represent one quantization error by fixed, disjoint channel groups.

    Returns unit-norm directions as rows and coefficients whose linear
    combination reconstructs ``error`` exactly.  ``groups=1`` is the complete
    error direction.  Larger values split the final (channel) dimension into a
    fixed number of contiguous groups; no data-dependent rank selection is
    performed.
    """
    if error.ndim < 1 or error.numel() == 0:
        raise ValueError('error must be a non-empty tensor')
    if groups < 1 or groups > error.shape[-1]:
        raise ValueError('groups must be between 1 and the channel count')
    directions, coefficients = [], []
    for indices in torch.tensor_split(torch.arange(error.shape[-1], device=error.device), groups):
        component = torch.zeros_like(error)
        component[..., indices] = error[..., indices]
        coefficient = component.norm()
        if float(coefficient) > 0:
            direction = component / coefficient
        else:
            # Preserve a fixed group count for the degenerate zero-error case.
            direction = torch.zeros_like(error)
            direction[..., indices] = 1 / (component[..., indices].numel() ** 0.5)
        directions.append(direction.flatten())
        coefficients.append(coefficient)
    return torch.stack(directions), torch.stack(coefficients)


def energy_rank(values, fraction=0.95):
    """Smallest leading rank reaching an energy fraction, computed on CPU.

    PyTorch does not provide a deterministic CUDA cumsum kernel on every
    supported version. This is reporting-only scalar work, so moving the
    detached eigenvalues to CPU preserves the result without weakening strict
    determinism for model/Fisher computation.
    """
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0, 1]')
    values = values.detach().double().cpu().tolist()
    total = sum(values)
    if total <= 0:
        return 0
    threshold, cumulative = total * fraction, 0.0
    for index, value in enumerate(values, 1):
        cumulative += value
        if cumulative >= threshold:
            return index
    return len(values)


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
    rank95 = energy_rank(values, 0.95)
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
