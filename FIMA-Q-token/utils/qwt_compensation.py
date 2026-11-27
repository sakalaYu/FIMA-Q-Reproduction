"""QwT-style closed-form block compensation for a reconstructed ViT.

Reference: https://github.com/wujx2001/QwT (CVPR 2025).
This integration uses streaming FP64 sufficient statistics, a minimum-norm
least-squares solve and an FP16 inference branch by default. It is an existing
method baseline, not a new contribution. No validation labels enter the fit.
"""

import math
import time

import torch
from torch import nn
from torch.nn import functional as F


class LinearCompensatedBlock(nn.Module):
    def __init__(self, base, weight, bias):
        super().__init__()
        self.base = base
        self.register_buffer('qwt_weight', weight.detach().contiguous())
        self.register_buffer('qwt_bias', bias.detach().contiguous())

    def forward(self, x):
        y = self.base(x)
        correction = F.linear(x.to(self.qwt_weight.dtype), self.qwt_weight, self.qwt_bias)
        return y + correction.to(y.dtype)


class RegressionStats:
    """CPU FP64 normal equations; memory depends on channels, not token count."""
    def __init__(self, width):
        self.xx = torch.zeros(width + 1, width + 1, dtype=torch.float64)
        self.xy = torch.zeros(width + 1, width, dtype=torch.float64)
        self.yy = 0.0
        self.y_sum = torch.zeros(width, dtype=torch.float64)
        self.rows = 0

    def update(self, x, residual):
        x = x.detach().reshape(-1, x.shape[-1]).to(device='cpu', dtype=torch.float64)
        y = residual.detach().reshape(-1, residual.shape[-1]).to(device='cpu', dtype=torch.float64)
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise RuntimeError('Non-finite QwT regression features')
        design = torch.cat((x, torch.ones(len(x), 1, dtype=x.dtype)), dim=1)
        self.xx.add_(design.T @ design)
        self.xy.add_(design.T @ y)
        self.yy += float(y.square().sum())
        self.y_sum.add_(y.sum(0))
        self.rows += len(y)

    def solve(self, ridge=0.0):
        if not self.rows:
            raise ValueError('Empty QwT regression')
        gram = self.xx.clone()
        if ridge:
            # Optional ridge is scaled per observation; intercept is unpenalized.
            penalty = torch.eye(len(gram), dtype=gram.dtype) * ridge * self.rows
            penalty[-1, -1] = 0
            gram += penalty
        # LayerNorm features can be rank deficient. Avoid an explicit inverse.
        solution = torch.linalg.lstsq(gram, self.xy, driver='gelsd', rcond=1e-10).solution
        if not bool(torch.isfinite(solution).all()):
            raise RuntimeError('Non-finite QwT coefficients')
        sse = max(0.0, self.yy - 2 * float((solution * self.xy).sum())
                  + float((solution * (self.xx @ solution)).sum()))
        sst = max(0.0, self.yy - float(self.y_sum.square().sum()) / self.rows)
        r2 = 1.0 - sse / sst if sst > 1e-20 else 0.0
        accepted = math.isfinite(r2) and r2 > 0
        if not accepted:
            solution.zero_()
        return solution[:-1].T.float(), solution[-1].float(), dict(
            training_r2=r2, accepted=accepted, train_residual_mse=self.yy / self.rows / len(self.y_sum),
            fitted_residual_mse=sse / self.rows / len(self.y_sum), token_rows=self.rows)


class _CapturedInput(Exception):
    pass


@torch.no_grad()
def capture_first_input(model, loader, samples, device):
    chunks, count = [], 0

    def capture(module, inputs):
        chunks.append(inputs[0].detach().cpu())
        raise _CapturedInput()

    hook = model.blocks[0].register_forward_pre_hook(capture)
    try:
        for images, _ in loader:
            remaining = samples - count
            if remaining <= 0:
                break
            images = images[:remaining]
            try:
                model(images.to(device))
            except _CapturedInput:
                pass
            else:
                raise RuntimeError('Model did not reach its first Transformer block')
            count += len(images)
    finally:
        hook.remove()
    if count != samples:
        raise ValueError('Insufficient cached calibration images for QwT')
    return torch.cat(chunks)


@torch.no_grad()
def fit_qwt(model, teacher, loader, samples=512, batch_size=8, ridge=0.0,
            branch_dtype='fp16', solver_threads=4):
    if len(model.blocks) != len(teacher.blocks):
        raise ValueError('QwT teacher/student block mismatch')
    device = next(model.parameters()).device
    teacher.to(device)
    dtype = {'fp16': torch.float16, 'fp32': torch.float32}[branch_dtype]
    started = time.perf_counter()
    features = capture_first_input(model, loader, samples, device)
    reports = []
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(solver_threads)
        for index in range(len(model.blocks)):
            block_started = time.perf_counter()
            base = model.blocks[index]
            stats = RegressionStats(features.shape[-1])
            # Match the QwT local target: FP and quantized block see the SAME
            # input from the already compensated quantized prefix.
            for start in range(0, samples, batch_size):
                x = features[start:start + batch_size].to(device)
                stats.update(x, teacher.blocks[index](x) - base(x))
            weight, bias, report = stats.solve(ridge)
            if not bool(torch.isfinite(weight.to(dtype)).all() and torch.isfinite(bias.to(dtype)).all()):
                raise RuntimeError('QwT coefficients overflow the requested branch precision')
            model.blocks[index] = LinearCompensatedBlock(
                base, weight.to(device=device, dtype=dtype), bias.to(device=device, dtype=dtype)).eval()
            next_features = torch.empty_like(features)
            deployed_sse, elements = 0.0, 0
            for start in range(0, samples, batch_size):
                x = features[start:start + batch_size].to(device)
                prediction = model.blocks[index](x)
                if not bool(torch.isfinite(prediction).all()):
                    raise RuntimeError('Non-finite deployed QwT block output')
                target = teacher.blocks[index](x)
                deployed_sse += float((prediction - target).square().sum())
                elements += prediction.numel()
                next_features[start:start + len(x)] = prediction.cpu()
            width = features.shape[-1]
            report.update(block=f'blocks.{index}', branch_dtype=branch_dtype,
                          deployed_training_mse=deployed_sse / elements,
                          added_parameters=width * width + width,
                          added_macs_per_image=features.shape[1] * width * width,
                          elapsed_seconds=time.perf_counter() - block_started)
            reports.append(report)
            features = next_features
            import json
            import logging
            logging.info('QwT block result: %s', json.dumps(report, sort_keys=True))
    finally:
        torch.set_num_threads(old_threads)
    return dict(method='qwt_closed_form_integration_v1', samples=samples, ridge=ridge,
                target='FP block on same compensated-prefix input', branch_dtype=branch_dtype,
                inference_precision=f'W3A3 backbone plus {branch_dtype} compensation', blocks=reports,
                added_parameters=sum(r['added_parameters'] for r in reports),
                added_storage_bytes=sum(r['added_parameters'] for r in reports) * torch.empty((), dtype=dtype).element_size(),
                added_macs_per_image=sum(r['added_macs_per_image'] for r in reports),
                elapsed_seconds=time.perf_counter() - started)


def attach_saved_qwt(model, state):
    """Recreate wrappers BEFORE the ordinary quantizer checkpoint loader runs."""
    for index in range(len(model.blocks)):
        key = f'blocks.{index}.qwt_weight'
        if key in state:
            model.blocks[index] = LinearCompensatedBlock(
                model.blocks[index], torch.zeros_like(state[key]),
                torch.zeros_like(state[f'blocks.{index}.qwt_bias']))
