"""Experimental activation-grid MR. Loaded only by run_mr_grid.py's child.

ReLU MR and plain/clipped supervision are borrowed from APHQ-ViT / GLF-Q.
STE and quantization-aware reconstruction are established techniques. The
candidate tested here combines nearby-grid worst-case supervision with a
holdout FP-fidelity constraint. Novelty and end-to-end gains are unverified.
"""

import logging
import math

import torch
from torch.nn import functional as F

from utils import lsvit_mr as mr
from utils.lsvit_pair import fingerprint


def fake_a3(x, scale):
    """Unsigned 3-bit affine grid; STE through rounding, exact clipping gradient."""
    clipped = (x / scale).clamp(0, 7)
    return (clipped + (clipped.round() - clipped).detach()) * scale


@torch.no_grad()
def measure(student, inputs, targets, grid, device):
    fp, nominal = mr.score_mlp(student, inputs, targets, grid, device)
    errors = [mr.score_mlp(student, inputs, targets, (grid[0] * factor, 0.), device)[1]
              for factor in (0.9, 1.1)]
    return dict(fp_nmse=fp, a3_nmse=nominal, worst_a3_nmse=max(nominal, *errors))


def fit_mlp(teacher, inputs, targets, device, steps, seed, holdout, variant):
    if steps != 3000:
        raise ValueError('Grid MR requires 3000 attempted updates per block')
    n = len(inputs) - holdout
    train_x, train_y = inputs[:n], targets[:n]
    val_x, val_y = inputs[n:], targets[n:]
    sample = mr.sample_activations(teacher, train_x, device)
    positive = sample[sample > 0]
    clip = float(torch.quantile(positive, .99)) if len(positive) else 0.
    baseline_grid = mr.fit_a3_range(sample)
    fit = mr.start_fit(teacher, steps, seed)
    student, optimizer, scheduler, generator = fit
    logging.info('Grid MR %s: shared warmup 0 -> 2000 updates', variant)
    mr.train_fit(fit, train_x, train_y, device, 2000, clip)
    anchor_sha = fingerprint((student.state_dict(), optimizer.state_dict(),
                              scheduler.state_dict(), generator.get_state()))
    grid = mr.fit_a3_range(mr.sample_activations(student, train_x, device))
    if grid[1] != 0.:
        raise ValueError('ReLU activation grid must have zero integer zero point')
    anchor = measure(student, val_x, val_y, grid, device)
    best = anchor
    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    selected_step = 2000
    history = [dict(step=2000, **anchor)]
    logging.info('Grid MR %s anchor: fp=%.6g a3=%.6g worst=%.6g',
                 variant, anchor['fp_nmse'], anchor['a3_nmse'], anchor['worst_a3_nmse'])
    if variant == 'uniform':
        mr.train_fit(fit, train_x, train_y, device, 1000, clip)
    else:
        for step in range(2001, 3001):
            ids = torch.randint(n, (4,), generator=generator)
            x, target = train_x[ids].to(device), train_y[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            with mr.capture_fc2_input(student) as captured:
                out = student(x)
                hidden = captured.pop()
            clipped = F.linear(hidden.clamp_max(clip), student.fc2.weight, student.fc2.bias)
            base_loss = F.mse_loss(out, target) + 2 * F.mse_loss(clipped, target)
            factors = (1.,) if variant == 'single' else (.9, 1., 1.1)
            q_losses = [F.mse_loss(F.linear(fake_a3(hidden, grid[0] * factor),
                                         student.fc2.weight, student.fc2.bias), target)
                        for factor in factors]
            ramp = min((step - 2000) / 500., 1.)
            loss = base_loss + .25 * ramp * torch.stack(q_losses).max()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f'Nonfinite grid MR loss at step {step}')
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 250 == 0:
                grid = mr.fit_a3_range(mr.sample_activations(student, train_x, device))
                metrics = measure(student, val_x, val_y, grid, device)
                feasible = metrics['fp_nmse'] <= max(anchor['fp_nmse'] * 1.1, 1e-10)
                objective = 'a3_nmse' if variant == 'single' else 'worst_a3_nmse'
                accept = feasible and metrics[objective] < best[objective]
                history.append(dict(step=step, feasible=feasible, accepted=accept, **metrics))
                if accept:
                    best = metrics
                    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                    selected_step = step
                logging.info('Grid MR %s step=%d fp=%.6g a3=%.6g worst=%.6g accepted=%s',
                             variant, step, metrics['fp_nmse'], metrics['a3_nmse'],
                             metrics['worst_a3_nmse'], accept)
        student.load_state_dict(best_state)
    final_grid = mr.fit_a3_range(mr.sample_activations(student, train_x, device))
    final = measure(student, val_x, val_y, final_grid, device)
    if not all(math.isfinite(v) for v in final.values()):
        raise RuntimeError('Nonfinite grid MR diagnostics')
    _, before = mr.score_mlp(teacher, val_x, val_y, baseline_grid, device)
    report = dict(clip=clip, gelu_a3_nmse=before, relu_fp_nmse=final['fp_nmse'],
                  relu_a3_nmse=final['a3_nmse'], worst_a3_nmse=final['worst_a3_nmse'],
                  anchor_sha256=anchor_sha, grid_variant=variant,
                  selected_step=3000 if variant == 'uniform' else selected_step,
                  attempted_steps=3000, anchor_metrics=anchor, selection_history=history)
    logging.info('Grid MR %s complete: attempted=3000 selected=%d', variant, report['selected_step'])
    return student, report
