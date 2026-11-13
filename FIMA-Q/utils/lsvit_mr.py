"""Fit ReLU MLPs before LS-ViT quantization and screen their A3 activation error."""

from contextlib import contextmanager
import copy
import json
import logging
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from utils.lsvit_pair import file_sha, write_json


@contextmanager
def capture_fc2_input(mlp):
    captured = []
    handle = mlp.fc2.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    try:
        yield captured
    finally:
        handle.remove()


@torch.no_grad()
def teacher_features(model, mlp, loader, device):
    inputs, outputs = [], []

    def capture(_, args, output):
        inputs.append(args[0].cpu())
        outputs.append(output.cpu())

    handle = mlp.register_forward_hook(capture)
    try:
        for images, _ in loader:
            model(images.to(device))
    finally:
        handle.remove()
    return torch.cat(inputs), torch.cat(outputs)


@torch.no_grad()
def sample_activations(mlp, inputs, device):
    parts = []
    with capture_fc2_input(mlp) as captured:
        for x in inputs[:32].split(4):
            mlp(x.to(device))
            parts.append(captured.pop().flatten()[::53].cpu())
    return torch.cat(parts)


def quantize_a3(x, scale, zero):
    return ((x / scale).round().add(zero).clamp(0, 7) - zero) * scale


def fit_a3_range(sample):
    # Screen with the same affine integer grid as UniformQuantizer. The full
    # pipeline still calibrates its own scales with the normal MSE search.
    tails = torch.tensor([0., 0.001, 0.005, 0.01, 0.02])
    low = torch.quantile(sample, tails).clamp_max(0.)
    high = torch.quantile(sample, 1. - tails).clamp_min(0.)
    scale = ((high - low) / 7).clamp_min(1e-8)
    zero = (-low / scale).round().clamp(0, 7)
    error = (quantize_a3(sample[:, None], scale, zero) - sample[:, None]).square().mean(0)
    chosen = int(error.argmin())
    return float(scale[chosen]), float(zero[chosen])


@torch.no_grad()
def score_mlp(mlp, inputs, targets, grid, device):
    fp_error = a3_error = energy = 0.
    with capture_fc2_input(mlp) as captured:
        for x, target in zip(inputs.split(4), targets.split(4)):
            x, target = x.to(device), target.to(device)
            out = mlp(x)
            hidden = quantize_a3(captured.pop(), *grid)
            a3_out = F.linear(hidden, mlp.fc2.weight, mlp.fc2.bias)
            fp_error += float((out - target).square().sum())
            a3_error += float((a3_out - target).square().sum())
            energy += float(target.square().sum())
    return fp_error / max(energy, 1e-12), a3_error / max(energy, 1e-12)


def start_fit(teacher, steps, seed):
    student = copy.deepcopy(teacher)
    student.act = nn.ReLU()
    student.requires_grad_(False)
    params = list(student.fc1.parameters()) + list(student.fc2.parameters())
    for parameter in params:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(params, lr=4e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    generator = torch.Generator().manual_seed(seed)
    return student, optimizer, scheduler, generator


def train_fit(fit, inputs, targets, device, steps, clip):
    student, optimizer, scheduler, generator = fit
    with capture_fc2_input(student) as captured:
        for _ in range(steps):
            indices = torch.randint(len(inputs), (4,), generator=generator)
            x, target = inputs[indices].to(device), targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = student(x)
            hidden = captured.pop().clamp_max(clip)
            clipped = F.linear(hidden, student.fc2.weight, student.fc2.bias)
            loss = F.mse_loss(out, target) + 2 * F.mse_loss(clipped, target)
            loss.backward()
            optimizer.step()
            scheduler.step()


def fit_mlp(teacher, inputs, targets, device, steps, seed, holdout):
    train_count = len(inputs) - holdout
    sample = sample_activations(teacher, inputs[:train_count], device)
    positive = sample[sample > 0]
    clip = float(torch.quantile(positive, 0.99)) if len(positive) else 0.
    baseline_grid = fit_a3_range(sample)
    fit = start_fit(teacher, steps, seed)
    train_fit(fit, inputs[:train_count], targets[:train_count], device, steps, clip)
    student = fit[0]
    student_grid = fit_a3_range(sample_activations(student, inputs[:train_count], device))
    _, before_a3 = score_mlp(teacher, inputs[train_count:], targets[train_count:], baseline_grid, device)
    after_fp, after_a3 = score_mlp(student, inputs[train_count:], targets[train_count:], student_grid, device)
    return student, dict(clip=clip, relu_fp_nmse=after_fp, gelu_a3_nmse=before_a3, relu_a3_nmse=after_a3)


def residual_budgets(errors):
    """Allocate 2k/3k/4k to the bottom/middle/top thirds of 12 ViT blocks."""
    if len(errors) != 12 or not bool(torch.isfinite(torch.as_tensor(errors)).all()):
        raise ValueError('MR allocation requires finite A3 errors from all 12 blocks')
    budgets = [3000] * len(errors)
    order = sorted(range(len(errors)), key=lambda i: (errors[i], i))
    for index in order[:4]:
        budgets[index] = 2000
    for index in order[-4:]:
        budgets[index] = 4000
    return budgets


def fit_with_residual_budget(model, loader, device, seed, holdout):
    pilot_steps = 1000
    pilots = []
    for index, block in enumerate(model.blocks):
        teacher = block.mlp
        inputs, targets = teacher_features(model, teacher, loader, device)
        train_count = len(inputs) - holdout
        sample = sample_activations(teacher, inputs[:train_count], device)
        positive = sample[sample > 0]
        clip = float(torch.quantile(positive, 0.99)) if len(positive) else 0.
        fit = start_fit(teacher, 3000, seed + index)
        student, optimizer, scheduler, generator = fit
        train_fit(fit, inputs[:train_count], targets[:train_count], device, pilot_steps, clip)
        grid = fit_a3_range(sample_activations(student, inputs[:train_count], device))
        _, residual = score_mlp(student, inputs[train_count:], targets[train_count:], grid, device)
        # Keep the pilot weights, Adam moments and sample stream for continuation.
        state = optimizer.state_dict()
        for values in state['state'].values():
            for name, value in values.items():
                values[name] = value.cpu()
        pilots.append((student.cpu().state_dict(), state, scheduler.state_dict(), generator.get_state(),
                       clip, fit_a3_range(sample), residual))
        logging.info('MR pilot blocks.%d: %d steps, holdout A3 NMSE %.6g', index, pilot_steps, residual)
        del fit, student, optimizer, scheduler, generator, inputs, targets

    budgets = residual_budgets([pilot[-1] for pilot in pilots])
    states, reports = [], []
    for index, (block, budget, pilot) in enumerate(zip(model.blocks, budgets, pilots)):
        weights, optimizer_state, scheduler_state, rng, clip, baseline_grid, residual = pilot
        inputs, targets = teacher_features(model, block.mlp, loader, device)
        train_count = len(inputs) - holdout
        student, optimizer, scheduler, generator = start_fit(block.mlp, 3000, seed + index)
        student.load_state_dict(weights)
        optimizer.load_state_dict(optimizer_state)
        generator.set_state(rng)
        remaining = budget - pilot_steps
        if budget == 3000:
            scheduler.load_state_dict(scheduler_state)
        else:
            # Start at the pilot's current LR and anneal to zero over the allotted steps.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
        train_fit((student, optimizer, scheduler, generator), inputs[:train_count], targets[:train_count],
                  device, remaining, clip)
        student_grid = fit_a3_range(sample_activations(student, inputs[:train_count], device))
        _, before = score_mlp(block.mlp, inputs[train_count:], targets[train_count:], baseline_grid, device)
        after_fp, after_a3 = score_mlp(student, inputs[train_count:], targets[train_count:], student_grid, device)
        report = dict(block=f'blocks.{index}', steps=budget, pilot_steps=pilot_steps, pilot_a3_nmse=residual,
                      clip=clip, relu_fp_nmse=after_fp, gelu_a3_nmse=before, relu_a3_nmse=after_a3)
        reports.append(report)
        states.append({name: value.detach().cpu() for name, value in student.state_dict().items()})
        logging.info('MLP reconstruction: %s', json.dumps(report, allow_nan=False))
    return states, reports


def apply_mlp_states(model, states):
    if len(states) != len(model.blocks):
        raise ValueError('MLP checkpoint block count does not match the model')
    for block, state in zip(model.blocks, states):
        if not isinstance(block.mlp.act, nn.GELU):
            raise ValueError('MLP checkpoint requires the original GELU model')
        block.mlp.load_state_dict(state, strict=True)
        block.mlp.act = nn.ReLU()


@torch.no_grad()
def holdout_logits(model, dataset, holdout, device):
    logits = []
    for start in range(len(dataset) - holdout, len(dataset), 16):
        images = torch.stack([dataset[i][0] for i in range(start, min(start + 16, len(dataset)))])
        logits.append(model(images.to(device)).cpu())
    return torch.cat(logits)


def screen_mlp_reconstruction(model, loader, directory, device, steps, seed, model_sha, hashes, budget='uniform'):
    started = time.perf_counter()
    holdout = 128
    count = len(loader.dataset)
    if count <= holdout or not all(isinstance(b.mlp.act, nn.GELU) for b in model.blocks):
        raise ValueError('MR screening requires a GELU ViT and a separate 128-image holdout')
    original = holdout_logits(model, loader.dataset, holdout, device)
    if budget == 'residual':
        if steps != 3000 or len(model.blocks) != 12:
            raise ValueError('Residual MR requires 12 blocks and a 36,000-step budget')
        states, reports = fit_with_residual_budget(model, loader, device, seed, holdout)
    else:
        states, reports = [], []
        for index, block in enumerate(model.blocks):
            inputs, targets = teacher_features(model, block.mlp, loader, device)
            student, report = fit_mlp(block.mlp, inputs, targets, device, steps, seed + index, holdout)
            report.update(block=f'blocks.{index}', steps=steps)
            reports.append(report)
            states.append({name: value.detach().cpu() for name, value in student.state_dict().items()})
            logging.info('MLP reconstruction: %s', json.dumps(report, allow_nan=False))
            del inputs, targets, student
    apply_mlp_states(model, states)
    relu = holdout_logits(model, loader.dataset, holdout, device)
    report = dict(seed=seed, fp_model_sha256=model_sha, data_hashes=hashes, steps=steps,
                  budget=budget, total_mr_steps=sum(r['steps'] for r in reports),
                  train_images=count - holdout, holdout_images=holdout, blocks=reports,
                  fp_top1_agreement=float((relu.argmax(-1) == original.argmax(-1)).float().mean()),
                  fp_teacher_kl=float(F.kl_div(relu.log_softmax(-1), original.softmax(-1), reduction='batchmean')),
                  mean_gelu_a3_nmse=sum(r['gelu_a3_nmse'] for r in reports) / len(reports),
                  mean_relu_a3_nmse=sum(r['relu_a3_nmse'] for r in reports) / len(reports))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'relu_mlp.pth'
    temporary = target.with_suffix('.tmp')
    torch.save(states, temporary)
    temporary.replace(target)
    report['checkpoint_sha256'] = file_sha(target)
    report['elapsed_seconds'] = time.perf_counter() - started
    report['status'] = 'complete'
    write_json(directory / 'mr_screen.json', report)
    return report


def load_mlp_reconstruction(model, directory, seed, steps, model_sha, hashes, budget='uniform'):
    directory = Path(directory)
    report = json.loads((directory / 'mr_screen.json').read_text(encoding='utf-8'))
    weights = directory / 'relu_mlp.pth'
    if (report['status'] != 'complete' or report['seed'] != seed or report['steps'] != steps
            or report.get('budget', 'uniform') != budget
            or report.get('total_mr_steps', steps * len(model.blocks)) != steps * len(model.blocks)
            or report['fp_model_sha256'] != model_sha or report['data_hashes'] != hashes
            or report['checkpoint_sha256'] != file_sha(weights)):
        raise RuntimeError('MR screen and quantization recipe fingerprints differ')
    apply_mlp_states(model, torch.load(weights, map_location='cpu', weights_only=True))
    return report['checkpoint_sha256']
