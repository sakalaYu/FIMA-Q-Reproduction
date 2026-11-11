"""Fit ReLU MLPs before LS-ViT quantization and screen their A3 activation error."""

from contextlib import contextmanager
import copy
import json
import logging
from pathlib import Path

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


def fit_mlp(teacher, inputs, targets, device, steps, seed, holdout):
    train_count = len(inputs) - holdout
    sample = sample_activations(teacher, inputs[:train_count], device)
    positive = sample[sample > 0]
    clip = float(torch.quantile(positive, 0.99)) if len(positive) else 0.
    baseline_grid = fit_a3_range(sample)
    student = copy.deepcopy(teacher)
    student.act = nn.ReLU()
    student.requires_grad_(False)
    params = list(student.fc1.parameters()) + list(student.fc2.parameters())
    for parameter in params:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(params, lr=4e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    generator = torch.Generator().manual_seed(seed)
    with capture_fc2_input(student) as captured:
        for _ in range(steps):
            indices = torch.randint(train_count, (4,), generator=generator)
            x, target = inputs[indices].to(device), targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = student(x)
            hidden = captured.pop().clamp_max(clip)
            clipped = F.linear(hidden, student.fc2.weight, student.fc2.bias)
            loss = F.mse_loss(out, target) + 2 * F.mse_loss(clipped, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
    student_grid = fit_a3_range(sample_activations(student, inputs[:train_count], device))
    _, before_a3 = score_mlp(teacher, inputs[train_count:], targets[train_count:], baseline_grid, device)
    after_fp, after_a3 = score_mlp(student, inputs[train_count:], targets[train_count:], student_grid, device)
    return student, dict(clip=clip, relu_fp_nmse=after_fp, gelu_a3_nmse=before_a3, relu_a3_nmse=after_a3)


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


def screen_mlp_reconstruction(model, loader, directory, device, steps, seed, model_sha, hashes):
    holdout = 128
    count = len(loader.dataset)
    if count <= holdout or not all(isinstance(b.mlp.act, nn.GELU) for b in model.blocks):
        raise ValueError('MR screening requires a GELU ViT and a separate 128-image holdout')
    original = holdout_logits(model, loader.dataset, holdout, device)
    states, reports = [], []
    for index, block in enumerate(model.blocks):
        inputs, targets = teacher_features(model, block.mlp, loader, device)
        student, report = fit_mlp(block.mlp, inputs, targets, device, steps, seed + index, holdout)
        report['block'] = f'blocks.{index}'
        reports.append(report)
        states.append({name: value.detach().cpu() for name, value in student.state_dict().items()})
        logging.info('MLP reconstruction: %s', json.dumps(report, allow_nan=False))
        del inputs, targets, student
    apply_mlp_states(model, states)
    relu = holdout_logits(model, loader.dataset, holdout, device)
    report = dict(seed=seed, fp_model_sha256=model_sha, data_hashes=hashes, steps=steps,
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
    report['status'] = 'complete'
    write_json(directory / 'mr_screen.json', report)
    return report


def load_mlp_reconstruction(model, directory, seed, steps, model_sha, hashes):
    directory = Path(directory)
    report = json.loads((directory / 'mr_screen.json').read_text(encoding='utf-8'))
    weights = directory / 'relu_mlp.pth'
    if (report['status'] != 'complete' or report['seed'] != seed or report['steps'] != steps
            or report['fp_model_sha256'] != model_sha or report['data_hashes'] != hashes
            or report['checkpoint_sha256'] != file_sha(weights)):
        raise RuntimeError('MR screen and quantization recipe fingerprints differ')
    apply_mlp_states(model, torch.load(weights, map_location='cpu', weights_only=True))
    return report['checkpoint_sha256']
