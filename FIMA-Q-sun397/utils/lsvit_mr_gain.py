"""Allocate MR updates by measured logit improvement in a shared quantized model."""

import copy
import json
import logging
import math

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from utils import lsvit_mr as mr
from utils.lsvit_pair import fingerprint


EARLY_STEP = 2000
PILOT_STEP = 2500
MAX_STEPS = 3500
PROBE_IMAGES = 32
PROBE_BATCH = 8


def gain_budgets(scores):
    if len(scores) != 12 or not all(math.isfinite(score) for score in scores):
        raise ValueError('MR gain allocation requires 12 finite scores')
    order = sorted(range(12), key=lambda i: (scores[i], i))
    budgets = [3000] * 12
    for i in order[:4]:
        budgets[i] = PILOT_STEP
    for i in order[-4:]:
        budgets[i] = MAX_STEPS
    return budgets


def cpu_weights(student):
    return {name: value.detach().to(device='cpu', copy=True) for name, value in student.state_dict().items()}


@torch.no_grad()
def probe_logit_gains(teacher, loader, device, cfg, early, late, teacher_logits):
    from utils.calibrator import QuantCalibrator
    from utils.wrap_net import wrap_modules_in_net

    cfg = copy.copy(cfg)
    cfg.calib_batch_size = PROBE_BATCH
    cfg.search_round = 1
    cfg.eq_n = 16
    dataset = loader.dataset
    fit_loader = DataLoader(Subset(dataset, range(PROBE_IMAGES)), batch_size=PROBE_BATCH)
    probe = copy.deepcopy(teacher)
    mr.apply_mlp_states(probe, late)
    probe = wrap_modules_in_net(probe, cfg, reparam=False).to(device).eval()
    QuantCalibrator(probe, fit_loader).batching_quant_calib()
    teacher_prob = teacher_logits.softmax(-1)
    observations = []
    for index, block in enumerate(probe.blocks):
        shared = block.mlp
        inputs = []
        handle = shared.register_forward_pre_hook(lambda _, args: inputs.append(args[0].detach().cpu()))
        try:
            for images, _ in fit_loader:
                probe(images.to(device))
        finally:
            handle.remove()
        candidate_inputs = torch.cat(inputs)
        candidate_loader = DataLoader(TensorDataset(candidate_inputs, torch.zeros(len(candidate_inputs))),
                                      batch_size=PROBE_BATCH)
        divergence = []
        try:
            for state in (early[index], late[index]):
                candidate = copy.deepcopy(teacher.blocks[index].mlp)
                candidate.load_state_dict(state)
                candidate.act = torch.nn.ReLU()
                candidate = wrap_modules_in_net(candidate, cfg, reparam=False).to(device).eval()
                QuantCalibrator(candidate, candidate_loader).batching_quant_calib()
                block.mlp = candidate
                logits = mr.holdout_logits(probe, dataset, len(teacher_logits), device, batch_size=PROBE_BATCH)
                divergence.append(float(F.kl_div(logits.log_softmax(-1), teacher_prob, reduction='batchmean')))
        finally:
            block.mlp = shared
        if not all(math.isfinite(value) for value in divergence):
            raise RuntimeError(f'Non-finite MR gain probe in blocks.{index}')
        observations.append(tuple(divergence))
        logging.info('MR gain probe blocks.%d: KL %.7f -> %.7f, gain %+.7f', index,
                     divergence[0], divergence[1], divergence[0] - divergence[1])
    return observations


def fit_with_gain_budget(model, loader, device, seed, holdout, cfg, teacher_logits, budget_policy):
    if cfg is None or (cfg.w_bit, cfg.a_bit) != (3, 3) or len(model.blocks) != 12:
        raise ValueError('MR gain allocation requires the ViT-S W3A3 configuration')
    early, late, pilots = [], [], []
    for index, block in enumerate(model.blocks):
        teacher = block.mlp
        inputs, targets = mr.teacher_features(model, teacher, loader, device)
        n = len(inputs) - holdout
        sample = mr.sample_activations(teacher, inputs[:n], device)
        positive = sample[sample > 0]
        clip = float(torch.quantile(positive, 0.99)) if len(positive) else 0.
        fit = mr.start_fit(teacher, MAX_STEPS, seed + index)
        student, optimizer, scheduler, generator = fit
        local_errors = []
        for step, saved, updates in ((EARLY_STEP, early, EARLY_STEP),
                                     (PILOT_STEP, late, PILOT_STEP - EARLY_STEP)):
            mr.train_fit(fit, inputs[:n], targets[:n], device, updates, clip)
            grid = mr.fit_a3_range(mr.sample_activations(student, inputs[:n], device))
            _, error = mr.score_mlp(student, inputs[n:], targets[n:], grid, device)
            local_errors.append(error)
            saved.append(cpu_weights(student))
            logging.info('MR pilot blocks.%d, step %d: holdout A3 NMSE %.7f', index, step, error)
        moments = optimizer.state_dict()
        for values in moments['state'].values():
            for name, value in values.items():
                values[name] = value.detach().cpu()
        pilots.append((moments, scheduler.state_dict(), generator.get_state(),
                       clip, mr.fit_a3_range(sample), local_errors))
        del fit, student, optimizer, scheduler, generator, inputs, targets

    # Include the weights, optimizer, LR schedule and sample stream in the pairing check.
    pilot_sha = fingerprint((early, late, pilots))
    scores = None
    budgets = [3000] * 12
    if budget_policy == 'gain':
        probe_device = torch.device(device)
        cuda_devices = [probe_device] if probe_device.type == 'cuda' else []
        with torch.random.fork_rng(devices=cuda_devices):
            scores = probe_logit_gains(model, loader, device, cfg, early, late, teacher_logits)
        budgets = gain_budgets([before - after for before, after in scores])
    del early
    logging.info('MR %s steps: %s (total=%d)', budget_policy, budgets, sum(budgets))

    states, reports = [], []
    for index, (block, budget, pilot) in enumerate(zip(model.blocks, budgets, pilots)):
        moments, schedule, rng, clip, baseline_grid, local_errors = pilot
        inputs, targets = mr.teacher_features(model, block.mlp, loader, device)
        n = len(inputs) - holdout
        fit = mr.start_fit(block.mlp, MAX_STEPS, seed + index)
        student, optimizer, scheduler, generator = fit
        student.load_state_dict(late[index])
        optimizer.load_state_dict(moments)
        scheduler.load_state_dict(schedule)
        generator.set_state(rng)
        mr.train_fit(fit, inputs[:n], targets[:n], device, budget - PILOT_STEP, clip)
        grid = mr.fit_a3_range(mr.sample_activations(student, inputs[:n], device))
        _, baseline = mr.score_mlp(block.mlp, inputs[n:], targets[n:], baseline_grid, device)
        after_fp, after_a3 = mr.score_mlp(student, inputs[n:], targets[n:], grid, device)
        report = dict(block=f'blocks.{index}', steps=budget, clip=clip, gelu_a3_nmse=baseline,
                      relu_fp_nmse=after_fp, relu_a3_nmse=after_a3,
                      early_a3_nmse=local_errors[0], pilot_a3_nmse=local_errors[1])
        if scores is not None:
            report['early_teacher_kl'], report['pilot_teacher_kl'] = scores[index]
            report['teacher_kl_gain'] = scores[index][0] - scores[index][1]
        logging.info('MLP reconstruction: %s', json.dumps(report, allow_nan=False))
        reports.append(report)
        states.append(cpu_weights(student))
        del fit, student, optimizer, scheduler, generator, inputs, targets

    info = dict(pilot_sha256=pilot_sha, early_step=EARLY_STEP, pilot_step=PILOT_STEP,
                cosine_steps=MAX_STEPS, probe_images=PROBE_IMAGES if scores is not None else 0)
    return states, reports, info
