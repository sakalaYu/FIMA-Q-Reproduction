"""Frozen semantic probes for label-free, block-wise PTQ distillation."""

import torch
from torch.nn import functional as F


PROTOCOL = 'semantic_probe_v1'


def sample_kl(logits, target, temperature):
    return F.kl_div(F.log_softmax(logits / temperature, -1), target,
                    reduction='none').sum(-1).clamp_min(0) * temperature ** 2


class SemanticProbeKD:
    def __init__(self, features, teacher_softmax, teacher_temperature, options, device):
        self.options = dict(options)
        self.temperature = options['temperature']
        n, width = features.shape
        holdout = options['holdout']
        if not 1 <= holdout < n or teacher_softmax.shape[0] != n:
            raise ValueError('Probe features and teacher predictions must share samples and a valid holdout')
        train = n - holdout
        # No sampling/optimizer and no RNG changes: pairs retain their original
        # QDrop/AdaRound random stream until the KD gradient is introduced.
        with torch.no_grad():
            x = features.detach().cpu().double()
            logits = teacher_softmax.detach().cpu().double().clamp_min(1e-30).log()
            logits = (logits - logits.mean(-1, keepdim=True)) * teacher_temperature
            mean = x[:train].mean(0)
            std = x[:train].std(0, unbiased=False).clamp_min(1e-5)
            normalized = (x - mean) / std
            bias = logits[:train].mean(0)
            gram = normalized[:train].T @ normalized[:train] / train
            ridge = options['ridge'] * gram.diagonal().mean().clamp_min(1e-12)
            weight = torch.linalg.solve(gram + ridge * torch.eye(width, dtype=x.dtype),
                                        normalized[:train].T @ (logits[:train] - bias) / train)
            predicted = normalized @ weight + bias
            full_prob = (logits / self.temperature).softmax(-1)
            prior = full_prob[:train].mean(0).clamp_min(1e-30)
            prior_loss = F.kl_div(prior.log().expand(holdout, -1), full_prob[train:],
                                  reduction='batchmean') * self.temperature ** 2
            probe_loss = sample_kl(predicted[train:], full_prob[train:], self.temperature).mean()
            # Skill over a constant teacher prior, evaluated ONLY on held-out
            # samples. A probe with no useful predictive skill contributes zero.
            skill = float((1 - probe_loss / prior_loss.clamp_min(1e-12)).clamp(0, 1))
            self.reliability = skill if float(prior_loss) > 1e-10 else 0.0
            self.gain = self.reliability if options['mode'] == 'reliable' else 1.0
            self.mean = mean.float().to(device)
            self.std = std.float().to(device)
            self.weight = weight.float().to(device)
            self.bias = bias.float().to(device)
            # Distil the SAME frozen probe on FP vs quantized block features.
            # This makes KD exactly zero at identical FP/student features,
            # avoiding forcing the student to compensate for probe fitting bias.
            self.targets = (self.project(features.detach().float().to(device)) /
                            self.temperature).softmax(-1).detach()
            self.metadata = dict(protocol=PROTOCOL, mode=options['mode'], train_samples=train,
                                 holdout_samples=holdout, feature_dim=width,
                                 teacher_prior_kl=float(prior_loss), teacher_probe_kl=float(probe_loss),
                                 reliability=self.reliability, effective_gain=self.gain,
                                 temperature=self.temperature, weight=options['weight'])
        self.normalizer = None
        self.active_steps = 0
        self.last_raw = self.last_auxiliary = 0.0

    def project(self, features):
        return ((features - self.mean) / self.std) @ self.weight + self.bias

    def loss(self, student_features, indices, step, total_steps):
        progress = step / total_steps
        ramp = min(1.0, max(0.0, (progress - 0.2) / 0.2))
        if ramp == 0 or self.gain == 0:
            return student_features.new_zeros(())
        target = self.targets[indices.to(self.targets.device)]
        raw = sample_kl(self.project(student_features.float()), target, self.temperature).mean()
        if self.normalizer is None:
            self.normalizer = raw.detach().clamp_min(1e-4)
        auxiliary = self.options['weight'] * self.gain * ramp * raw / self.normalizer
        self.active_steps += 1
        self.last_raw = float(raw.detach())
        self.last_auxiliary = float(auxiliary.detach())
        return auxiliary

    def report(self):
        return dict(self.metadata, active_steps=self.active_steps, last_raw_kl=self.last_raw,
                    initial_kd_normalizer=float(self.normalizer) if self.normalizer is not None else None,
                    last_auxiliary=self.last_auxiliary)
