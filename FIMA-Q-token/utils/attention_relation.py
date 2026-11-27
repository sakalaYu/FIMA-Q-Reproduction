"""Task-weighted attention relations inside the SAME DPLR/AdaRound loop.

Teacher query rows are cached during the existing FP cache pass. Task weights
reuse score gradients from existing Fisher refreshes. No extra model pass,
optimizer step, trainable parameter or inference operator is introduced.
"""

from contextlib import contextmanager
import json
import logging
import math

import torch
from torch.nn import functional as F


PROTOCOL = "relation_v1"
ALIGNMENT_PROTOCOL = "relation_v2_alignment"


def query_indices(tokens, rows):
    """Always include CLS and cover patch queries without consuming RNG."""
    if tokens < 1 or rows < 1:
        raise ValueError("Token and query counts must be positive")
    if rows >= tokens:
        return torch.arange(tokens, dtype=torch.long)
    if rows == 1:
        return torch.zeros(1, dtype=torch.long)
    patches = torch.linspace(1, tokens - 1, steps=rows - 1).round().long()
    return torch.cat([torch.zeros(1, dtype=torch.long), patches]).unique(sorted=True)


def weighted_relation_kl(student_scores, teacher_log_prob, weights):
    # Half-precision cache storage only; probability normalization and losses
    # run in FP32. Renormalization avoids treating FP16 rounding as KL error.
    teacher_log = F.log_softmax(teacher_log_prob.detach().float(), dim=-1)
    teacher_prob = teacher_log.exp()
    student_log = F.log_softmax(student_scores.float(), dim=-1)
    per_query = (teacher_prob * (teacher_log - student_log)).sum(-1).clamp_min(0)
    weights = weights.detach().to(per_query.device, dtype=per_query.dtype)
    return (per_query * weights.unsqueeze(0)).mean()


def budgeted_auxiliary(raw, reference, rec_loss, coefficient):
    """Cap loss VALUE at coefficient * reconstruction loss (not gradient norm)."""
    normalized = raw / reference.detach().clamp_min(1e-6)
    budget = rec_loss.detach().clamp_min(0)
    factor = (budget / normalized.detach().clamp_min(1e-8)).clamp(max=1)
    return coefficient * factor * normalized


class AttentionRelation:
    def __init__(self, block, name, options):
        self.name = name
        self.attention = block.attn
        if not hasattr(self.attention, "matmul1"):
            raise ValueError("Attention relations require wrapped timm ViT attention")
        self.options = dict(options)
        if options["mode"] not in ("uniform", "fisher", "aligned"):
            raise ValueError("Attention relation mode must be uniform, fisher or aligned")
        self.protocol = options.get("protocol", PROTOCOL)
        expected = ALIGNMENT_PROTOCOL if options["mode"] == "aligned" else PROTOCOL
        if self.protocol != expected:
            raise ValueError("Attention relation mode and protocol do not match")
        if options["rows"] < 1 or options["weight"] <= 0:
            raise ValueError("Attention rows and weight must be positive")
        if not 0 <= options["start"] < options["full"] <= 1:
            raise ValueError("Require 0 <= relation start < full <= 1")
        self.indices = None
        self.teacher_chunks = []
        self.teacher = None
        self.weights = None
        self.reference = None
        self._student = None
        self._gradient_sum = None
        self._gradient_count = 0
        self._refresh_cursor = 0
        self._refresh_relation_gradient = None
        self.alignment_positive_fraction = None
        self.alignment_mean = None
        self.refreshes = 0
        self.gradient_refreshes = 0
        self.active_steps = 0
        self.first_kl = None
        self.last_kl = None
        self.last_aux = None
        self.last_coefficient = None

    def cache_teacher(self, module, inputs, output):
        if self.indices is None:
            self.indices = query_indices(output.shape[-2], self.options["rows"])
        chosen = self.indices.to(output.device)
        scores = output.detach().index_select(-2, chosen).float() * self.attention.scale
        log_probability = F.log_softmax(scores, -1).to(device="cpu", dtype=torch.float16)
        self.teacher_chunks.append(log_probability)

    def finish_cache(self, count, device, keep_gpu):
        if not self.teacher_chunks:
            raise RuntimeError("No FP attention queries were captured")
        self.teacher = torch.cat(self.teacher_chunks, dim=0)
        self.teacher_chunks.clear()
        if len(self.teacher) != count:
            raise RuntimeError("Attention cache and block input counts differ")
        if keep_gpu:
            self.teacher = self.teacher.to(device)
        self.weights = torch.ones(self.teacher.shape[1:3], device=device, dtype=torch.float32)
        logging.info("Attention relation cache: %s", json.dumps(dict(
            block=self.name, protocol=self.protocol, samples=count,
            query_indices=self.indices.tolist(), cache_bytes=self.teacher.numel() * self.teacher.element_size(),
            mode=self.options["mode"], weight=self.options["weight"])))

    def begin_refresh(self):
        self._gradient_sum = None
        self._gradient_count = 0
        self._refresh_cursor = 0
        self._refresh_relation_gradient = None

    def refresh_score_forward(self, module, inputs, output):
        """Pair this refresh batch with cached teacher relations in loader order."""
        batch = output.shape[0]
        target = self.teacher[self._refresh_cursor:self._refresh_cursor + batch]
        if target.shape[0] != batch:
            raise RuntimeError("Fisher refresh exceeded the attention teacher cache")
        selected = output.detach().index_select(-2, self.indices.to(output.device)).float()
        student_probability = F.softmax(selected * self.attention.scale, dim=-1)
        teacher_probability = F.softmax(target.to(output.device).float(), dim=-1)
        self._refresh_relation_gradient = student_probability - teacher_probability
        self._refresh_cursor += batch

    def score_gradient(self, module, grad_input, grad_output):
        gradient = grad_output[0]
        if gradient is None:
            return
        chosen = gradient.detach().index_select(-2, self.indices.to(gradient.device)).float()
        if self.options["mode"] == "aligned":
            relation_gradient = self._refresh_relation_gradient
            if relation_gradient is None or relation_gradient.shape != chosen.shape:
                raise RuntimeError("Missing matched attention relation during Fisher refresh")
            numerator = (chosen * relation_gradient).sum(dim=-1)
            denominator = chosen.norm(dim=-1) * relation_gradient.norm(dim=-1)
            cosine = torch.where(denominator > 1e-12,
                                 numerator / denominator.clamp_min(1e-12),
                                 torch.zeros_like(numerator)).clamp(-1, 1)
            contribution = cosine.sum(dim=0)
            self._refresh_relation_gradient = None
        else:
            # Undo batchmean scaling before forming a gradient-squared proxy.
            chosen = chosen * gradient.shape[0]
            contribution = chosen.square().sum(dim=(0, 3))
        self._gradient_sum = contribution if self._gradient_sum is None else self._gradient_sum + contribution
        self._gradient_count += gradient.shape[0]

    def end_refresh(self):
        self.refreshes += 1
        self.reference = None
        if self.options["mode"] == "uniform":
            return
        if self._gradient_sum is None or not self._gradient_count:
            raise RuntimeError("No task gradients were captured during Fisher refresh")
        importance = self._gradient_sum / self._gradient_count
        if not torch.isfinite(importance).all():
            raise RuntimeError("Non-finite attention task-importance weights")
        if self.options["mode"] == "aligned":
            if self._refresh_cursor != len(self.teacher) or self._refresh_relation_gradient is not None:
                raise RuntimeError("Fisher refresh and attention target order differ")
            # Positive cosine means descending relation KL also descends the
            # classification KL in score space. Keep the mask bounded by 2.
            mask = (importance > 0).float()
            fraction = float(mask.mean())
            self.weights = (mask / max(0.5, fraction)).detach()
            self.alignment_positive_fraction = fraction
            self.alignment_mean = float(importance.mean())
        elif float(importance.mean()) > 0:
            relative = (importance / importance.mean()).clamp(max=4)
            weight = 0.5 + 0.5 * relative
            self.weights = (weight / weight.mean()).detach()
        else:
            self.weights.fill_(1)
        self.gradient_refreshes += 1
        self._gradient_sum = None
        logging.info("Attention relation weights: %s", json.dumps(dict(
            block=self.name, refresh=self.refreshes, min=float(self.weights.min()),
            max=float(self.weights.max()), mean=float(self.weights.mean()),
            alignment_positive_fraction=self.alignment_positive_fraction,
            alignment_mean=self.alignment_mean)))

    def coefficient(self, iteration, iterations):
        progress = iteration / max(1, iterations)
        fraction = (progress - self.options["start"]) / (self.options["full"] - self.options["start"])
        return self.options["weight"] * min(1.0, max(0.0, fraction))

    @contextmanager
    def capture_student(self, enabled):
        handle = None
        if enabled:
            def capture(module, inputs, output):
                self._student = output.index_select(-2, self.indices.to(output.device)) * self.attention.scale
            handle = self.attention.matmul1.register_forward_hook(capture)
        try:
            yield
        finally:
            if handle is not None:
                handle.remove()
            self._student = None

    def loss(self, sample_indices, rec_loss, coefficient, iteration):
        if self._student is None:
            raise RuntimeError("Student attention logits were not captured")
        indices = sample_indices.to(self.teacher.device)
        teacher = self.teacher.index_select(0, indices).to(self._student.device)
        raw = weighted_relation_kl(self._student, teacher, self.weights)
        if self.reference is None:
            self.reference = raw.detach().clamp_min(1e-6)
        auxiliary = budgeted_auxiliary(raw, self.reference, rec_loss, coefficient)
        if not torch.isfinite(auxiliary):
            raise RuntimeError("Non-finite attention relation loss")
        self.active_steps += 1
        value = float(raw.detach())
        if self.first_kl is None:
            self.first_kl = value
        self.last_kl = value
        self.last_aux = float(auxiliary.detach())
        self.last_coefficient = coefficient
        if iteration % 500 == 0:
            logging.info("Attention relation loss: %s", json.dumps(dict(
                block=self.name, iteration=iteration, coefficient=coefficient,
                raw_kl=value, auxiliary=self.last_aux, rec_loss=float(rec_loss))))
        return auxiliary

    def report(self):
        return dict(block=self.name, protocol=self.protocol, mode=self.options["mode"],
                    weight=self.options["weight"], query_indices=self.indices.tolist(),
                    teacher_samples=len(self.teacher), cache_bytes=self.teacher.numel() * self.teacher.element_size(),
                    refreshes=self.refreshes, gradient_refreshes=self.gradient_refreshes,
                    active_steps=self.active_steps, first_training_kl=self.first_kl,
                    last_training_kl=self.last_kl, last_auxiliary=self.last_aux,
                    weight_min=float(self.weights.min()), weight_max=float(self.weights.max()),
                    alignment_positive_fraction=self.alignment_positive_fraction,
                    alignment_mean=self.alignment_mean)

    def close(self):
        self._student = self.teacher = self.weights = self.reference = None
        self._gradient_sum = None
        self.teacher_chunks.clear()
