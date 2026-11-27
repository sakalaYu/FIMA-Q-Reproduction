"""Label-free Q/K calibration before AdaRound, with foldable transforms.

A: joint attention-aware Q/K scale factors.
B: A + bounded, reciprocal Q/K channel scaling.
C: B + per-head Cayley rotations.
The block objective can be selected independently of the transform variant.

This module trains a temporary copy of ONE block. Only ordinary QKV weights,
biases and quantizer state are committed; no inference module is introduced.
"""

import copy
import json
import logging
import math
import time
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


QK_JOINT_PROTOCOL = "v2_integer_zp"
LOW_RANK_PROTOCOL = "v3_lowrank_gate"


def round_ste(x):
    return x + (x.round() - x).detach()


def fake_quant(x, scale, zero_point, quantizer):
    scale = scale.clamp_min(1e-12)
    integer = round_ste(x / scale)
    if quantizer.sym:
        return integer.clamp(-quantizer.n_levels, quantizer.n_levels - 1) * scale
    zero = zero_point.detach().round()
    return ((integer + zero).clamp(0, 2 * quantizer.n_levels - 1) - zero) * scale


class QKParameters(nn.Module):
    """An additional R,D relative to the already Hadamard-rotated weights."""

    def __init__(self, attention, variant):
        super().__init__()
        if variant not in ("A", "B", "C"):
            raise ValueError("Q/K variant must be A, B or C")
        self.variant = variant
        self.heads = int(attention.num_heads)
        qkv = attention.qkv
        self.width = qkv.in_features
        self.dim = self.width // self.heads
        if qkv.out_features != 3 * self.width or self.width % self.heads:
            raise ValueError("Expected fused ViT QKV")
        if qkv.w_quantizer.sym:
            raise ValueError("This implementation expects the asymmetric FIMA-Q quantizer")
        for name in ("q_norm", "k_norm"):
            norm = getattr(attention, name, None)
            if norm is not None and not isinstance(norm, nn.Identity):
                raise ValueError("Q/K transforms require identity q_norm and k_norm")
        self.register_buffer("weight", qkv.weight.detach().clone())
        self.register_buffer("bias", qkv.bias.detach().clone() if qkv.bias is not None else None)
        self.register_buffer("weight_scale", qkv.w_quantizer.scale.detach().clone())
        self.register_buffer("weight_zero", qkv.w_quantizer.zero_point.detach().clone())
        self.register_buffer("q_scale", attention.matmul1.A_quantizer.scale.detach().clone())
        self.register_buffer("k_scale", attention.matmul1.B_quantizer.scale.detach().clone())
        self.register_buffer("q_zero", attention.matmul1.A_quantizer.zero_point.detach().clone())
        self.register_buffer("k_zero", attention.matmul1.B_quantizer.zero_point.detach().clone())
        self.register_buffer("eye", torch.eye(self.dim, device=self.weight.device, dtype=self.weight.dtype))
        self.log_q_scale = nn.Parameter(self.weight.new_zeros(1, self.heads, 1, 1))
        self.log_k_scale = nn.Parameter(self.weight.new_zeros(1, self.heads, 1, 1))
        self.log_diag = nn.Parameter(self.weight.new_zeros(self.heads, self.dim), requires_grad=variant != "A")
        self.skew = nn.Parameter(self.weight.new_zeros(self.heads, self.dim, self.dim), requires_grad=variant == "C")

    def rotation(self):
        if self.variant != "C":
            return self.eye.expand(self.heads, -1, -1)
        skew = self.skew - self.skew.transpose(-1, -2)
        return torch.linalg.solve(self.eye - 0.5 * skew, self.eye + 0.5 * skew)

    def transformed(self):
        if self.variant == "A":
            return self.weight, self.bias, self.weight_scale, self.weight_zero
        rotation = self.rotation()
        diagonal = self.log_diag.exp()
        parts, biases = [], []
        for branch, factor in ((0, diagonal), (1, diagonal.reciprocal())):
            start = branch * self.width
            value = self.weight[start:start + self.width].reshape(self.heads, self.dim, self.width)
            parts.append((rotation.transpose(-1, -2) @ value * factor.unsqueeze(-1)).reshape(self.width, self.width))
            if self.bias is not None:
                value = self.bias[start:start + self.width].reshape(self.heads, self.dim, 1)
                biases.append((rotation.transpose(-1, -2) @ value * factor.unsqueeze(-1)).reshape(self.width))
        weight = torch.cat(parts + [self.weight[2 * self.width:]], dim=0)
        bias = torch.cat(biases + [self.bias[2 * self.width:]], dim=0) if self.bias is not None else None
        # Preserve each calibrated row's relative clipping endpoints while its
        # basis/range changes. The very same scales are used after folding.
        old = self.weight.reshape_as(weight).view(3, self.width, self.width)
        new = weight.view(3, self.width, self.width)
        old_min = old.amin(-1, keepdim=True)
        old_range = (old.amax(-1, keepdim=True) - old_min).clamp_min(1e-12)
        new_min = new.amin(-1, keepdim=True)
        new_range = (new.amax(-1, keepdim=True) - new_min).clamp_min(1e-12)
        scale = self.weight_scale * (new_range / old_range)
        old_lower = -self.weight_zero.round() * self.weight_scale
        lower = new_min + ((old_lower - old_min) / old_range) * new_range
        zero = -lower / scale.clamp_min(1e-12)
        # V is deliberately unchanged, including its calibrated quantizer state.
        scale = torch.cat([scale[:2], self.weight_scale[2:]], dim=0)
        zero = torch.cat([zero[:2], self.weight_zero[2:]], dim=0)
        return weight, bias, scale, zero

    def project(self):
        with torch.no_grad():
            self.log_q_scale.clamp_(-math.log(2), math.log(2))
            self.log_k_scale.clamp_(-math.log(2), math.log(2))
            self.log_diag.clamp_(-math.log(2), math.log(2))
            self.skew.clamp_(-0.1, 0.1)

    def penalty(self):
        return 0.01 * self.log_diag.square().mean() + 0.001 * self.skew.square().mean()

    def fold_into(self, attention):
        with torch.no_grad():
            weight, bias, scale, zero = self.transformed()
            attention.qkv.weight.copy_(weight)
            if bias is not None:
                attention.qkv.bias.copy_(bias)
            attention.qkv.w_quantizer.scale.copy_(scale)
            if self.variant != "A":
                # UniformQuantizer rounds zero points in forward; AdaRound uses
                # the stored value directly. Commit the integer zero point used
                # by calibration so entering AdaRound cannot change its grid.
                zero = zero.clone()
                zero[:2].round_()
            attention.qkv.w_quantizer.zero_point.copy_(zero)
            attention.matmul1.A_quantizer.scale.copy_(self.q_scale * self.log_q_scale.exp())
            attention.matmul1.B_quantizer.scale.copy_(self.k_scale * self.log_k_scale.exp())


class LowRankQKParameters(QKParameters):
    """A rank-bounded Cayley tangent update on top of the fixed Hadamard."""

    def __init__(self, attention, rank):
        super().__init__(attention, "C")
        if not 1 <= rank <= self.dim // 2:
            raise ValueError("Rotation rank must be between 1 and half the head dimension")
        del self.skew
        basis = torch.randn(self.heads, self.dim, rank, device=self.weight.device,
                            dtype=self.weight.dtype)
        basis = torch.linalg.qr(basis).Q
        self.rot_a = nn.Parameter(basis)
        self.rot_b = nn.Parameter(torch.zeros_like(basis))

    def tangent(self):
        product = self.rot_a @ self.rot_b.transpose(-1, -2)
        return product - product.transpose(-1, -2)

    def rotation(self):
        skew = self.tangent()
        return torch.linalg.solve(self.eye - 0.5 * skew, self.eye + 0.5 * skew)

    def project(self):
        with torch.no_grad():
            self.log_q_scale.clamp_(-math.log(2), math.log(2))
            self.log_k_scale.clamp_(-math.log(2), math.log(2))
            self.log_diag.clamp_(-math.log(2), math.log(2))
            # Bound the actual rotation generator, not its non-unique factors.
            norms = self.tangent().norm(dim=(-2, -1)).clamp_min(1e-12)
            self.rot_b.mul_((2.0 / norms).clamp(max=1.0)[:, None, None])

    def penalty(self):
        return 0.01 * self.log_diag.square().mean() + 0.001 * self.tangent().square().mean()


def _quant_weight_bias(qkv):
    fit = qkv._qk_fit
    weight, bias, scale, zero = fit.transformed()
    return fake_quant(weight.view(3, fit.width, fit.width), scale, zero,
                      qkv.w_quantizer).reshape_as(weight), bias


def _quant_q(matmul, value):
    fit = matmul.__dict__["_qk_fit_ref"]
    return fake_quant(value, fit.q_scale * fit.log_q_scale.exp(), fit.q_zero, matmul.A_quantizer)


def _quant_k(matmul, value):
    fit = matmul.__dict__["_qk_fit_ref"]
    return fake_quant(value, fit.k_scale * fit.log_k_scale.exp(), fit.k_zero, matmul.B_quantizer)


def make_work_block(block, variant, rotation_rank=0):
    # Avoid copying the hundreds of MB of reconstruction caches with the block.
    names = ("raw_input", "raw_out", "raw_grad", "quanted_input", "quanted_out",
             "tmp_input", "tmp_out", "tmp_grad", "delta_out", "inverse_B")
    saved = {name: getattr(block, name) for name in names if hasattr(block, name)}
    try:
        for name in saved:
            setattr(block, name, None)
        work = copy.deepcopy(block)
    finally:
        for name, value in saved.items():
            setattr(block, name, value)
    work.eval()
    for parameter in work.parameters():
        parameter.requires_grad_(False)
    for module in work.modules():
        if hasattr(module, "mode"):
            module.mode = "quant_forward"
        if hasattr(module, "training_mode"):
            module.training_mode = True
        if hasattr(module, "drop_prob"):
            module.drop_prob = 1.0
    fit = (LowRankQKParameters(work.attn, rotation_rank)
           if variant == "C" and rotation_rank else QKParameters(work.attn, variant))
    work.attn.qkv.add_module("_qk_fit", fit)
    work.attn.qkv.quant_weight_bias = MethodType(_quant_weight_bias, work.attn.qkv)
    work.attn.matmul1.__dict__["_qk_fit_ref"] = fit
    work.attn.matmul1.quant_input_A = MethodType(_quant_q, work.attn.matmul1)
    work.attn.matmul1.quant_input_B = MethodType(_quant_k, work.attn.matmul1)
    return work, fit


def teacher_attention(block, x):
    attention = block.attn
    value = F.linear(block.norm1(x), attention.qkv.weight, attention.qkv.bias)
    batch, tokens, width3 = value.shape
    q, k, _ = value.reshape(batch, tokens, 3, attention.num_heads,
                            width3 // (3 * attention.num_heads)).permute(2, 0, 3, 1, 4)
    q = attention.q_norm(q) if hasattr(attention, "q_norm") else q
    k = attention.k_norm(k) if hasattr(attention, "k_norm") else k
    return ((q @ k.transpose(-1, -2)) * attention.scale).softmax(-1)


def attention_kl(scores, target):
    return F.kl_div(F.log_softmax(scores, dim=-1), target, reduction="none").sum(-1).mean()


def reconstruction_objective(variant, requested="auto"):
    """Separate learned-rotation capacity from its pre-calibration objective."""
    if requested not in ("auto", "mse", "dplr"):
        raise ValueError("Q/K reconstruction loss must be auto, mse or dplr")
    return ("dplr" if variant == "C" else "mse") if requested == "auto" else requested


def accept_qk_candidate(before_kl, best_kl, min_relative_kl):
    relative_gain = (before_kl - best_kl) / max(before_kl, 1e-12)
    return best_kl < before_kl - 1e-8 and relative_gain >= min_relative_kl, relative_gain


def dplr_terms(pred, target, gradient, inverse):
    error = (pred - target).abs().flatten(1)
    projected = error @ gradient.t()
    low_rank = ((projected @ inverse) * projected).sum(-1).mean().clamp_min(0)
    diagonal = (error.square() * gradient.mean(0)).mean()
    return low_rank, diagonal


def _module_state(model):
    return [(module, {key: getattr(module, key) for key in ("mode", "training_mode", "drop_prob")
                      if hasattr(module, key)}) for module in model.modules()]


def _restore_state(state):
    for module, values in state:
        for name, value in values.items():
            setattr(module, name, value)


def _images(reconstructor, indices, device):
    dataset = reconstructor.calib_loader.dataset
    return torch.stack([dataset[int(index)][0] for index in indices]).to(device)


def _prefix_modes(reconstructor, block):
    for name, item in reconstructor.blocks.items():
        reconstructor.set_block_mode(item, "quant_forward" if name in reconstructor.quanted_blocks
                                     or item is block else "raw")


def estimate_dplr(reconstructor, block, train_count, batch_size, device):
    """FIMA-style gradient/perturbation anchors, using ONLY optimization samples.

    The ridge-stabilized Gram inverse is local to this calibration stage and
    never changes the anchors or rank of the subsequent original reconstruction.
    """
    state = _module_state(reconstructor.model)
    gradients, directions = [], []
    teacher = torch.cat(reconstructor.raw_pred_softmaxs, dim=0).detach()
    captured = {}

    def capture(module, inputs, output):
        captured["output"] = output.detach()
        output.register_hook(lambda grad: captured.__setitem__("gradient", grad.detach()))

    hook = block.register_forward_hook(capture)
    try:
        _prefix_modes(reconstructor, block)
        for module in block.modules():
            if hasattr(module, "training_mode"):
                module.training_mode = True
            if hasattr(module, "drop_prob"):
                module.drop_prob = 0.5
        for anchor in range(reconstructor.k):
            total_grad = total_delta = None
            for start in range(0, train_count, batch_size):
                stop = min(start + batch_size, train_count)
                reconstructor.model.zero_grad(set_to_none=True)
                pred = reconstructor.model(_images(reconstructor, range(start, stop), device))
                loss = F.kl_div(F.log_softmax(pred / reconstructor.temperature, -1),
                                teacher[start:stop].to(device), reduction="batchmean")
                loss.backward()
                # Remove batchmean scaling before aggregating samples.
                grad = captured.pop("gradient").abs().sum(0) * (stop - start)
                delta = (captured.pop("output") - block.raw_out[start:stop].to(device)).abs().sum(0)
                total_grad = grad if total_grad is None else total_grad + grad
                total_delta = delta if total_delta is None else total_delta + delta
            gradients.append((total_grad / train_count).flatten())
            directions.append((total_delta / train_count).flatten())
            logging.info("QK joint DPLR anchor: %d/%d", anchor + 1, reconstructor.k)
        gradient = torch.stack(gradients).detach()
        direction = torch.stack(directions).detach()
        gram = direction.double() @ direction.double().t()
        ridge = gram.diagonal().mean().clamp_min(1e-20) * 1e-6
        inverse = torch.linalg.inv(gram + ridge * torch.eye(len(gram), device=device,
                                                           dtype=gram.dtype)).to(gradient.dtype)
        return gradient, inverse
    finally:
        hook.remove()
        _restore_state(state)
        reconstructor.model.zero_grad(set_to_none=True)


def full_teacher_kl(reconstructor, block, work, indices, batch_size, device):
    """Evaluate a candidate through the real prefix and suffix, without labels."""
    state = _module_state(reconstructor.model)
    teacher = torch.cat(reconstructor.raw_pred_softmaxs, dim=0).detach()
    # The reconstructor caches softmax(logits/T). Recover probabilities at T=1.
    teacher = (teacher.clamp_min(1e-30).log() * reconstructor.temperature).softmax(-1)
    hook = block.register_forward_hook(lambda module, inputs, output: work(inputs[0]))
    total = 0.0
    try:
        _prefix_modes(reconstructor, block)
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                chosen = indices[start:start + batch_size]
                pred = reconstructor.model(_images(reconstructor, chosen, device))
                target = teacher[chosen].to(device)
                total += float(F.kl_div(F.log_softmax(pred, -1), target, reduction="sum"))
        return max(0.0, total / len(indices))
    finally:
        hook.remove()
        _restore_state(state)


def post_round_teacher_kl(reconstructor, block, holdout, batch_size, device):
    """Measure teacher KL after the actual AdaRound stage on the same images."""
    state = _module_state(reconstructor.model)
    teacher = torch.cat(reconstructor.raw_pred_softmaxs, dim=0).detach()
    teacher = (teacher.clamp_min(1e-30).log() * reconstructor.temperature).softmax(-1)
    indices = list(range(len(teacher) - holdout, len(teacher)))
    total = 0.0
    try:
        _prefix_modes(reconstructor, block)
        with torch.no_grad():
            for start in range(0, holdout, batch_size):
                chosen = indices[start:start + batch_size]
                pred = reconstructor.model(_images(reconstructor, chosen, device))
                total += float(F.kl_div(F.log_softmax(pred, -1),
                                        teacher[chosen].to(device), reduction="sum"))
        return max(0.0, total / holdout)
    finally:
        _restore_state(state)


def _calibrate(reconstructor, name, block, device, options):
    variant = options["variant"]
    rec_objective = reconstruction_objective(variant, options.get("reconstruction_loss", "auto"))
    protocol = options.get("protocol", QK_JOINT_PROTOCOL)
    rotation_rank = options.get("rotation_rank", 0)
    min_relative_kl = options.get("min_relative_kl", 0.0)
    if protocol not in (QK_JOINT_PROTOCOL, LOW_RANK_PROTOCOL):
        raise ValueError("Unknown Q/K calibration protocol")
    if protocol == QK_JOINT_PROTOCOL and (rotation_rank or min_relative_kl):
        raise ValueError("Low-rank parameters require the low-rank protocol")
    if protocol == LOW_RANK_PROTOCOL and (variant != "C" or rotation_rank <= 0):
        raise ValueError("Low-rank protocol requires C and a positive rotation rank")
    if not 0 <= min_relative_kl < 1:
        raise ValueError("Relative KL threshold must be in [0, 1)")
    count = len(block.raw_input)
    holdout = options["holdout"]
    if not 1 <= holdout < count:
        raise ValueError("QK holdout must be positive and smaller than optim_size")
    train_count = count - holdout
    # The initial 128 calibration images are the prefix of the 1024 selected
    # images. Reserve the LAST 128, keeping that prefix in the training portion.
    validation = list(range(train_count, count))
    batch_size = options["batch_size"]
    work, fit = make_work_block(block, variant, rotation_rank=rotation_rank)
    train_parameters = [parameter for parameter in fit.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(train_parameters, lr=options["lr"])
    generator = torch.Generator().manual_seed(options["seed"] + int(name.split(".")[-1]))
    scores = {}
    hook = work.attn.matmul1.register_forward_hook(
        lambda module, inputs, output: scores.__setitem__("value", output * work.attn.scale))
    gradient = inverse = None
    try:
        if rec_objective == "dplr":
            gradient, inverse = estimate_dplr(reconstructor, block, train_count, batch_size, device)

        def terms(indices):
            x = block.raw_input[indices].to(device)
            target = block.raw_out[indices].to(device)
            with torch.no_grad():
                probability = teacher_attention(block, x)
            prediction = work(x)
            kl = attention_kl(scores.pop("value"), probability)
            if gradient is not None:
                rec = dplr_terms(prediction, target, gradient, inverse)
            else:
                rec = ((prediction - target).square().mean(),)
            return (kl,) + rec

        initial = {key: value.detach().clone() for key, value in fit.named_parameters()}

        def restore(parameters):
            with torch.no_grad():
                for key, parameter in fit.named_parameters():
                    parameter.copy_(parameters[key])

        with torch.no_grad():
            normalizers = tuple(value.detach().clamp_min(1e-30)
                                for value in terms(list(range(min(batch_size, train_count)))))

        def objective(values):
            normalized = [value / scale for value, scale in zip(values, normalizers)]
            return normalized[0] + sum(normalized[1:]) / len(normalized[1:])

        def local_score():
            total = 0.0
            with torch.no_grad():
                for start in range(0, holdout, batch_size):
                    chosen = validation[start:start + batch_size]
                    total += float(objective(terms(chosen))) * len(chosen)
            return total / holdout

        initial_local = best_score = local_score()
        selected_local = initial_local
        best = initial
        before_kl = full_teacher_kl(reconstructor, block, work, validation, batch_size, device)
        best_kl = last_candidate_kl = before_kl
        for step in range(options["steps"]):
            indices = torch.randperm(train_count, generator=generator)[:batch_size].tolist()
            optimizer.zero_grad(set_to_none=True)
            loss = objective(terms(indices)) + fit.penalty()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite QK calibration loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_parameters, 1.0)
            optimizer.step()
            fit.project()
            if (step + 1) % options["eval_every"] == 0 or step + 1 == options["steps"]:
                candidate = local_score()
                last_candidate_kl = full_teacher_kl(reconstructor, block, work, validation, batch_size, device)
                logging.info("QK joint %s %s step %d/%d: holdout objective %.6g; "
                             "teacher KL %.6g (best %.6g)", variant, name, step + 1,
                             options["steps"], candidate, last_candidate_kl, best_kl)
                if math.isfinite(candidate) and candidate < best_score:
                    best_score = candidate
                if math.isfinite(last_candidate_kl) and last_candidate_kl < best_kl - 1e-8:
                    best_kl = last_candidate_kl
                    selected_local = candidate
                    best = {key: value.detach().clone() for key, value in fit.named_parameters()}
        restore(best)
        accepted, relative_kl_gain = accept_qk_candidate(before_kl, best_kl, min_relative_kl)
        if not accepted:
            restore(initial)
        # Verify both function preservation AND equality of the temporary fake
        # quant path with the ordinary saved-model path after folding.
        sample = block.raw_input[:min(2, count)].to(device)
        original_state = {key: value.detach().clone() for key, value in block.state_dict().items()}
        original_modes = _module_state(block)
        try:
            with torch.no_grad():
                for module in block.modules():
                    if hasattr(module, "mode"):
                        module.mode = "raw"
                fp_before = block(sample)
                quant_before_fold = work(sample)
                if accepted:
                    fit.fold_into(block.attn)
                fp_after = block(sample)
                if not torch.allclose(fp_before, fp_after, atol=2e-5, rtol=2e-5):
                    raise RuntimeError("QK folding changed floating-point block outputs")
                for module in block.modules():
                    if hasattr(module, "mode"):
                        module.mode = "quant_forward"
                quant_after_fold = block(sample)
                if not torch.allclose(quant_before_fold, quant_after_fold, atol=2e-5, rtol=2e-5):
                    raise RuntimeError("Folded quantized block disagrees with calibration path")
                fp_max_error = float((fp_before - fp_after).abs().max())
        except BaseException:
            block.load_state_dict(original_state)
            raise
        finally:
            _restore_state(original_modes)
        return dict(block=name, variant=variant, protocol=protocol,
                    reconstruction_loss=rec_objective,
                    train_samples=train_count, holdout_samples=holdout,
                    steps=options["steps"], trainable_parameters=sum(p.numel() for p in train_parameters),
                    rotation_rank=rotation_rank, min_relative_kl=min_relative_kl,
                    relative_kl_gain=relative_kl_gain,
                    fisher_rank=reconstructor.k if rec_objective == "dplr" else 0,
                    initial_local_objective=initial_local, best_local_objective=best_score,
                    selected_local_objective=selected_local if accepted else initial_local,
                    teacher_kl_before=before_kl,
                    teacher_kl_after=best_kl if accepted else before_kl,
                    teacher_kl_best_candidate=best_kl,
                    teacher_kl_last_candidate=last_candidate_kl,
                    accepted=accepted, fp_max_abs_error=fp_max_error)
    finally:
        hook.remove()
        del work, fit


def calibrate_qk_block(reconstructor, name, block, device, options):
    """Preserve caller RNG state, including the later QDrop/AdaRound schedule."""
    started = time.perf_counter()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(options["seed"] + int(name.split(".")[-1]))
        result = _calibrate(reconstructor, name, block, device, options)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result["elapsed_seconds"] = time.perf_counter() - started
    logging.info("QK joint result: %s", json.dumps(result, sort_keys=True))
    return result
