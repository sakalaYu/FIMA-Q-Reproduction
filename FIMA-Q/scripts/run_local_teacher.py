"""W3A3 post-reconstruction scale repair. Invoked by run_local_teacher_experiments.py.

All training labels are discarded. E4/E5 differ only in the frozen suffix.
Each block's teacher targets/anchor remain fixed until that block is completed.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "configs/local_teacher_w3a3.json"


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("train_count", "holdout_count", "steps_per_block", "batch_size",
                "eval_batch_size", "fisher_count", "fisher_rank", "save_every"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if cfg["fisher_count"] > cfg["train_count"]:
        raise ValueError("fisher_count exceeds train_count")
    if not 0 < cfg["scale_radius"] < 1 or cfg["relative_lr"] <= 0:
        raise ValueError("Invalid scale constraints")
    if cfg["num_workers"] < 0 or min(cfg["local_weight"], cfg["global_weight"]) < 0:
        raise ValueError("Invalid workers/loss weights")
    if min(cfg["kd_temperature"], cfg["fisher_temperature"]) <= 0:
        raise ValueError("Temperatures must be positive")
    blocks = cfg["blocks"]
    if not blocks or len(set(blocks)) != len(blocks) or any(i not in range(12) for i in blocks):
        raise ValueError("blocks must contain unique ViT-S block indices")
    return cfg


def experiment_identity(cfg):
    paths = [Path(__file__), ROOT / "scripts/run_local_teacher_experiments.py",
             ROOT / "utils/local_teacher.py", ROOT / "utils/wrap_net.py",
             ROOT / "configs/3bit/best.py"]
    for folder in ("quantizers", "quant_layers"):
        paths.extend((ROOT / folder).rglob("*.py"))
    paths.extend(resolve_path(cfg[name]) for name in ("checkpoint", "teacher_checkpoint"))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError("Required file missing; sync it or edit JSON: " + str(path))
    files = {str(p.relative_to(ROOT)) if ROOT in p.parents else str(p): sha_file(p)
             for p in sorted(set(paths))}
    payload = dict(config=cfg, files=files)
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return signature, payload


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".%s.tmp" % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path, value):
    import torch
    temporary = Path(path).with_suffix(".%s.tmp" % os.getpid())
    torch.save(value, temporary)
    temporary.replace(path)


@contextmanager
def file_lock(path):
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_models(cfg, device):
    import timm
    import torch
    from torch import nn
    from quantizers.uniform import UniformQuantizer
    from utils.wrap_net import wrap_modules_in_net
    from utils.local_teacher import quant_mode

    teacher = timm.create_model("vit_small_patch16_224", pretrained=False,
                               checkpoint_path=str(resolve_path(cfg["teacher_checkpoint"])))
    # Exact block boundaries are unchanged by Hadamard, Q/K reciprocal scaling
    # and internal LayerNorm reparameterization. Original FP blocks are valid
    # functions on h_Q; do NOT use hard quantized weights as an FP teacher.
    for block in teacher.blocks:
        if hasattr(block.attn, "fused_attn"):
            block.attn.fused_attn = False
    spec = importlib.util.spec_from_file_location("local_teacher_3bit", ROOT / "configs/3bit/best.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    student = wrap_modules_in_net(timm.create_model("vit_small_patch16_224", pretrained=False),
                                  module.Config(), reparam=False)
    state = torch.load(resolve_path(cfg["checkpoint"]), map_location="cpu")
    if not isinstance(state, dict) or "blocks.0.attn.qkv.weight" not in state:
        raise ValueError("Expected a final FIMA-Q ViT-S state_dict")
    for name, quantizer in student.named_modules():
        if isinstance(quantizer, UniformQuantizer):
            for attr in ("scale", "zero_point"):
                key = name + "." + attr
                if key in state:
                    setattr(quantizer, attr, nn.Parameter(state[key].clone(), requires_grad=False))
            quantizer.inited = True
    student.load_state_dict(state, strict=True)
    for model in (student, teacher):
        model.eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    quant_mode(student)
    if any(q.n_bits != 3 for name, q in student.named_modules()
           if isinstance(q, UniformQuantizer) and name.startswith("blocks.")):
        raise ValueError("Only W3A3 transformer blocks are supported")
    return student, teacher


def record(directory, event):
    with (directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
    print(json.dumps(event, allow_nan=False), flush=True)


def validation(student, cfg, transform, device, baseline=None):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from utils.local_teacher import quant_state
    dataset = ImageFolder(str(resolve_path(cfg["dataset"]) / "val"), transform=transform)
    loader = DataLoader(dataset, batch_size=cfg["eval_batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    predictions, correct, labels_all = [], [], []
    top5_count = count = 0
    with quant_state(student, ste=False), torch.no_grad():
        for images, labels in loader:
            logits = student(images.to(device))
            top = logits.topk(5, -1).indices.cpu()
            predictions.append(top[:, 0])
            correct.append(top[:, 0].eq(labels))
            labels_all.append(labels)
            top5_count += int(top.eq(labels[:, None]).any(-1).sum())
            count += len(labels)
    correct = torch.cat(correct)
    labels_all = torch.cat(labels_all)
    result = dict(images=count, top1=float(correct.float().mean()) * 100,
                  top5=100 * top5_count / count)
    if baseline is not None:
        if not torch.equal(labels_all, baseline["labels"]):
            raise ValueError("Validation order/labels changed; cannot pair predictions")
        old = baseline["correct"]
        result.update(repaired=int((~old & correct).sum()),
                      damaged=int((old & ~correct).sum()),
                      delta_pp=100 * (int(correct.sum()) - int(old.sum())) / count)
    return result, dict(predictions=torch.cat(predictions), correct=correct, labels=labels_all)


def prepare(directory, cfg, signature, student, teacher, device):
    import numpy as np
    import torch
    from torchvision.datasets import ImageFolder
    from timm.data import resolve_data_config, create_transform
    from utils.local_teacher import quant_state, vit_prefix, vit_suffix

    if (directory / "prepared.json").exists():
        return
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    transform = create_transform(**resolve_data_config(teacher.default_cfg, model=teacher))
    dataset = ImageFolder(str(resolve_path(cfg["dataset"]) / "train"), transform=transform)
    total = cfg["train_count"] + cfg["holdout_count"]
    if len(dataset) < total:
        raise ValueError("Insufficient calibration images")
    # Same RandomState index convention as FIMA-Q. Additional holdout indices
    # are beyond the original 1024, rather than held out only from this repair.
    indices = np.random.RandomState(cfg["seed"]).permutation(len(dataset))[:total]
    images, labels, samples = [], [], []
    digest = hashlib.sha256()
    for index in indices:
        image, label = dataset[int(index)]
        path = Path(dataset.samples[int(index)][0])
        relative = str(path.relative_to(resolve_path(cfg["dataset"])))
        digest.update(relative.encode())
        digest.update(sha_file(path).encode())
        digest.update(image.contiguous().numpy().tobytes())
        images.append(image)
        labels.append(label)
        samples.append(dict(index=int(index), path=relative))
    images = torch.stack(images)
    logits = []
    with torch.no_grad():
        for batch in images.split(cfg["eval_batch_size"]):
            logits.append(teacher(batch.to(device)).cpu())
    with torch.no_grad():
        probe = images[:2].to(device)
        reference = student(probe)
        with quant_state(student, ste=True):
            ste = student(probe)
            split = vit_suffix(student, student.blocks[0](vit_prefix(student, probe, 0)), 0)
        max_error = float((reference - ste).abs().max())
        if not torch.allclose(reference, ste, atol=1e-5, rtol=1e-5):
            raise RuntimeError("STE changed forward logits")
        if not torch.allclose(reference, split, atol=1e-5, rtol=1e-5):
            raise RuntimeError("ViT split forward does not reproduce deployed forward")
    record(directory, dict(event="calibration_prepared", samples=total,
                          tensor_image_sha256=digest.hexdigest(), ste_max_error=max_error))
    baseline, predictions = validation(student, cfg, transform, device)
    expected = cfg.get("expected_baseline_top1")
    record(directory, dict(event="baseline", **baseline))
    if expected is not None and abs(baseline["top1"] - expected) > cfg["baseline_tolerance_pp"]:
        raise RuntimeError("Loaded checkpoint Top-1 %.5f differs from expected %.5f; "
                           "check checkpoint/config/loading before training" % (baseline["top1"], expected))
    atomic_torch(directory / "data.pt", dict(images=images, labels=torch.tensor(labels),
                 global_logits=torch.cat(logits), signature=signature))
    atomic_torch(directory / "baseline_predictions.pt", predictions)
    atomic_json(directory / "prepared.json", dict(signature=signature, baseline=baseline,
                data_sha256=digest.hexdigest(), samples=samples, transform=repr(transform),
                torch=torch.__version__, cuda=torch.version.cuda,
                preparation_seconds=time.perf_counter() - started,
                preparation_peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda" else 0))


def block_cache(student, teacher, images, block_index, batch_size, device):
    import torch
    from utils.local_teacher import vit_prefix
    inputs, targets = [], []
    with torch.no_grad():
        for batch in images.split(batch_size):
            h = vit_prefix(student, batch.to(device), block_index)
            inputs.append(h.cpu())
            targets.append(teacher.blocks[block_index](h).cpu())
    return torch.cat(inputs), torch.cat(targets)


def local_logits(suffix, targets, block_index, batch_size, device):
    import torch
    from utils.local_teacher import vit_suffix
    with torch.no_grad():
        return torch.cat([vit_suffix(suffix, batch.to(device), block_index).cpu()
                          for batch in targets.split(batch_size)])


def holdout_metrics(student, teacher, cfg, block_index, inputs, targets,
                    global_logits, local_target, device, route):
    import torch
    from utils.local_teacher import quant_state, vit_suffix, distillation
    total = dict(global_kl=0.0, standardized_kl=0.0, local_kl=0.0)
    with quant_state(student, ste=False), torch.no_grad():
        for start in range(0, len(inputs), cfg["eval_batch_size"]):
            stop = min(start + cfg["eval_batch_size"], len(inputs))
            output = student.blocks[block_index](inputs[start:stop].to(device))
            deployed = vit_suffix(student, output, block_index)
            target = global_logits[start:stop].to(device)
            total["global_kl"] += float(distillation(deployed, target, cfg["kd_temperature"])) * (stop-start)
            total["standardized_kl"] += float(distillation(deployed, target, cfg["kd_temperature"], True)) * (stop-start)
            if local_target is not None:
                logits = vit_suffix(teacher, output, block_index) if route == "fp" else deployed
                total["local_kl"] += float(distillation(logits, local_target[start:stop].to(device),
                                                       cfg["kd_temperature"])) * (stop-start)
    return {name: value / len(inputs) for name, value in total.items()}


def probe_paths(directory, student, teacher, data, cfg, device):
    import torch
    from torch.nn import functional as F
    from utils.local_teacher import quant_state, vit_suffix, distillation
    rows = []
    for b in cfg["blocks"]:
        inputs, targets = block_cache(student, teacher, data["images"], b,
                                      cfg["eval_batch_size"], device)
        row = dict(block=b, splits={})
        cached = {"quant": local_logits(student, targets, b, cfg["eval_batch_size"], device),
                  "fp": local_logits(teacher, targets, b, cfg["eval_batch_size"], device)}
        with torch.no_grad():
            deployed = local_logits(student, torch.cat([
                student.blocks[b](x.to(device)).cpu() for x in inputs.split(cfg["eval_batch_size"])
            ]), b, cfg["eval_batch_size"], device)
        for name, selection in (("train", slice(0, cfg["train_count"])),
                                ("holdout", slice(cfg["train_count"], None))):
            labels = data["labels"][selection]
            old = deployed[selection].argmax(-1).eq(labels)
            stats = {}
            for route, logits in cached.items():
                pred = logits[selection].argmax(-1)
                correct = pred.eq(labels)
                stats[route] = dict(top1=float(correct.float().mean()) * 100,
                                   repaired=int((~old & correct).sum()), damaged=int((old & ~correct).sum()),
                                   global_disagreements=int(pred.ne(data["global_logits"][selection].argmax(-1)).sum()))
            row["splits"][name] = stats
        # Direction diagnostic at the SAME output tensor, no labels used.
        with torch.no_grad():
            value = student.blocks[b](inputs[:cfg["batch_size"]].to(device))
        gradients = {}
        with quant_state(student, ste=True):
            for route, suffix in (("fp", teacher), ("quant", student)):
                leaf = value.detach().requires_grad_(True)
                loss = distillation(vit_suffix(suffix, leaf, b),
                                     cached[route][:len(leaf)].to(device), cfg["kd_temperature"])
                gradients[route] = torch.autograd.grad(loss, leaf)[0].flatten()
        row["gradient_cosine"] = float(F.cosine_similarity(
            gradients["fp"], gradients["quant"], dim=0))
        row["gradient_norms"] = {key: float(value.norm()) for key, value in gradients.items()}
        rows.append(row)
        record(directory, dict(event="path_probe", **row))
    atomic_json(directory / "probe.json", dict(rows=rows,
                note="Labels used for diagnosis only; no block selection or training uses labels."))


def train_arm(directory, root, arm, cfg, signature, student, teacher, data, device, started):
    import torch
    from timm.data import resolve_data_config, create_transform
    from utils.local_teacher import (ARMS, quant_mode, activation_scales, clamp_scales,
        estimate_anchor, dplr_loss, distillation, vit_suffix, cpu_tree, anchor_to)
    route, global_kind = ARMS[arm]
    n = cfg["train_count"]
    progress_path = directory / "progress.pt"
    progress = torch.load(progress_path, map_location="cpu") if progress_path.exists() else None
    if progress and progress["signature"] != signature:
        raise RuntimeError("Resume fingerprint differs")
    if progress:
        student.load_state_dict(progress["model"], strict=True)
    first = progress["position"] if progress else 0
    elapsed_before = progress.get("elapsed_seconds", 0.0) if progress else 0.0
    origin = torch.load(resolve_path(cfg["checkpoint"]), map_location="cpu")
    allowed_changes = {"blocks.%d.%s" % (b, name) for b in cfg["blocks"]
                       for name in activation_scales(student.blocks[b])}
    for position in range(first, len(cfg["blocks"])):
        b = cfg["blocks"][position]
        block_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        quant_mode(student, ste=False)
        inputs, targets = block_cache(student, teacher, data["images"], b,
                                      cfg["eval_batch_size"], device)
        local_target = local_logits(teacher if route == "fp" else student, targets, b,
                                    cfg["eval_batch_size"], device) if route else None
        parameters = activation_scales(student.blocks[b])
        for parameter in parameters.values():
            parameter.requires_grad_(True)
        resumed = progress is not None and progress["position"] == position and progress["step"] > 0
        centers = progress["centers"] if resumed else {key: value.detach().cpu().clone()
                                                      for key, value in parameters.items()}
        if any((value <= 0).any() or not torch.isfinite(value).all() for value in centers.values()):
            raise ValueError("Activation scale centers must be finite and positive")
        optimizer = torch.optim.Adam([dict(params=[value],
            lr=cfg["relative_lr"] * float(centers[key].abs().mean())) for key, value in parameters.items()])
        generator = torch.Generator().manual_seed(cfg["seed"] + 1009 * b)
        if resumed:
            anchor = anchor_to(progress["anchor"], device)
            optimizer.load_state_dict(progress["optimizer"])
            generator.set_state(progress["sampler_state"])
            start_step = progress["step"]
        else:
            anchor = estimate_anchor(student, teacher, b, inputs[:n], targets[:n],
                     data["global_logits"][:n], cfg["fisher_count"], cfg["fisher_rank"],
                     cfg["batch_size"], cfg["fisher_temperature"], cfg["seed"] + b)
            start_step = 0
        record(directory, dict(event="block_start", block=b, resumed_step=start_step,
                trainable_parameters=sum(value.numel() for value in parameters.values()),
                gram_condition=anchor["gram_condition"],
                holdout=holdout_metrics(student, teacher, cfg, b, inputs[n:], targets[n:],
                    data["global_logits"][n:], local_target[n:] if local_target is not None else None,
                    device, route)))

        def save_progress(step, next_position=None):
            atomic_torch(progress_path, dict(signature=signature, model=cpu_tree(student.state_dict()),
                position=position if next_position is None else next_position,
                step=step, centers=cpu_tree(centers), anchor=cpu_tree(anchor),
                optimizer=cpu_tree(optimizer.state_dict()), sampler_state=generator.get_state(),
                elapsed_seconds=elapsed_before + time.perf_counter() - started))

        quant_mode(student, ste=True)
        save_progress(start_step)
        for step in range(start_step, cfg["steps_per_block"]):
            indices = torch.randperm(n, generator=generator)[:cfg["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            output = student.blocks[b](inputs[indices].to(device))
            reconstruction = dplr_loss(output, targets[indices].to(device), anchor)
            local_loss = global_loss = output.new_zeros(())
            quant_logits = None
            if route:
                suffix = teacher if route == "fp" else student
                logits = vit_suffix(suffix, output, b)
                if route == "quant":
                    quant_logits = logits
                local_loss = distillation(logits, local_target[indices].to(device), cfg["kd_temperature"])
            if global_kind:
                if quant_logits is None:
                    quant_logits = vit_suffix(student, output, b)
                global_loss = distillation(quant_logits, data["global_logits"][indices].to(device),
                                          cfg["kd_temperature"], global_kind == "standardized")
            loss = reconstruction + cfg["local_weight"] * local_loss + cfg["global_weight"] * global_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; last progress is preserved")
            loss.backward()
            norm = sum(float(value.grad.detach().square().sum()) for value in parameters.values()
                       if value.grad is not None) ** 0.5
            if all(value.grad is None for value in parameters.values()):
                raise RuntimeError("Activation scales are disconnected from the loss")
            if not all(value.grad is None or torch.isfinite(value.grad).all()
                       for value in parameters.values()):
                raise FloatingPointError("Non-finite activation scale gradient")
            optimizer.step()
            clamp_scales(parameters, centers, cfg["scale_radius"])
            if (step + 1) % cfg["save_every"] == 0 or step + 1 == cfg["steps_per_block"]:
                save_progress(step + 1)
                record(directory, dict(event="train", block=b, step=step+1,
                    reconstruction=float(reconstruction), local_kd=float(local_loss),
                    global_kd=float(global_loss), gradient_norm=norm))
        for value in parameters.values():
            value.requires_grad_(False)
        quant_mode(student, ste=False)
        record(directory, dict(event="block_complete", block=b,
            seconds=time.perf_counter() - block_started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            holdout=holdout_metrics(student, teacher, cfg, b, inputs[n:], targets[n:],
                data["global_logits"][n:], local_target[n:] if local_target is not None else None,
                device, route)))
        save_progress(0, position + 1)
        progress = None
        del inputs, targets, local_target, anchor, optimizer
    current = student.state_dict()
    changed = [key for key in current if not torch.equal(current[key].cpu(), origin[key])]
    if any(key not in allowed_changes for key in changed):
        raise RuntimeError("An undeclared tensor changed: " + str(set(changed) - allowed_changes))
    atomic_torch(directory / "model.pth", cpu_tree(current))
    transform = create_transform(**resolve_data_config(teacher.default_cfg, model=teacher))
    baseline_predictions = torch.load(root / "baseline_predictions.pt", map_location="cpu")
    result, predictions = validation(student, cfg, transform, device, baseline_predictions)
    atomic_torch(directory / "predictions.pt", predictions)
    result.update(status="completed", arm=arm, signature=signature, changed_tensors=changed,
        elapsed_seconds=elapsed_before + time.perf_counter() - started,
        model_file_bytes=(directory / "model.pth").stat().st_size,
        total_parameters=sum(p.numel() for p in student.parameters()))
    prepared_path = root / "prepared.json"
    if prepared_path.exists():
        prepared = json.loads(prepared_path.read_text())
        result["shared_preparation_seconds"] = prepared["preparation_seconds"]
        result["seconds_including_shared_preparation"] = (
            result["elapsed_seconds"] + prepared["preparation_seconds"])
    atomic_json(directory / "result.json", result)
    record(directory, dict(event="complete", **result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--arm", choices=("prepare", "probe", "E1", "E2", "E3", "E4", "E5", "E6"), required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = read_config(args.config)
    signature, identity = experiment_identity(cfg)
    if signature != args.signature:
        raise RuntimeError("Source/config/checkpoint changed since launch")
    root = resolve_path(cfg["output_root"]) / signature[:16]
    directory = root if args.arm == "prepare" else root / args.arm
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    import numpy as np
    import torch
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    student, teacher = load_models(cfg, device)
    if args.arm == "prepare":
        with file_lock(root / "prepare.lock"):
            prepare(root, cfg, signature, student, teacher, device)
        atomic_json(root / "sources.json", identity)
        return
    if not (root / "prepared.json").exists():
        raise RuntimeError("Run launcher --suite prepare first")
    metadata = json.loads((root / "prepared.json").read_text())
    if metadata["signature"] != signature:
        raise RuntimeError("Preparation fingerprint differs")
    data = torch.load(root / "data.pt", map_location="cpu")
    if data["signature"] != signature:
        raise RuntimeError("Data cache fingerprint differs")
    if args.arm == "probe":
        probe_paths(directory, student, teacher, data, cfg, device)
    else:
        train_arm(directory, root, args.arm, cfg, signature, student, teacher, data, device, started)


if __name__ == "__main__":
    main()
