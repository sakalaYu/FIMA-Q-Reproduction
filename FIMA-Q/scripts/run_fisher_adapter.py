"""Train tiny ViT branch corrections with fixed-data teacher distillation.

The original W4A4 FIMA-Q checkpoint remains frozen.  The Fisher variant uses
teacher pseudo-label gradients only to weight channel-wise feature matching.
"""

import argparse
import hashlib
import json
import pathlib
import random
import sys
import time
from datetime import datetime


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default=str(ROOT.parent / 'imagenet_fimaq'))
    p.add_argument('--checkpoint', default=str(ROOT / 'checkpoints/quant_result/20260916_1917/vit_small_w4_a4_optimsize_1024_fisher_dplr_dis_mode_q_rank_5_qdrop.pth'))
    p.add_argument('--teacher-checkpoint', default=None)
    p.add_argument('--config', default=str(ROOT / 'configs/4bit/best.py'))
    p.add_argument('--mode', choices=['plain', 'fisher'], required=True)
    p.add_argument('--blocks', default='all', help='all or comma-separated ViT block indices')
    p.add_argument('--rank', type=int, default=4)
    p.add_argument('--train-count', type=int, default=960)
    p.add_argument('--holdout-count', type=int, default=64)
    p.add_argument('--fisher-count', type=int, default=128)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--max-steps', type=int, default=0, help='0 means every training batch')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--feature-weight', type=float, default=1.0)
    p.add_argument('--kd-weight', type=float, default=1.0)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--manifest', type=pathlib.Path)
    p.add_argument('--output-root', type=pathlib.Path, default=ROOT / 'checkpoints/fisher_adapter/results')
    p.add_argument('--full-val', action='store_true')
    return p.parse_args()


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha_model(model):
    import torch

    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode('utf-8'))
        digest.update(tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_or_create_manifest(path, dataset, args, transform):
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(args.dataset).resolve()
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        expected = (str(root), len(dataset), args.seed, args.train_count,
                    args.holdout_count, repr(transform))
        actual = (manifest['dataset'], manifest['dataset_size'], manifest['seed'],
                  manifest['train_count'], manifest['holdout_count'], manifest['transform'])
        if actual != expected:
            raise ValueError('Existing image manifest does not match this experiment')
        image_digest = hashlib.sha256()
        for index, relative_path, label in manifest['samples']:
            image_path, current_label = dataset.samples[index]
            if (str(pathlib.Path(image_path).resolve().relative_to(root)) != relative_path
                    or current_label != label):
                raise ValueError('Dataset changed since the image manifest was created')
            image_digest.update(relative_path.encode('utf-8'))
            image_digest.update(sha_file(image_path).encode('ascii'))
        if image_digest.hexdigest() != manifest['image_sha256']:
            raise ValueError('Selected ImageNet image bytes changed since manifest creation')
        return manifest

    count = args.train_count + args.holdout_count
    if count > len(dataset):
        raise ValueError('ImageNet train set has fewer images than requested')
    indices = np.random.default_rng(args.seed).permutation(len(dataset))[:count].tolist()
    samples = []
    image_digest = hashlib.sha256()
    for index in indices:
        image_path, label = dataset.samples[index]
        relative = str(pathlib.Path(image_path).resolve().relative_to(root))
        samples.append([index, relative, label])
        image_digest.update(relative.encode('utf-8'))
        image_digest.update(sha_file(image_path).encode('ascii'))
    manifest = dict(dataset=str(root), dataset_size=len(dataset), seed=args.seed,
                    train_count=args.train_count, holdout_count=args.holdout_count,
                    transform=repr(transform), samples=samples,
                    image_sha256=image_digest.hexdigest())
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def teacher_model(args, device):
    import timm

    name = 'vit_small_patch16_224'
    if args.teacher_checkpoint:
        model = timm.create_model(name, checkpoint_path=args.teacher_checkpoint)
        source = str(pathlib.Path(args.teacher_checkpoint).resolve())
    else:
        raw_path = ROOT / 'checkpoints/vit_raw/vit_small_patch16_224.bin'
        if raw_path.exists():
            model = timm.create_model(name, checkpoint_path=str(raw_path))
            source = str(raw_path)
        else:
            model = timm.create_model(name, pretrained=True)
            source = 'timm pretrained vit_small_patch16_224'
    return model.to(device).eval(), source


def quantized_model(args, device):
    import timm
    import torch
    from torch import nn
    from utils.wrap_net import wrap_modules_in_net
    from quantizers.uniform import UniformQuantizer

    config_dir = str(pathlib.Path(args.config).resolve().parent)
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)
    config_module = __import__(pathlib.Path(args.config).stem)
    cfg = config_module.Config()
    model = timm.create_model('vit_small_patch16_224', pretrained=False)
    model = wrap_modules_in_net(model, cfg, reparam=False)
    state = torch.load(args.checkpoint, map_location='cpu')
    if not isinstance(state, dict):
        raise ValueError('Expected a FIMA-Q state_dict checkpoint')
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            module.mode = 'quant_forward'
            module.calibrated = True
        if isinstance(module, nn.Linear) and 'reduction' in name and module.bias is None:
            module.bias = nn.Parameter(torch.zeros(module.out_features))
        for attr in ('a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer'):
            quantizer = getattr(module, attr, None)
            scale_key = '{}.{}.scale'.format(name, attr)
            if isinstance(quantizer, UniformQuantizer) and scale_key in state:
                quantizer.scale.data = state[scale_key].clone()
                quantizer.inited = True
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError('W4A4 checkpoint mismatch: missing={}, unexpected={}'.format(
            missing, unexpected))
    return model.to(device).eval()


def selected_blocks(specification, count):
    if specification == 'all':
        return list(range(count))
    blocks = [int(part) for part in specification.split(',')]
    if not blocks or len(set(blocks)) != len(blocks) or any(i < 0 or i >= count for i in blocks):
        raise ValueError('Invalid or repeated block index')
    return blocks


def attach_adapters(model, blocks, rank):
    from utils.fisher_adapter import CorrectedBranch

    names = []
    for index in blocks:
        for branch in ('attn', 'mlp'):
            name = 'blocks.{}.{}'.format(index, branch)
            base = getattr(model.blocks[index], branch)
            setattr(model.blocks[index], branch,
                    CorrectedBranch(base, model.embed_dim, rank))
            names.append(name)
    return names


def capture_features(model, names, storage):
    handles = []
    modules = dict(model.named_modules())
    for name in names:
        def save(_module, _inputs, output, key=name):
            storage[key] = output
        handles.append(modules[name].register_forward_hook(save))
    return handles


def set_ste(model, enabled):
    from quantizers.uniform import UniformQuantizer

    for module in model.modules():
        if isinstance(module, UniformQuantizer):
            module.training_mode = enabled
            module.drop_prob = 1.0


def teacher_fisher_weights(teacher, loader, names, count, temperature, device):
    import torch
    import torch.nn.functional as F

    feature_map = {}
    handles = capture_features(teacher, names, feature_map)
    accumulators = {name: torch.zeros(teacher.embed_dim, device=device) for name in names}
    seen = 0
    try:
        for images, _labels in loader:
            if seen >= count:
                break
            images = images[:count - seen].to(device, non_blocking=True)
            feature_map.clear()
            logits = teacher(images) / temperature
            probabilities = F.softmax(logits.detach(), dim=-1)
            sampled_class = torch.multinomial(probabilities, 1)
            log_probability = F.log_softmax(logits, dim=-1).gather(1, sampled_class).sum()
            gradients = torch.autograd.grad(log_probability,
                                            [feature_map[name] for name in names])
            for name, gradient in zip(names, gradients):
                accumulators[name] += gradient.detach().square().sum(dim=(0, 1))
            seen += len(images)
            print('Fisher importance: {}/{} images'.format(seen, count), flush=True)
    finally:
        for handle in handles:
            handle.remove()
        feature_map.clear()
    if seen != count:
        raise RuntimeError('Fisher importance received fewer images than requested')
    weights = {}
    for name, squared_gradient in accumulators.items():
        # Mean-one normalization keeps the plain and Fisher feature losses on
        # comparable scales. A broad fixed clamp limits few-shot noise.
        normalized = squared_gradient / squared_gradient.mean().clamp_min(1e-20)
        clamped = normalized.clamp(0.1, 10.0)
        weights[name] = (clamped / clamped.mean()).cpu()
    return weights


def adapter_state(model, names):
    modules = dict(model.named_modules())
    return {name: {key: value.detach().cpu().clone()
                   for key, value in modules[name].adapter.state_dict().items()}
            for name in names}


def restore_adapters(model, names, state):
    modules = dict(model.named_modules())
    for name in names:
        modules[name].adapter.load_state_dict(state[name])


def evaluate(model, loader, device, teacher=None, temperature=2.0):
    import torch
    import torch.nn.functional as F

    set_ste(model, False)
    model.eval()
    n = correct1 = correct5 = 0
    kl_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            top5 = logits.topk(5, dim=-1).indices
            correct1 += (top5[:, 0] == labels).sum().item()
            correct5 += (top5 == labels[:, None]).any(dim=-1).sum().item()
            if teacher is not None:
                target = teacher(images)
                kl = F.kl_div(F.log_softmax(logits / temperature, dim=-1),
                              F.softmax(target / temperature, dim=-1),
                              reduction='sum') * temperature ** 2
                kl_sum += kl.item()
            n += len(images)
    return dict(images=n, top1=100.0 * correct1 / n,
                top5=100.0 * correct5 / n,
                teacher_kl=kl_sum / n if teacher is not None else None)


def main(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder
    from timm.data import resolve_data_config, create_transform
    from utils.fisher_adapter import normalized_feature_loss

    if min(args.train_count, args.holdout_count, args.batch_size, args.rank, args.epochs) < 1:
        raise ValueError('sample counts, batch size, rank, and epochs must be positive')
    if args.mode == 'fisher' and not 1 <= args.fisher_count <= args.train_count:
        raise ValueError('fisher-count must be between 1 and train-count')
    if args.temperature <= 0 or args.lr <= 0 or args.kd_weight < 0 or args.feature_weight < 0:
        raise ValueError('invalid optimization setting')
    if args.kd_weight + args.feature_weight <= 0:
        raise ValueError('at least one loss weight must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')

    teacher, teacher_source = teacher_model(args, device)
    transform = create_transform(**resolve_data_config(teacher.default_cfg, model=teacher))
    dataset = ImageFolder(str(pathlib.Path(args.dataset) / 'train'), transform=transform)
    manifest_path = args.manifest or (ROOT / 'checkpoints/fisher_adapter' /
        'manifest_{}_seed{}.json'.format(args.train_count + args.holdout_count, args.seed))
    manifest = load_or_create_manifest(manifest_path, dataset, args, transform)
    indices = [row[0] for row in manifest['samples']]
    train_indices = indices[:args.train_count]
    holdout_indices = indices[args.train_count:]
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True,
                              generator=torch.Generator().manual_seed(args.seed))
    fisher_loader = DataLoader(Subset(dataset, train_indices[:args.fisher_count]),
                               batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
    holdout_loader = DataLoader(Subset(dataset, holdout_indices),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
    val_loader = None
    if args.full_val:
        val_dataset = ImageFolder(str(pathlib.Path(args.dataset) / 'val'),
                                  transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

    student = quantized_model(args, device)
    blocks = selected_blocks(args.blocks, len(student.blocks))
    names = attach_adapters(student, blocks, args.rank)
    student.to(device)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    trainable = []
    modules = dict(student.named_modules())
    for name in names:
        for parameter in modules[name].adapter.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    parameter_count = sum(parameter.numel() for parameter in trainable)
    print('Adapter locations:', names, flush=True)
    print('Trainable adapter parameters:', parameter_count, flush=True)

    fisher_weights = None
    if args.mode == 'fisher':
        fisher_weights = teacher_fisher_weights(teacher, fisher_loader, names,
                                                args.fisher_count, args.temperature, device)
        print('Fisher channel ranges:', {name: [float(w.min()), float(w.max())]
                                          for name, w in fisher_weights.items()}, flush=True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output = args.output_root / datetime.now().strftime('%Y%m%d_%H%M%S_%f_{}'.format(args.mode))
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(args={key: str(value) if isinstance(value, pathlib.Path) else value
                          for key, value in vars(args).items()},
                    teacher_source=teacher_source,
                    teacher_state_sha256=sha_model(teacher),
                    quant_checkpoint_sha256=sha_file(args.checkpoint),
                    image_manifest=str(manifest_path.resolve()),
                    image_sha256=manifest['image_sha256'],
                    branches=names, adapter_parameters=parameter_count,
                    torch=torch.__version__)
    (output / 'experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if fisher_weights is not None:
        torch.save(fisher_weights, output / 'fisher_weights.pt')
    print('Results:', output, flush=True)

    teacher_features = {}
    student_features = {}
    teacher_handles = capture_features(teacher, names, teacher_features)
    student_handles = capture_features(student, names, student_features)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    def record(event):
        with (output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
        print(event, flush=True)

    started = time.perf_counter()
    try:
        teacher_baseline = evaluate(teacher, holdout_loader, device)
        student_baseline = evaluate(student, holdout_loader, device, teacher,
                                    args.temperature)
        record(dict(event='baseline', teacher=teacher_baseline, student=student_baseline))
        if val_loader is not None:
            record(dict(event='imagenet_val_baseline', student=evaluate(student, val_loader, device)))
        best_kl = student_baseline['teacher_kl']
        best_epoch = 0
        torch.save(dict(adapters=adapter_state(student, names), metadata=metadata,
                        epoch=0, holdout=student_baseline), output / 'best_adapters.pt')

        for epoch in range(1, args.epochs + 1):
            set_ste(student, True)
            student.eval()
            loss_sum = logit_sum = feature_sum = 0.0
            sample_count = 0
            for step, (images, _labels) in enumerate(train_loader, start=1):
                if args.max_steps and step > args.max_steps:
                    break
                images = images.to(device, non_blocking=True)
                teacher_features.clear()
                student_features.clear()
                with torch.no_grad():
                    teacher_logits = teacher(images)
                student_logits = student(images)
                logit_loss = F.kl_div(
                    F.log_softmax(student_logits / args.temperature, dim=-1),
                    F.softmax(teacher_logits / args.temperature, dim=-1),
                    reduction='batchmean') * args.temperature ** 2
                feature_loss = normalized_feature_loss(
                    student_features, teacher_features, fisher_weights)
                loss = args.kd_weight * logit_loss + args.feature_weight * feature_loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite distillation loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                size = len(images)
                sample_count += size
                loss_sum += loss.item() * size
                logit_sum += logit_loss.item() * size
                feature_sum += feature_loss.item() * size
                teacher_features.clear()
                student_features.clear()
                if step % 20 == 0:
                    print('epoch {} step {}: loss {:.5f}, logits {:.5f}, features {:.5f}'.format(
                        epoch, step, loss_sum / sample_count, logit_sum / sample_count,
                        feature_sum / sample_count), flush=True)
            holdout = evaluate(student, holdout_loader, device, teacher,
                               args.temperature)
            event = dict(event='epoch', epoch=epoch, train_images=sample_count,
                         train_loss=loss_sum / sample_count,
                         train_logit_kl=logit_sum / sample_count,
                         train_feature_loss=feature_sum / sample_count, holdout=holdout)
            record(event)
            torch.save(dict(adapters=adapter_state(student, names), metadata=metadata,
                            epoch=epoch, holdout=holdout), output / 'last_adapters.pt')
            if holdout['teacher_kl'] < best_kl:
                best_kl = holdout['teacher_kl']
                best_epoch = epoch
                torch.save(dict(adapters=adapter_state(student, names), metadata=metadata,
                                epoch=epoch, holdout=holdout), output / 'best_adapters.pt')

        best = torch.load(output / 'best_adapters.pt', map_location='cpu')
        restore_adapters(student, names, best['adapters'])
        record(dict(event='selection', best_epoch=best_epoch, best_holdout_kl=best_kl))
        if val_loader is not None:
            record(dict(event='imagenet_val', best_epoch=best_epoch,
                        student=evaluate(student, val_loader, device)))
        record(dict(event='complete', elapsed_seconds=time.perf_counter() - started))
    finally:
        for handle in teacher_handles + student_handles:
            handle.remove()


if __name__ == '__main__':
    main(arguments())
