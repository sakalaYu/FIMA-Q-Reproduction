"""Evaluate the saved ImageNet-calibrated DPLR arms on ImageNet-V2 Matched Frequency."""

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath

from run_formal_budget import (IMAGENET_HISTORY, check_match, digest, imagenet_history, read_json, write_json)
from run_headwise_experiments import ROOT, dataset_path, gpu_lock, job_lock
from run_lsvit_controls import fp_checkpoint


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


class V2Samples:
    def __init__(self, samples, transform, loader):
        self.samples = samples
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(self.loader(str(path))), label


def numeric_samples(root):
    classes = {}
    for folder in root.iterdir():
        if folder.is_dir() and folder.name.isascii() and folder.name.isdecimal():
            label = int(folder.name)
            if label in classes:
                raise ValueError(f'Duplicate V2 class ID {label} in {root}')
            classes[label] = folder
    if classes.keys() != set(range(1000)):
        raise ValueError(f'Expected the extracted V2 directory with numeric ImageNet classes 0–999: {root}')
    samples = []
    for label, folder in sorted(classes.items()):
        files = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if len(files) != 10:
            raise ValueError(f'Expected 10 images for V2 class {label}, found {len(files)}')
        samples.extend((path, label) for path in files)
    return samples


def compatible_depth(uniform):
    paths = (ROOT / 'checkpoints/quant_result').glob('*_formalbudget_depth_seed3408_*/formal_budget_result.json')
    for path in sorted(paths, reverse=True):
        result = read_json(path)
        if (result['status'] != 'complete' or result['plan']['variant'] != 'depth'
                or result['plan'].get('history') != IMAGENET_HISTORY.parent.name
                or result['plan'].get('sun397_dir')):
            continue
        check_match(result, uniform)
        if any(result['plan'][key] != uniform['plan'][key]
               for key in ('seed', 'fp_model_sha256', 'mr_weights_sha256', 'calibration_hashes')):
            raise ValueError(f'Depth record does not match the original arms: {path}')
        return dict(result, path=str(path))
    return None


def paired_delta(candidate, reference):
    import numpy as np

    wins = int((candidate & ~reference).sum())
    losses = int((reference & ~candidate).sum())
    sizes = np.array([losses, len(reference) - wins - losses, wins])
    bootstrap = np.random.RandomState(3408).multinomial(len(reference), sizes / len(reference), 10000)
    interval = np.quantile((bootstrap[:, 2] - bootstrap[:, 0]) * 100 / len(reference), [0.025, 0.975])
    return dict(delta_pp=(wins - losses) * 100 / len(reference), wins=wins, losses=losses,
                paired_image_bootstrap95_pp=interval.tolist())


def evaluate(args, samples, records, checkpoints, fp, target):
    import torch
    from torch.nn import functional as F
    from torchvision.datasets.folder import default_loader
    from endpoint_diagnosis_impl import load_model, load_teacher
    from utils.datasets import ViTImageNetLoaderGenerator

    uniform = records['uniform']
    teacher = load_teacher(uniform['plan'], fp)
    transform = ViTImageNetLoaderGenerator('', args.batch_size, args.num_workers,
                                          kwargs={'model': teacher}).val_transform
    dataset = V2Samples(samples, transform, default_loader)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)
    correctness, metrics = {}, {}
    with torch.inference_mode():
        for name, record in records.items():
            model = load_model(teacher, dict(cfg=record['cfg']), checkpoints[name]).cuda()
            correct, top5, loss = [], 0, 0.
            for i, (images, labels) in enumerate(loader):
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                logits = model(images)
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f'Nonfinite ImageNet-V2 logits for {name}')
                matches = logits.topk(5, dim=1).indices.eq(labels[:, None])
                correct.append(matches[:, 0].cpu())
                top5 += int(matches.any(dim=1).sum())
                loss += float(F.cross_entropy(logits, labels, reduction='sum'))
                if i % 50 == 0:
                    print(f'{name}: batch {i + 1}/{len(loader)}', flush=True)
            correctness[name] = torch.cat(correct).numpy()
            metrics[name] = dict(top1=int(correctness[name].sum()) * 100 / len(samples),
                                 top5=top5 * 100 / len(samples), loss=loss / len(samples),
                                 source_result=record['path'], checkpoint_sha256=digest(checkpoints[name]))
            print(f'{name}: Top-1 {metrics[name]["top1"]:.3f}%', flush=True)
            del model
    comparisons = {name: paired_delta(correctness['residual'], other)
                   for name, other in correctness.items() if name != 'residual'}
    manifest = hashlib.sha256()
    for path, label in samples:
        manifest.update(f'{path.parent.name}/{path.name}\t{label}\n'.encode())
    write_json(target, dict(dataset='ImageNet-V2 Matched Frequency', images=len(samples),
                            dataset_path=str(samples[0][0].parent.parent), manifest_sha256=manifest.hexdigest(),
                            metrics=metrics, residual_minus=comparisons))
    print(f'Saved: {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v2-dir', required=True, help='Extracted official Matched Frequency directory')
    parser.add_argument('--fp-checkpoint', type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--gpu-slots', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--require-depth', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('Batch size must be positive and num-workers nonnegative')
    path = dataset_path(args.v2_dir)
    if not path.is_dir():
        parser.error(f'ImageNet-V2 directory does not exist: {path}')
    if 'matched-frequency' not in path.name.lower().replace('_', '-'):
        parser.error('Pass the extracted official imagenetv2-matched-frequency directory')
    samples = numeric_samples(path)
    records = imagenet_history()
    depth = compatible_depth(records['uniform'])
    if depth is not None:
        records['depth'] = depth
    elif args.require_depth:
        parser.error('The matched ImageNet depth arm must finish first')
    checkpoints = {name: Path(record['path']).with_name(PurePosixPath(record['checkpoint']).name)
                   for name, record in records.items()}
    fp = fp_checkpoint(args)
    for name, checkpoint in checkpoints.items():
        print(f'{name}: {checkpoint} (present: {checkpoint.is_file()})', flush=True)
    if args.dry_run:
        print(f'Found {len(samples)} samples in 1,000 numeric classes', flush=True)
        return
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing or (fp is not None and not fp.is_file()):
        parser.error(f'Restore missing weights: {missing or [str(fp)]}')
    target = IMAGENET_HISTORY.with_name('imagenet_v2.json')
    with job_lock('formalbudget_imagenet_v2'), gpu_lock(args.gpu, args.gpu_slots) as uuid:
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        evaluate(args, samples, records, checkpoints, fp, target)


if __name__ == '__main__':
    main()
