"""Screen deployable A3 scale changes on the saved seed-3407 LS+MR endpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time

from run_endpoint_diagnosis import REFERENCES, read_json
from run_headwise_experiments import ROOT, dataset_path, gpu_lock, job_lock


def fresh_batches(generator, meta, data, batch_size):
    import numpy as np
    import torch
    from endpoint_diagnosis_impl import PROBE_SIZE, probe_indices

    dataset = generator.train_set
    prior_probe = probe_indices(len(dataset), meta)  # Also checks the upstream calibration hashes.
    start = max(meta['cfg']['calib_size'], meta['cfg']['optim_size']) + len(prior_probe)
    order = np.random.RandomState(meta['args']['seed']).permutation(len(dataset))
    selected = order[start:start + 2 * PROBE_SIZE].tolist()
    if len(selected) != 2 * PROBE_SIZE:
        raise RuntimeError('The original ImageNet train split has too few fresh probe images')
    dataset.transform = generator.val_transform
    digest = hashlib.sha256()
    batches = []
    for start in range(0, len(selected), PROBE_SIZE):
        images = []
        for index in selected[start:start + PROBE_SIZE]:
            image, _ = dataset[index]
            path = Path(dataset.samples[index][0]).relative_to(data).as_posix()
            digest.update(path.encode('utf-8') + b'\0')
            digest.update(image.contiguous().numpy().tobytes())
            images.append(image)
        batches.append([torch.stack(images[i:i + batch_size]).cuda() for i in range(0, len(images), batch_size)])
    return batches, dict(indices=selected, per_split=PROBE_SIZE, tensor_sha256=digest.hexdigest())


def choose(rows, base_kl, min_gain):
    # Unity is the first candidate, so equal candidate scores keep the original grid.
    winner = min(rows, key=lambda row: row['kl'])
    return winner if base_kl - winner['kl'] > min_gain else rows[0]


def confirm_gain(base, changed):
    import numpy as np

    delta = (base - changed).double().numpy()
    indices = np.random.RandomState(931).randint(len(delta), size=(5000, len(delta)))
    low, high = np.quantile(delta[indices].mean(axis=1), (0.025, 0.975))
    mean = float(delta.mean())
    return dict(kl=float(changed.mean()), kl_gain=mean, relative_gain=mean / max(float(base.mean()), 1e-12),
                bootstrap_95=[float(low), float(high)])


def screen(args, meta, checkpoint, data, fp_path, identity, target):
    import torch
    import endpoint_diagnosis_impl as ep
    from utils.datasets import ViTImageNetLoaderGenerator

    started = time.perf_counter()
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    with torch.inference_mode():
        teacher = ep.load_teacher(meta, fp_path)
        generator = ViTImageNetLoaderGenerator(data, args.val_batch_size, args.num_workers, kwargs={'model': teacher})
        batches, images = fresh_batches(generator, meta, data, args.probe_batch_size)
        model = ep.load_model(teacher, meta, checkpoint).cuda()
        teacher.cuda()
        targets = [[teacher(group).float().log_softmax(dim=-1) for group in split] for split in batches]
        del teacher, generator
        base = [ep.batch_kl(model, batch, logits) for batch, logits in zip(batches, targets)]
        base_fit = float(base[0].mean())
        print(f'Baseline KL (fit / confirm): {base_fit:.6f} / {float(base[1].mean()):.6f}', flush=True)

        rows = []
        for index, block in enumerate(model.blocks):
            fc2 = block.mlp.fc2
            if fc2.a_quantizer.n_bits != 3:
                raise RuntimeError(f'Unexpected activation precision for FC2 block {index}')
            # The local policy uses only the NMSE field of this probe, on the fit split.
            local = ep.probe_site(model, (fc2.a_quantizer,), batches[:1], targets[:1], base[:1], fc2=fc2)
            candidates = [dict(factor=1.0, kl=base_fit)]
            for factor in ep.FACTORS[1:]:
                with ep.apply_scales(model, {str(index): factor}):
                    kl = float(ep.batch_kl(model, batches[0], targets[0]).mean())
                candidates.append(dict(factor=factor, kl=kl))
            best = choose(candidates, base_fit, ep.MIN_KL_GAIN)
            rows.append(dict(block=index, local_nmse=local['local_nmse'][0], candidates=candidates, best=best))
            print(f'Block {index} best {best["factor"]:.2f}, fit KL gain {base_fit - best["kl"]:+.6f}', flush=True)

        direct = min(rows, key=lambda row: row['best']['kl'])
        local = max(rows, key=lambda row: row['local_nmse'])
        plans = {}
        for name, row in (('global', direct), ('local', local)):
            factor = row['best']['factor']
            factors = {str(row['block']): factor} if factor != 1.0 else {}
            with ep.apply_scales(model, factors):
                confirmed = confirm_gain(base[1], ep.batch_kl(model, batches[1], targets[1]))
            plans[name] = dict(factors=factors, block=row['block'], fit_kl=row['best']['kl'], confirm=confirmed)
            print(f'{name} {factors}: confirm KL change {confirmed["kl_gain"]:+.6f} (positive is better)', flush=True)

    confirmed = plans['global']['confirm']
    if plans['global']['factors'] == plans['local']['factors']:
        decision = 'stop: both policies produce the same scale settings, including possible no change'
    elif confirmed['relative_gain'] < 0.01 or confirmed['bootstrap_95'][0] <= 0:
        decision = 'stop: direct scale change lacks a confirmed KL gain (at least 1% with positive bootstrap lower bound)'
    else:
        decision = 'ready: direct FC2 scale change passes the confirmation screen'
    ep.write_json(target, dict(identity=identity, historical_top1=meta['after']['top1'], images=images,
        policy=dict(global_='rank blocks by achievable KL gain in the scale grid', local='rank blocks by local NMSE'),
        baseline_kl=[float(k.mean()) for k in base], rows=rows, plans=plans, ready_for_validation=decision.startswith('ready:'),
        decision=decision, seconds=time.perf_counter() - started))
    print(f'{decision}\nSaved: {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('screen', 'validate'), required=True)
    parser.add_argument('--dataset', default='../imagenet_fimaq')
    parser.add_argument('--fp-checkpoint', help='Original ViT-S floating-point weights')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--gpu-slots', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--probe-batch-size', type=int, default=8)
    parser.add_argument('--val-batch-size', type=int, default=64)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if min(args.probe_batch_size, args.val_batch_size) < 1 or args.num_workers < 0:
        parser.error('Batch sizes must be positive and --num-workers must be nonnegative')

    record = ROOT / 'checkpoints/quant_result' / REFERENCES['residual'] / 'method_result.json'
    if not record.is_file():
        parser.error(f'Original residual MR run record missing: {record}')
    meta = read_json(record)
    if (meta['status'] != 'complete' or meta['control'] != 'fixed' or meta['mr_variant'] != 'relu'
            or meta['args']['model'] != 'vit_small' or meta['args']['seed'] != 3407
            or meta['args']['ls_mr_budget'] != 'residual'
            or (meta['cfg']['w_bit'], meta['cfg']['a_bit'], meta['cfg']['optim_metric']) != (3, 3, 'lsvit')):
        parser.error(f'Unexpected source checkpoint recipe in {record}')
    checkpoint = record.with_name(PurePosixPath(meta['checkpoint_after']).name)
    data = dataset_path(args.dataset)
    local = [ROOT / 'checkpoints/vit_raw' / name for name in ('model.safetensors', 'vit_small_patch16_224.bin')]
    fp_path = dataset_path(args.fp_checkpoint) if args.fp_checkpoint else next((p for p in local if p.is_file()), None)
    print(f'3407 residual MR: {meta["after"]["top1"]:.3f}%\nCheckpoint: {checkpoint}\nDataset: {data}', flush=True)
    print(f'FP teacher: {fp_path or "timm pretrained weights (fingerprint checked)"}', flush=True)
    if args.dry_run:
        print(f'Checkpoint present: {checkpoint.is_file()}')
        return
    if not checkpoint.is_file():
        parser.error(f'Original optimized checkpoint missing: {checkpoint}')
    if fp_path is not None and not fp_path.is_file():
        parser.error(f'FP weights missing: {fp_path}')
    if not (data / ('train' if args.stage == 'screen' else 'val')).is_dir():
        parser.error(f'Missing ImageNet split under {data}')

    with job_lock('fc2_scale_screen_seed3407'), gpu_lock(args.gpu, args.gpu_slots) as uuid:
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        sys.path.insert(0, str(ROOT))
        from endpoint_diagnosis_impl import validate_plan
        from utils.lsvit_pair import file_sha

        sources = ('scripts/run_fc2_scale_screen.py', 'scripts/endpoint_diagnosis_impl.py', 'utils/wrap_net.py',
                   'utils/datasets.py', 'utils/test_utils.py', 'quantizers/uniform.py', 'quantizers/adaround.py',
                   'quant_layers/linear.py', 'quant_layers/matmul.py', 'quant_layers/conv.py')
        identity = dict(checkpoint_sha256=file_sha(checkpoint), record_sha256=file_sha(record), dataset=str(data),
                        probe_batch_size=args.probe_batch_size, sources={path: file_sha(ROOT / path) for path in sources})
        tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        directory = ROOT / 'checkpoints/fc2_scale_screen' / f'seed3407_residual_{tag}'
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / 'screen.json'
        if args.stage == 'screen' and not target.is_file():
            screen(args, meta, checkpoint, data, fp_path, identity, target)
            return
        if not target.is_file():
            parser.error('Run --stage screen with the same options first')
        plan = read_json(target)
        if plan['identity'] != identity:
            raise RuntimeError(f'Saved screen fingerprint mismatch: {target}')
        print(f'{plan["decision"]}\nSaved: {target}', flush=True)
        if args.stage == 'validate' and plan['ready_for_validation']:
            validate_plan(args, meta, checkpoint, data, fp_path, plan, directory / 'validation.json')


if __name__ == '__main__':
    main()
