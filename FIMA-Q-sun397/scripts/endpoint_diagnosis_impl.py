"""Measure quantizer interventions in the saved, fully quantized ViT."""

from contextlib import contextmanager
import copy
import hashlib
import json
import logging
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from quantizers.adaround import AdaRoundQuantizer
from quantizers.uniform import UniformQuantizer
from utils.lsvit_pair import write_json


PROBE_SIZE = 128  # Per split. Both splits exclude the original reconstruction data.
TOP_K = 3
FACTORS = (1.0, 0.90, 0.95, 1.05, 1.10)
MIN_KL_GAIN = 1e-5


def load_teacher(meta, fp_path):
    import timm

    options = {'checkpoint_path': str(fp_path)} if fp_path else {'pretrained': True}
    model = timm.create_model('vit_small_patch16_224', **options).eval()
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode('utf-8'))
        digest.update(value.cpu().contiguous().numpy().tobytes())
    if digest.hexdigest() != meta['fp_model_sha256']:
        raise RuntimeError('FP teacher fingerprint differs from the original reconstruction run')
    return model


def restore_quantized_state(model, state):
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            module.mode = 'quant_forward'
            module.calibrated = True
        for attr in ('a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer'):
            quantizer = getattr(module, attr, None)
            if quantizer is None:
                continue
            if not isinstance(quantizer, UniformQuantizer):
                raise TypeError(f'Unexpected quantizer: {name}.{attr}')
            quantizer.inited = True
            for prop in ('scale', 'zero_point'):
                if hasattr(quantizer, prop):
                    value = state[f'{name}.{attr}.{prop}']
                    if not torch.isfinite(value).all() or (prop == 'scale' and (value == 0).any()):
                        raise ValueError(f'Invalid saved quantizer parameter: {name}.{attr}.{prop}')
                    # Some calibrated activation scales have token-dependent shapes.
                    getattr(quantizer, prop).data = value.clone()
            if attr == 'w_quantizer':
                # Reconstruction saves baked weights in AdaRound nearest mode. That
                # forward uses the stored weight zero point without rounding it.
                quantizer.round_mode = 'nearest'
                quantizer.forward = MethodType(AdaRoundQuantizer.forward, quantizer)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def load_model(teacher, meta, checkpoint):
    from utils.wrap_net import wrap_modules_in_net

    model = copy.deepcopy(teacher)
    if len(model.blocks) != 12 or any(not isinstance(b.mlp.act, nn.GELU) for b in model.blocks):
        raise RuntimeError('Expected the original 12-block ViT-S structure')
    for block in model.blocks:
        block.mlp.act = nn.ReLU()
    model = wrap_modules_in_net(model, SimpleNamespace(**meta['cfg']), reparam=False)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    return restore_quantized_state(model, state)


def probe_indices(length, meta):
    permutation = np.random.RandomState(meta['args']['seed']).permutation(length)
    hashes = {int(count): sample_hash for count, sample_hash, _ in meta['calibration_hashes']}
    counts = (meta['cfg']['calib_size'], meta['cfg']['optim_size'])
    for count in counts:
        sample_hash = hashlib.sha256(np.asarray(permutation[:count], dtype='<i8').tobytes()).hexdigest()
        if sample_hash != hashes.get(count):
            raise RuntimeError(f'The original {count}-image index fingerprint differs for this ImageNet train split')
    selected = permutation[max(counts):max(counts) + 2 * PROBE_SIZE]
    if len(selected) != 2 * PROBE_SIZE:
        raise RuntimeError('Not enough fresh training images for two disjoint probe splits')
    return selected.tolist()


def make_probe_batches(generator, meta, data, batch_size):
    dataset = generator.train_set
    indices = probe_indices(len(dataset), meta)
    dataset.transform = generator.val_transform
    digest = hashlib.sha256()
    paths = []
    batches = []
    for start in range(0, len(indices), PROBE_SIZE):
        images = []
        for index in indices[start:start + PROBE_SIZE]:
            path = Path(dataset.samples[index][0]).relative_to(data).as_posix()
            image, _ = dataset[index]
            paths.append(path)
            digest.update(path.encode('utf-8') + b'\0')
            digest.update(image.contiguous().numpy().tobytes())
            images.append(image)
        batches.append([torch.stack(images[i:i + batch_size]) for i in range(0, len(images), batch_size)])
    return batches, dict(indices=indices, paths=paths, tensor_sha256=digest.hexdigest(), per_split=PROBE_SIZE)


def batch_kl(model, batches, targets):
    scores = []
    for images, teacher_logp in zip(batches, targets):
        logp = model(images).float().log_softmax(dim=-1)
        scores.append(F.kl_div(logp, teacher_logp, log_target=True, reduction='none').sum(dim=-1))
    values = torch.cat(scores).cpu()
    if not torch.isfinite(values).all():
        raise RuntimeError('Nonfinite teacher KL during endpoint evaluation')
    return values


@contextmanager
def bypass_quantizers(quantizers):
    handles = []
    try:
        for quantizer in quantizers:
            handles.append(quantizer.register_forward_hook(lambda _, args, output: args[0]))
        yield
    finally:
        for handle in handles:
            handle.remove()


def probe_site(model, quantizers, batches, targets, baseline, fc2=None):
    local = []

    def collect(module, args, unquantized_input_output):
        x = args[0]
        # Direct forward skips the hook so this still uses the original A3 grid.
        qx = module.a_quantizer.forward(x)
        weight, _ = module.quant_weight_bias()
        error = F.linear(qx - x, weight).float().square().flatten(1).sum(dim=1)
        energy = unquantized_input_output.float().square().flatten(1).sum(dim=1)
        local.append(torch.stack((error, energy)).cpu())

    handle = fc2.register_forward_hook(collect) if fc2 is not None else None
    try:
        with bypass_quantizers(quantizers):
            bypass = [batch_kl(model, group, logp) for group, logp in zip(batches, targets)]
    finally:
        if handle is not None:
            handle.remove()
    row = dict(recovery=[float((base - other).mean()) for base, other in zip(baseline, bypass)])
    if len(local):
        row['local_nmse'] = [float(error.sum() / energy.sum().clamp_min(1e-12))
                             for error, energy in torch.cat(local, dim=1).split(PROBE_SIZE, dim=1)]
        if not np.isfinite(row['local_nmse']).all():
            raise RuntimeError('Nonfinite FC2 local probe NMSE')
    return row


def rank(rows, metric, split):
    return sorted(rows, key=lambda block: (-rows[block][metric][split], block))


def spearman(x, y):
    def rank_values(values):
        _, groups, sizes = np.unique(values, return_inverse=True, return_counts=True)
        return (sizes.cumsum() - (sizes - 1) / 2)[groups]

    left, right = rank_values(x), rank_values(y)
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


@contextmanager
def apply_scales(model, factors):
    originals = {}
    try:
        for index, factor in factors.items():
            scale = model.blocks[int(index)].mlp.fc2.a_quantizer.scale
            originals[index] = (scale, scale.detach().clone())
            scale.mul_(factor)
        yield
    finally:
        for scale, original in originals.values():
            scale.copy_(original)


def fit_scales(model, blocks, batches, targets):
    factors, trials = {}, []
    for index in sorted(blocks):
        results = []
        for factor in FACTORS:
            with apply_scales(model, {**factors, str(index): factor}):
                score = float(batch_kl(model, batches, targets).mean())
            results.append(dict(factor=factor, fit_kl=score))
        winner = min(results, key=lambda entry: entry['fit_kl'])
        if results[0]['fit_kl'] - winner['fit_kl'] <= MIN_KL_GAIN:
            winner = results[0]
        factors[str(index)] = winner['factor']
        trials.append(dict(block=index, candidates=results, chosen=winner['factor']))
        print(f'FC2 block {index}: factor {winner["factor"]:.2f}, fit KL {winner["fit_kl"]:.6f}', flush=True)
    return dict(factors=factors, trials=trials)


@torch.inference_mode()
def diagnose(args, meta, checkpoint, data, fp_path, identity, target):
    from utils.datasets import ViTImageNetLoaderGenerator

    started = time.perf_counter()
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    teacher = load_teacher(meta, fp_path)
    generator = ViTImageNetLoaderGenerator(data, args.val_batch_size, args.num_workers, kwargs={'model': teacher})
    batches, images = make_probe_batches(generator, meta, data, args.probe_batch_size)
    batches = [[x.cuda() for x in split] for split in batches]
    model = load_model(teacher, meta, checkpoint).cuda()
    teacher.cuda()
    targets = [[teacher(x).float().log_softmax(dim=-1) for x in split] for split in batches]
    del teacher, generator
    baseline = [batch_kl(model, split, logp) for split, logp in zip(batches, targets)]
    base_kl = [float(values.mean()) for values in baseline]
    print(f'Baseline teacher KL (fit / confirm): {base_kl}', flush=True)

    sites = {'fc2': {}, 'fc1': {}, 'qk': {}}
    for index, block in enumerate(model.blocks):
        quantizers = {'fc2': (block.mlp.fc2.a_quantizer,), 'fc1': (block.mlp.fc1.a_quantizer,),
                      'qk': (block.attn.matmul1.A_quantizer, block.attn.matmul1.B_quantizer)}
        for name, modules in quantizers.items():
            if any(q.n_bits != 3 for q in modules):
                raise RuntimeError(f'Expected A3 at block {index}, {name}')
            sites[name][index] = probe_site(model, modules, batches, targets, baseline,
                                           fc2=block.mlp.fc2 if name == 'fc2' else None)
        print(f'Block {index}: ' + ', '.join(f'{name} KL recovery {rows[index]["recovery"]}'
                                             for name, rows in sites.items()), flush=True)

    rows = sites['fc2']
    global_fit, global_confirm = (rank(rows, 'recovery', split) for split in (0, 1))
    local_fit, local_confirm = (rank(rows, 'local_nmse', split) for split in (0, 1))
    selected = global_fit[:TOP_K]
    overlap = set(selected) & set(global_confirm[:TOP_K])
    recovered = [i for i in overlap if min(rows[i]['recovery']) > MIN_KL_GAIN]
    global_stability = spearman([r['recovery'][0] for r in rows.values()], [r['recovery'][1] for r in rows.values()])
    local_vs_global = [spearman([r['local_nmse'][s] for r in rows.values()],
                                [r['recovery'][s] for r in rows.values()]) for s in (0, 1)]
    report = dict(identity=identity, historic_top1=meta['after']['top1'], images=images,
                  baseline_kl=base_kl, sites=sites, global_rank=[global_fit, global_confirm],
                  local_rank=[local_fit, local_confirm], global_rank_spearman=global_stability,
                  global_top3_shared=sorted(overlap), global_top3_positive_both=sorted(recovered),
                  local_vs_global_spearman=local_vs_global, plans={}, ready_for_validation=False)
    if global_stability is None or global_stability < 0.5 or len(recovered) < 2:
        decision = 'stop: FC2 recovery ranking is not sufficiently stable across the two calibration splits'
    elif set(selected) == set(local_fit[:TOP_K]):
        decision = 'stop: both rankings select the same three blocks, so this comparison cannot isolate site selection'
    else:
        for name, blocks in (('global', selected), ('local', local_fit[:TOP_K])):
            print(f'{name} FC2 scale search: blocks {blocks}', flush=True)
            plan = fit_scales(model, blocks, batches[0], targets[0])
            with apply_scales(model, plan['factors']):
                plan['kl'] = [float(batch_kl(model, batch, logp).mean()) for batch, logp in zip(batches, targets)]
            report['plans'][name] = plan
            print(f'{name} selected scales: {plan["factors"]}, KL fit/confirm {plan["kl"]}', flush=True)
        global_plan, local_plan = (report['plans'][name] for name in ('global', 'local'))
        changed = [{block: factor for block, factor in plan['factors'].items() if factor != 1.0}
                   for plan in (global_plan, local_plan)]
        if base_kl[1] - global_plan['kl'][1] <= MIN_KL_GAIN:
            decision = 'stop: scale search does not improve held-out teacher KL'
        elif changed[0] == changed[1]:
            decision = 'stop: both rankings lead to identical effective scale changes'
        else:
            report['ready_for_validation'] = True
            decision = 'ready: global repair improves held-out teacher KL'
    report.update(decision=decision, seconds=time.perf_counter() - started)
    write_json(target, report)
    print(f'{decision}\nSaved: {target}', flush=True)


@torch.inference_mode()
def validate_plan(args, meta, checkpoint, data, fp_path, plan, target):
    from utils.datasets import ViTImageNetLoaderGenerator
    from utils.test_utils import validate

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    teacher = load_teacher(meta, fp_path)
    generator = ViTImageNetLoaderGenerator(data, args.val_batch_size, args.num_workers, kwargs={'model': teacher})
    loader = generator.val_loader()
    if len(loader.dataset) != 50000:
        raise RuntimeError(f'Expected 50,000 ImageNet validation images, got {len(loader.dataset)}')
    arms = {'baseline': {}, **{name: item['factors'] for name, item in plan['plans'].items()}}
    identity = dict(screen=plan['identity'], calibration_sha256=plan['images']['tensor_sha256'],
                    val_batch_size=args.val_batch_size, factors=arms)
    result = dict(identity=identity, rows={}, status='partial', historical_top1=meta['after']['top1'])
    if target.is_file():
        result = json.loads(target.read_text(encoding='utf-8'))
        if result['identity'] != identity:
            raise RuntimeError(f'Validation options differ from the saved run: {target}')
        if result['status'] == 'complete':
            print(f'Validation already complete: {target}', flush=True)
            return
    model = load_model(teacher, meta, checkpoint).cuda()
    del teacher, generator
    criterion = nn.CrossEntropyLoss().cuda()
    for name, factors in arms.items():
        if name in result['rows']:
            print(f'Reusing completed {name}: {result["rows"][name]["top1"]:.3f}%', flush=True)
        else:
            print(f'ImageNet validation: {name}, FC2 scale factors {factors}', flush=True)
            started = time.perf_counter()
            with apply_scales(model, factors):
                loss, top1, top5 = validate(loader, model, criterion, print_freq=50, device='cuda')
            if not np.isfinite((loss, top1, top5)).all():
                raise RuntimeError(f'Nonfinite validation metric for {name}')
            result['rows'][name] = dict(top1=top1, top5=top5, loss=loss, seconds=time.perf_counter() - started)
            write_json(target, result)
        if name == 'baseline' and abs(result['rows'][name]['top1'] - meta['after']['top1']) > 0.02:
            result['status'] = 'baseline_mismatch'
            write_json(target, result)
            raise RuntimeError('Loaded endpoint differs from its historical baseline by more than 0.02 pp. '
                               'Check the checkpoint and validation dataset before comparing corrections.')
    b, g, l = (result['rows'][name]['top1'] for name in ('baseline', 'global', 'local'))
    result.update(status='complete', global_delta_pp=g - b, local_delta_pp=l - b, global_vs_local_pp=g - l)
    write_json(target, result)
    print(f'Top-1 {b:.3f}% baseline, {g:.3f}% global, {l:.3f}% local; global vs local {g - l:+.3f} pp', flush=True)
    print(f'Saved: {target}', flush=True)
