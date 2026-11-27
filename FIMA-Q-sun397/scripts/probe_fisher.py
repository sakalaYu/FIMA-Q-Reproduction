"""Structure diagnostics and forward-only projected Fisher validation.

No model training or checkpoint writing. Run --help for experiment controls.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import time
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['prepare', 'structure', 'forward', 'both'], default='both')
    p.add_argument('--dataset', default=str(ROOT.parent / 'imagenet_fimaq'))
    p.add_argument('--checkpoint', default=str(ROOT / 'checkpoints/quant_result/20260916_1818/vit_small_w4_a4_calibsize_128_mse.pth'))
    p.add_argument('--config', default=str(ROOT / 'configs/4bit/best.py'))
    p.add_argument('--model', default='vit_small_patch16_224', choices=[
        'vit_small_patch16_224', 'vit_base_patch16_224', 'deit_tiny_patch16_224',
        'deit_small_patch16_224', 'deit_base_patch16_224'])
    p.add_argument('--cache', default=str(ROOT/'checkpoints/fisher_probe/shared_images.pt'))
    p.add_argument('--basis-size', type=int, default=48)
    p.add_argument('--eval-size', type=int, default=16)
    p.add_argument('--ranks', type=int, nargs='+', default=[4, 8, 16])
    p.add_argument('--blocks', nargs='+', default=['0', '5', '11'], help='zero-based indices or all')
    p.add_argument('--epsilons', type=float, nargs='+', default=[0.0001, 0.001, 0.01])
    p.add_argument('--test-relative-norm', type=float, default=0.001)
    p.add_argument('--test-directions', type=int, default=3)
    p.add_argument('--temperature', type=float, default=20.)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output-root', default=str(ROOT/'checkpoints/fisher_probe/results'))
    return p


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_torch(path):
    # Only use your own trusted cache/checkpoint; compatible with older PyTorch.
    return torch.load(path, map_location='cpu', weights_only=False)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def timed(fn, device):
    sync(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
    started = time.perf_counter()
    result = fn()
    sync(device)
    stats = dict(seconds=time.perf_counter()-started,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                 incremental_peak_bytes=max(0, torch.cuda.max_memory_allocated(device)-baseline) if device.type == 'cuda' else None)
    return result, stats


def images_cache(args, model):
    from timm.data import resolve_data_config, create_transform
    from torchvision.datasets import ImageFolder
    config = resolve_data_config(model.default_cfg, model=model)
    transform = create_transform(**config, is_training=False)
    spec = dict(model=args.model, dataset=str(pathlib.Path(args.dataset).resolve()),
                seed=args.seed, basis_size=args.basis_size, eval_size=args.eval_size,
                transform=repr(transform), schema=1)
    path = pathlib.Path(args.cache)
    if path.exists():
        cached = load_torch(path)
        if cached['spec'] != spec:
            raise ValueError('Cache settings differ; use another --cache path, do not overwrite the shared cache')
    else:
        dataset = ImageFolder(str(pathlib.Path(args.dataset)/'train'), transform=transform)
        n = args.basis_size + args.eval_size
        if len(dataset) < n:
            raise ValueError('Not enough training images for disjoint basis/evaluation sets')
        indices = np.random.default_rng(args.seed).permutation(len(dataset))[:n].tolist()
        cached = dict(spec=spec, indices=indices,
                      paths=[os.path.relpath(dataset.samples[i][0], args.dataset) for i in indices],
                      images=torch.stack([dataset[i][0] for i in indices]))
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental concurrent overwrites.
        with open(path, 'xb') as f:
            torch.save(cached, f)
    images = cached['images']
    if len(images) != args.basis_size + args.eval_size or not torch.isfinite(images).all():
        raise ValueError('Invalid image cache')
    return images, dict(spec=spec, paths=cached['paths'], indices=cached['indices'],
                       tensor_sha256=hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest(),
                       file_sha256=sha_file(path))


def build_quant_model(args, model):
    from utils.wrap_net import wrap_modules_in_net
    spec = importlib.util.spec_from_file_location('probe_config', args.config)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.Config()
    bits = re.search(r'_w(\d+)_a(\d+)_', pathlib.Path(args.checkpoint).name)
    if bits and (int(bits[1]), int(bits[2])) != (cfg.w_bit, cfg.a_bit):
        raise ValueError('Checkpoint bit widths do not match the selected configuration')
    model = wrap_modules_in_net(model, cfg, reparam=False)
    state = load_torch(args.checkpoint)
    # Quantizer shapes are determined by calibration (e.g. token-wise scales).
    for name, parameter in model.named_parameters():
        if name in state and 'quantizer.' in name:
            parameter.data = state[name].clone()
    model.load_state_dict(state, strict=True)
    for m in model.modules():
        if hasattr(m, 'mode'):
            m.mode = 'raw'
            m.calibrated = True
        if hasattr(m, 'inited'):
            m.inited = True
        if hasattr(m, 'drop_prob'):
            m.drop_prob = 1.
        if hasattr(m, 'training_mode'):
            m.training_mode = False
        if hasattr(m, 'fused_attn'):
            m.fused_attn = False
    model.eval().requires_grad_(False)
    return model, vars(cfg)


def main(args):
    global torch, np
    import numpy as np
    import torch
    import torch.nn.functional as F
    import timm
    from utils.fisher_geometry import (energy_rank, error_basis, projected_fisher,
        response_jacobian, matrix_summary, quadratic_predictions)
    from utils.fisher_probe_model import capture_branch, make_suffix
    if min(args.basis_size, args.eval_size, args.test_directions, *args.ranks) < 1:
        raise ValueError('Sample sizes, ranks and test direction count must be positive')
    if max(args.ranks) > args.basis_size:
        raise ValueError('max rank cannot exceed basis-size')
    if any(not np.isfinite(x) or x <= 0 for x in [args.temperature, args.test_relative_norm, *args.epsilons]):
        raise ValueError('temperature and perturbation scales must be finite and positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; select --device cpu only for tiny tests')
    # No network download: all weights come from the calibrated checkpoint.
    model = timm.create_model(args.model, pretrained=False).eval()
    images, cache_metadata = images_cache(args, model)
    print('Shared cache:', args.cache, cache_metadata['tensor_sha256'], flush=True)
    if args.stage == 'prepare':
        return
    if 'calib' not in pathlib.Path(args.checkpoint).name:
        raise ValueError('Use a calibrated FP-weight checkpoint, not an optimized hard-rounded checkpoint')
    model, cfg = build_quant_model(args, model)
    model = model.to(device)
    blocks = list(range(len(model.blocks))) if args.blocks == ['all'] else [int(x) for x in args.blocks]
    if any(b < 0 or b >= len(model.blocks) for b in blocks) or len(set(blocks)) != len(blocks):
        raise ValueError('Invalid or duplicate block indices')
    output = pathlib.Path(args.output_root)/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(args=vars(args), config=cfg, cache=cache_metadata,
                    checkpoint_sha256=sha_file(args.checkpoint), torch=torch.__version__, timm=timm.__version__,
                    device=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                    coordinate_system='signed quantization-error PCA; per-branch local FP suffix',
                    source_sha256={str(p.relative_to(ROOT)):sha_file(p) for p in [pathlib.Path(__file__),
                        ROOT/'utils/fisher_geometry.py', ROOT/'utils/fisher_probe_model.py']})
    (output/'experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Results:', output, flush=True)
    start = time.perf_counter()

    def record(data):
        with open(output/'diagnostics.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, allow_nan=False) + '\n')

    for b in blocks:
        for branch in ('attn', 'mlp'):
            name = 'blocks.{}.{}'.format(b, branch)
            print('Collecting', name, flush=True)
            errors, contexts = [], []
            for i, image in enumerate(images):
                x, h, error, logits = capture_branch(model, b, branch, image[None].to(device))
                if i < args.basis_size:
                    errors.append(error.cpu())
                else:
                    contexts.append(tuple(t.cpu() for t in (x, h, error, logits)))
            basis, energies, numerical_rank = error_basis(torch.cat(errors), max(args.ranks))
            record(dict(event='basis', module=name, singular_values=energies.sqrt().tolist(),
                        numerical_rank=numerical_rank, retained_rank=len(basis),
                        error_rank95=energy_rank(energies, 0.95),
                        error_condition_positive=float((energies[0]/energies[numerical_rank-1]).sqrt()),
                        retained_energy=float(energies[:len(basis)].sum()/energies.sum())))
            basis = basis.to(device)
            reference_matrices = []
            fd_matrices = {str(eps):[] for eps in args.epsilons} if args.stage in ('forward','both') else {}
            for i, context in enumerate(contexts):
                x, h, error, logits = [t.to(device) for t in context]
                suffix = make_suffix(model, b, branch, x)
                with torch.no_grad():
                    torch.testing.assert_close(suffix(h), logits, rtol=1e-4, atol=1e-4)
                norm = max(float(h.norm()), 1e-12)
                def response(coeff):
                    delta = (coeff @ basis).reshape_as(h)
                    return (suffix(h + delta)/args.temperature).squeeze(0)
                with torch.no_grad():
                    p = F.softmax(response(torch.zeros(len(basis), device=device)).double(), dim=-1)
                # Warm both forward and double-backward paths before timed trials.
                if i == 0:
                    response_jacobian(response, 1, device, 'autograd') if len(basis) == 1 else torch.autograd.functional.jvp(
                        response, torch.zeros(len(basis),device=device), torch.ones(len(basis),device=device))
                vref, ref_cost = timed(lambda: response_jacobian(response, len(basis), device, 'autograd'), device)
                reference = projected_fisher(vref, p)
                reference_matrices.append(reference.cpu())
                fd_results = {}
                for eps in fd_matrices:
                    vfd, cost = timed(lambda: response_jacobian(response, len(basis), device,
                        'finite_difference', float(eps)*norm), device)
                    ffd = projected_fisher(vfd, p)
                    fd_matrices[eps].append(ffd.cpu())
                    fd_results[eps] = ffd
                    record(dict(event='forward_comparison', module=name, eval_index=i, epsilon=float(eps),
                        step=float(eps)*norm, rank=len(basis), suffix_forward_evaluations=2*len(basis),
                        relative_fisher_error=float((ffd-reference).norm()/reference.norm().clamp_min(1e-30)),
                        relative_response_error=float((vfd-vref).norm()/vref.norm().clamp_min(1e-30)), **cost))
                record(dict(event='reference', module=name, eval_index=i, rank=len(basis),
                    jvp_calls=len(basis), actual_error_norm=float(error.norm()),
                    error_projection_energy=float((error.flatten() @ basis.T).square().sum()/error.square().sum().clamp_min(1e-30)),
                    **ref_cost))
                # A held-out image's real quantization error was NOT used to fit
                # the basis. Separate subspace coverage from curvature accuracy.
                coeff_error = error.flatten() @ basis.T
                with torch.no_grad():
                    actual_projected = F.kl_div(F.log_softmax(response(coeff_error).double(),-1),p,reduction='sum').clamp_min(0)
                    actual_full = F.kl_div(F.log_softmax((suffix(h+error)/args.temperature).double().squeeze(0),-1),p,reduction='sum').clamp_min(0)
                error_predictions = {'autograd':quadratic_predictions(reference,coeff_error)}
                for eps,ffd in fd_results.items():
                    error_predictions[eps] = quadratic_predictions(ffd,coeff_error)
                record(dict(event='quantization_error',module=name,eval_index=i,rank=len(basis),
                            actual_full_kl=float(actual_full),actual_kl=float(actual_projected),
                            predictions=error_predictions))
                generator = torch.Generator(device='cpu').manual_seed(args.seed + b*1000 + i*17)
                for rank in sorted(set(min(r,len(basis)) for r in args.ranks)):
                    fr = reference[:rank,:rank]
                    record(dict(event='spectrum', module=name, eval_index=i, rank=rank, **matrix_summary(fr)))
                    for trial in range(args.test_directions):
                        a = torch.randn(rank,generator=generator).to(device)
                        a = a/a.norm()*args.test_relative_norm*norm
                        coeff = torch.zeros(len(basis), device=device)
                        coeff[:rank] = a
                        with torch.no_grad():
                            actual = F.kl_div(F.log_softmax(response(coeff).double(),-1),p,reduction='sum').clamp_min(0)
                        predictions = {'autograd':quadratic_predictions(fr,a)}
                        for eps, ffd in fd_results.items():
                            predictions[eps] = quadratic_predictions(ffd[:rank,:rank],a)
                        record(dict(event='heldout_direction', module=name, eval_index=i, rank=rank,
                            trial=trial, actual_kl=float(actual), predictions=predictions))
                print(name, 'evaluation', i+1, '/', len(contexts), flush=True)
            torch.save(dict(basis=basis.cpu(), error_gram_eigenvalues=energies,
                            reference=torch.stack(reference_matrices),
                            finite_difference={k:torch.stack(v) for k,v in fd_matrices.items()}), output/(name+'.pt'))
            del contexts, errors, basis
    sync(device)
    (output/'complete.json').write_text(json.dumps(dict(status='complete',seconds=time.perf_counter()-start)), encoding='utf-8')
    print('Completed:', output, flush=True)


if __name__ == '__main__':
    main(parser().parse_args())
