"""Validate fixed-step forward-only Fisher on the true quantization error.

This is a diagnostic only: it does not train a model or write a checkpoint.
It compares one complete error direction with a fixed channel-group
decomposition, using one global finite-difference epsilon for every module.
"""
import argparse
import json
import os
import pathlib
import random
import sys
import time
from datetime import datetime


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default=str(ROOT.parent / 'imagenet_fimaq'))
    p.add_argument('--checkpoint', default=str(ROOT / 'checkpoints/quant_result/20260916_1818/vit_small_w4_a4_calibsize_128_mse.pth'))
    p.add_argument('--config', default=str(ROOT / 'configs/4bit/best.py'))
    p.add_argument('--model', default='vit_small_patch16_224', choices=[
        'vit_small_patch16_224', 'vit_base_patch16_224', 'deit_tiny_patch16_224',
        'deit_small_patch16_224', 'deit_base_patch16_224'])
    p.add_argument('--cache', default=str(ROOT / 'checkpoints/fisher_probe/fixed_images.pt'))
    p.add_argument('--images', type=int, default=32)
    p.add_argument('--blocks', nargs='+', default=['0', '3', '5', '8', '11'])
    p.add_argument('--epsilon', type=float, default=0.001)
    p.add_argument('--groups', type=int, default=4)
    p.add_argument('--variants', nargs='+', choices=['single', 'grouped'],
                   default=['single', 'grouped'])
    p.add_argument('--scales', type=float, nargs='*', default=[])
    p.add_argument('--temperature', type=float, default=20.)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output-root', default=str(ROOT / 'checkpoints/fisher_probe/fixed_results'))
    p.add_argument('--output-dir', type=pathlib.Path)
    return p


def main(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import timm
    import scripts.probe_fisher as shared
    from utils.fisher_geometry import (fixed_error_directions, projected_fisher,
        quadratic_predictions, response_jacobian)
    from utils.fisher_probe_model import capture_branch, make_suffix

    if args.images < 1 or args.groups < 1:
        raise ValueError('images and groups must be positive')
    if not np.isfinite(args.epsilon) or args.epsilon <= 0:
        raise ValueError('epsilon must be finite and positive')
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if any(not np.isfinite(scale) or scale <= 0 for scale in args.scales):
        raise ValueError('scales must be finite and positive')
    if args.scales and not any(abs(scale - 1.) < 1e-12 for scale in args.scales):
        raise ValueError('amplitude sweep must include scale 1.0 for the secant anchor')

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

    # Reuse the already-tested checkpoint and deterministic image-cache helpers.
    shared.torch, shared.np = torch, np
    model = timm.create_model(args.model, pretrained=False).eval()
    cache_args = argparse.Namespace(**vars(args), basis_size=0, eval_size=args.images)
    images, cache_metadata = shared.images_cache(cache_args, model)
    print('Shared cache:', args.cache, cache_metadata['tensor_sha256'], flush=True)
    if 'calib' not in pathlib.Path(args.checkpoint).name:
        raise ValueError('Use a calibrated FP-weight checkpoint, not an optimized hard-rounded checkpoint')
    model, cfg = shared.build_quant_model(args, model)
    model = model.to(device)

    blocks = list(range(len(model.blocks))) if args.blocks == ['all'] else [int(x) for x in args.blocks]
    if any(b < 0 or b >= len(model.blocks) for b in blocks) or len(set(blocks)) != len(blocks):
        raise ValueError('Invalid or duplicate block indices')
    if args.groups > model.embed_dim:
        raise ValueError('groups cannot exceed the model channel count')

    output = args.output_dir or pathlib.Path(args.output_root) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = [pathlib.Path(__file__), ROOT / 'scripts/probe_fisher.py',
               ROOT / 'utils/fisher_geometry.py', ROOT / 'utils/fisher_probe_model.py']
    metadata = dict(
        args={**vars(args), 'output_dir': str(args.output_dir) if args.output_dir else None},
        config=cfg,
        cache=cache_metadata,
        checkpoint_sha256=shared.sha_file(args.checkpoint),
        torch=torch.__version__,
        timm=timm.__version__,
        device=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        method='fixed epsilon; true quantization-error direction; fixed contiguous output-channel groups',
        source_sha256={str(path.relative_to(ROOT)): shared.sha_file(path) for path in sources},
    )
    (output / 'experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Results:', output, flush=True)
    started = time.perf_counter()

    def record(data):
        with open(output / 'diagnostics.jsonl', 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(data, allow_nan=False) + '\n')

    for block_index in blocks:
        for branch in ('attn', 'mlp'):
            name = 'blocks.{}.{}'.format(block_index, branch)
            print('Collecting', name, flush=True)
            saved = {variant: {'reference': [], 'finite_difference': []}
                     for variant in args.variants}
            for image_index, image in enumerate(images):
                x, h, error, logits = capture_branch(model, block_index, branch, image[None].to(device))
                suffix = make_suffix(model, block_index, branch, x)
                with torch.no_grad():
                    torch.testing.assert_close(suffix(h), logits, rtol=1e-4, atol=1e-4)
                    baseline_logits = (suffix(h) / args.temperature).squeeze(0)
                    probabilities = F.softmax(baseline_logits.double(), dim=-1)
                    actual_kl = F.kl_div(
                        F.log_softmax((suffix(h + error) / args.temperature).double().squeeze(0), dim=-1),
                        probabilities,
                        reduction='sum',
                    ).clamp_min(0)
                    sweep_actual = {}
                    for scale in args.scales:
                        sweep_actual[str(scale)] = float(F.kl_div(
                            F.log_softmax((suffix(h + scale * error) / args.temperature).double().squeeze(0), dim=-1),
                            probabilities,
                            reduction='sum',
                        ).clamp_min(0))
                activation_norm = max(float(h.norm()), 1e-12)

                configured = [('single', 1), ('grouped', args.groups)]
                for variant, group_count in (item for item in configured if item[0] in args.variants):
                    directions, coefficients = fixed_error_directions(error, group_count)
                    directions = directions.to(device)
                    coefficients = coefficients.to(device)
                    reconstructed = (coefficients @ directions).reshape_as(error)
                    reconstruction_error = float((reconstructed - error).norm() / error.norm().clamp_min(1e-30))

                    def response(value):
                        delta = (value @ directions).reshape_as(h)
                        return (suffix(h + delta) / args.temperature).squeeze(0)

                    rank = len(directions)
                    if image_index == 0:
                        response_jacobian(response, rank, device, 'autograd')
                    reference_response, reference_cost = shared.timed(
                        lambda: response_jacobian(response, rank, device, 'autograd'), device)
                    step = args.epsilon * activation_norm
                    forward_response, forward_cost = shared.timed(
                        lambda: response_jacobian(response, rank, device, 'finite_difference', step), device)
                    reference = projected_fisher(reference_response, probabilities)
                    forward = projected_fisher(forward_response, probabilities)
                    saved[variant]['reference'].append(reference.cpu())
                    saved[variant]['finite_difference'].append(forward.cpu())

                    reference_all = quadratic_predictions(reference, coefficients)
                    forward_all = quadratic_predictions(forward, coefficients)
                    # Fixed ablations only: retain the complete small matrix and
                    # its diagonal. Do not select a data-dependent low rank.
                    reference_predictions = {key: reference_all[key] for key in ('full', 'diagonal')}
                    forward_predictions = {key: forward_all[key] for key in ('full', 'diagonal')}
                    record(dict(
                        event='direction_set', module=name, image_index=image_index,
                        variant=variant, rank=rank, groups=group_count,
                        quantization_error_norm=float(error.norm()),
                        reconstruction_relative_error=reconstruction_error,
                        coefficients=coefficients.detach().cpu().tolist(),
                    ))
                    record(dict(
                        event='reference', module=name, image_index=image_index,
                        variant=variant, rank=rank, jvp_calls=rank, **reference_cost,
                    ))
                    record(dict(
                        event='forward_comparison', module=name, image_index=image_index,
                        variant=variant, rank=rank, epsilon=args.epsilon, step=step,
                        suffix_forward_evaluations=2 * rank,
                        relative_fisher_error=float((forward - reference).norm() / reference.norm().clamp_min(1e-30)),
                        relative_response_error=float((forward_response - reference_response).norm() / reference_response.norm().clamp_min(1e-30)),
                        **forward_cost,
                    ))
                    record(dict(
                        event='directional_prediction', module=name, image_index=image_index,
                        variant=variant, rank=rank, actual_kl=float(actual_kl),
                        predictions={'autograd': reference_predictions, 'finite_difference': forward_predictions},
                    ))
                    if variant == 'single' and args.scales:
                        anchor = sweep_actual[next(key for key in sweep_actual if abs(float(key) - 1.) < 1e-12)]
                        for scale in args.scales:
                            record(dict(
                                event='amplitude_sweep', module=name, image_index=image_index,
                                variant=variant, epsilon=args.epsilon, scale=scale,
                                actual_kl=sweep_actual[str(scale)],
                                predictions={
                                    'local_fisher': scale * scale * forward_predictions['full'],
                                    'forward_secant': scale * scale * anchor,
                                },
                            ))
                print(name, 'image', image_index + 1, '/', len(images), flush=True)
            torch.save(saved, output / (name + '.pt'))

    shared.sync(device)
    (output / 'complete.json').write_text(json.dumps(
        dict(status='complete', seconds=time.perf_counter() - started)), encoding='utf-8')
    print('Completed:', output, flush=True)


if __name__ == '__main__':
    main(parser().parse_args())
