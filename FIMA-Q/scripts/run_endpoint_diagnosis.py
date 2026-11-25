"""Diagnose saved ViT-S W3A3 MR endpoints and compare two FC2 scale-search policies."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

from run_headwise_experiments import ROOT, dataset_path, gpu_lock, job_lock


REFERENCES = {
    'uniform': '20261113_130804_lsvit_mr_relu_seed3407_a1b23ee46e83',
    'residual': '20261113_130900_lsvit_mr_relu_seed3407_42a305a3eb20',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', choices=REFERENCES, required=True)
    parser.add_argument('--stage', choices=('diagnose', 'validate'), required=True)
    parser.add_argument('--dataset', default='../imagenet_fimaq')
    parser.add_argument('--fp-checkpoint', help='Original ViT-S weights. By default use local weights, then timm.')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--gpu-slots', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--probe-batch-size', type=int, default=8)
    parser.add_argument('--val-batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if min(args.probe_batch_size, args.val_batch_size) < 1 or args.num_workers < 0:
        parser.error('Batch sizes must be positive and --num-workers must be nonnegative')

    record = ROOT / 'checkpoints/quant_result' / REFERENCES[args.baseline] / 'method_result.json'
    if not record.is_file():
        parser.error(f'Sync this original run record to the server: {record}')
    meta = read_json(record)
    original = meta['args']
    if (meta['status'] != 'complete' or meta['control'] != 'fixed' or meta['mr_variant'] != 'relu'
            or original['model'] != 'vit_small' or original['seed'] != 3407
            or original.get('ls_mr_budget', 'uniform') != args.baseline
            or not original['headwise_hadamard'] or original['headwise_mode'] != 'all'
            or original['qk_joint_variant'] != 'none'
            or (meta['cfg']['w_bit'], meta['cfg']['a_bit'], meta['cfg']['optim_metric']) != (3, 3, 'lsvit')):
        parser.error(f'Unexpected source recipe in {record}')
    checkpoint = record.with_name(PurePosixPath(meta['checkpoint_after']).name)
    data = dataset_path(args.dataset)
    fp_candidates = ('checkpoints/vit_raw/model.safetensors', 'checkpoints/vit_raw/vit_small_patch16_224.bin')
    fp_path = dataset_path(args.fp_checkpoint) if args.fp_checkpoint else next(
        (ROOT / path for path in fp_candidates if (ROOT / path).is_file()), None)
    print(f'Source {args.baseline}: {meta["after"]["top1"]:.3f}%\nCheckpoint: {checkpoint}\nDataset: {data}')
    print(f'FP teacher: {fp_path or "timm pretrained weights (fingerprint checked)"}')
    if args.dry_run:
        print(f'Checkpoint present: {checkpoint.is_file()}')
        return
    if not checkpoint.is_file():
        parser.error(f'Optimized checkpoint missing: {checkpoint}. Restore the original .pth on the server.')
    if fp_path is not None and not fp_path.is_file():
        parser.error(f'FP weights missing: {fp_path}')
    if not (data / ('train' if args.stage == 'diagnose' else 'val')).is_dir():
        parser.error(f'Missing ImageNet split under {data}')

    # Set CUDA visibility before importing torch or constructing the network.
    with job_lock(f'endpoint_diagnosis_{args.baseline}'), gpu_lock(args.gpu, args.gpu_slots) as uuid:
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        sys.path.insert(0, str(ROOT))
        from endpoint_diagnosis_impl import diagnose, validate_plan
        from utils.lsvit_pair import file_sha

        sources = ('scripts/run_endpoint_diagnosis.py', 'scripts/endpoint_diagnosis_impl.py',
                   'utils/wrap_net.py', 'utils/datasets.py', 'utils/test_utils.py', 'quantizers/uniform.py',
                   'quantizers/adaround.py', 'quant_layers/linear.py', 'quant_layers/matmul.py', 'quant_layers/conv.py')
        identity = dict(baseline=args.baseline, checkpoint_sha256=file_sha(checkpoint), record_sha256=file_sha(record),
                        dataset=str(data), probe_batch_size=args.probe_batch_size,
                        sources={name: file_sha(ROOT / name) for name in sources})
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        directory = ROOT / 'checkpoints/endpoint_diagnosis' / f'seed3407_{args.baseline}_{key}'
        directory.mkdir(parents=True, exist_ok=True)
        print(f'Output: {directory}', flush=True)
        plan_file = directory / 'diagnosis.json'
        if args.stage == 'diagnose':
            if plan_file.is_file():
                report = read_json(plan_file)
                if report['identity'] != identity:
                    raise RuntimeError('Saved diagnosis identity mismatch')
                print(f'Existing diagnosis: {report["decision"]}', flush=True)
                return
            diagnose(args, meta, checkpoint, data, fp_path, identity, plan_file)
        else:
            if not plan_file.is_file():
                parser.error('Run --stage diagnose with the same baseline, dataset and probe batch size first')
            plan = read_json(plan_file)
            if plan['identity'] != identity:
                raise RuntimeError('Saved diagnosis identity mismatch')
            if not plan['ready_for_validation']:
                print(f'Validation skipped: {plan["decision"]}', flush=True)
                return
            validate_plan(args, meta, checkpoint, data, fp_path, plan, directory / 'validation.json')


if __name__ == '__main__':
    main()
