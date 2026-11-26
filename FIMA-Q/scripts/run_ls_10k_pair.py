"""Compare 10k formal reconstruction steps with the historical 67.492% run."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tempfile
import zipfile

from run_headwise_experiments import ROOT, dataset_path, gpu_lock, job_lock


REFERENCE = ROOT / 'checkpoints/quant_result/20261111_194420_lsvit_mr_relu_seed3408_a4008f61fe80/method_result.json'
REVISION = '404f9eee16d89ef1f88b6ae225c7e7cb2644a648'
STEPS = 10000
SOURCES = ('test_quant.py', 'scripts/run_method_adaptations.py', 'utils', 'quantizers', 'quant_layers',
           'configs', 'scripts/run_lsvit_controls.py', 'scripts/run_lsvit_mr.py')
CALIBRATION = re.compile(r'Calibration cache: count=(\d+) sample_sha256=([a-f0-9]{64}) tensor_sha256=([a-f0-9]{64})')
ELAPSED = re.compile(r'^Block reconstruction elapsed seconds: (\d+(?:\.\d+)?)$', re.M)
BLOCKS = {'patch_embed', *(f'blocks.{i}' for i in range(12)), 'head'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def archived_source(reference):
    prefix = 'FIMA-Q/'
    raw = subprocess.check_output(['git', '-C', str(ROOT.parent), 'archive', '--format=zip', REVISION,
                                   *[prefix + p for p in SOURCES]])
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        sources = {name[len(prefix):]: archive.read(name) for name in archive.namelist()
                   if name.startswith(prefix) and name.endswith('.py') and not name.endswith('/')}

    def content(path):
        return sources[path].decode('utf-8').replace('\r\n', '\n').replace('\r', '\n').encode()

    digest = hashlib.sha256()
    paths = list(SOURCES[:2])
    for folder in SOURCES[2:6]:
        paths.extend(sorted(path for path in sources if path.startswith(folder + '/')))
    for path in paths:
        digest.update(path.encode())
        digest.update(content(path))
    digest = hashlib.sha256(digest.hexdigest().encode() + content('scripts/run_lsvit_controls.py'))
    if (digest.hexdigest() != reference['launch']['code_sha256']
            or hashlib.sha256(content('scripts/run_lsvit_mr.py')).hexdigest() != reference['launch']['mr_runner_sha256']):
        raise ValueError('Archived code does not match the historical 20k launch fingerprints')
    return sources


def install_source(work, sources):
    target = work / f'source_{REVISION[:7]}'
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=work) as staging:
            source = Path(staging) / 'source'
            for name, content in sources.items():
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            (source / '.gitignore').write_text('*\n', encoding='utf-8')
            # Keep historical test_quant.py output paths, routed to the current experiment results.
            (source / 'checkpoints').symlink_to(ROOT / 'checkpoints', target_is_directory=True)
            source.rename(target)
    saved = {p.relative_to(target).as_posix() for folder in ('utils', 'quantizers', 'quant_layers', 'configs', 'scripts')
             for p in (target / folder).rglob('*.py')}
    if (saved != set(sources) - {'test_quant.py'}
            or any(not (target / path).is_file() or (target / path).read_bytes() != content
                   for path, content in sources.items())
            or (target / 'checkpoints').resolve() != (ROOT / 'checkpoints').resolve()):
        raise ValueError(f'Isolated source tree differs from the archived version: {target}')
    return target


def historical_seconds(path):
    matches = ELAPSED.findall(path.read_text(encoding='utf-8'))
    if len(matches) != 1 or float(matches[0]) <= 0:
        raise ValueError(f'Expected one positive formal reconstruction duration in {path}')
    return float(matches[0])


def set_option(command, name, value):
    if name not in command:
        command.extend((name, str(value)))
    else:
        command[command.index(name) + 1] = str(value)


def make_command(reference, data, mr_dir, fp_path):
    command = [sys.executable, *reference['launch']['command']]
    set_option(command, '--dataset', data)
    set_option(command, '--ls-mr-dir', mr_dir)
    set_option(command, '--recon-iters', STEPS)
    if fp_path is not None:
        set_option(command, '--fp-checkpoint', fp_path)
    return command


def launch(command, source, env, expected_hashes):
    # Check both augmented image hashes before reconstruction; stream progress in the terminal.
    hashes = []
    with subprocess.Popen(command, cwd=source, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1) as proc:
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                found = CALIBRATION.search(line)
                if found:
                    hashes.append(list(found.groups()))
                    if hashes != expected_hashes[:len(hashes)]:
                        raise RuntimeError('Calibration images or augmented tensors differ from the 20k run')
                if 'guided block reconstruction' in line and ' - start ' in line and hashes != expected_hashes:
                    raise RuntimeError('Both historical calibration fingerprints are required before reconstruction')
            code = proc.wait()
            if code != 0:
                raise subprocess.CalledProcessError(code, command)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


def check_result(record, result, payload, reference):
    metrics = result.get('after', {})
    if (result.get('status') != 'complete' or result.get('launch') != payload
            or result.get('cfg') != dict(reference['cfg'], recon_iters=STEPS)
            or any(result.get(k) != reference[k] for k in ('control', 'fp_model_sha256',
                   'calibration_hashes', 'mr_variant', 'mr_weights_sha256', 'qk_blocks'))
            or not all(isinstance(metrics.get(k), (float, int)) and math.isfinite(metrics[k])
                       for k in ('top1', 'top5', 'loss'))):
        raise ValueError(f'Completed run does not match the historical pair: {record}')
    log = record.with_name('output.log')
    finished = re.findall(r'^finished reconstructing ([\w.]+)\.$', log.read_text(encoding='utf-8'), re.M)
    if set(finished) != BLOCKS or len(finished) != len(BLOCKS):
        raise ValueError(f'Missing reconstructed modules in {log}')
    return historical_seconds(log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='../imagenet_fimaq')
    parser.add_argument('--fp-checkpoint', help='Explicit local FP weights; tensor hash must match AugReg ViT-S')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--gpu-slots', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    reference = read_json(REFERENCE)
    if (reference['status'] != 'complete' or reference['control'] != 'fixed' or reference['after']['top1'] != 67.492
            or reference['args']['seed'] != 3408 or reference['mr_variant'] != 'relu'
            or reference['args'].get('ls_mr_budget', 'uniform') != 'uniform'
            or reference['cfg']['recon_iters'] != 20000):
        parser.error('Unexpected historical reference; expected the 67.492% 20k fixed-LS run')
    base_seconds = historical_seconds(REFERENCE.with_name('output.log'))
    data = dataset_path(args.dataset)
    name = PurePosixPath(reference['args']['ls_mr_dir']).name
    mr_dir = ROOT / 'checkpoints/lsvit_controls/mr' / name
    mr_report, mr_weights = mr_dir / 'mr_screen.json', mr_dir / 'relu_mlp.pth'
    fp_candidates = [ROOT / 'checkpoints/vit_raw' / name for name in ('model.safetensors', 'vit_small_patch16_224.bin')]
    fp_path = dataset_path(args.fp_checkpoint) if args.fp_checkpoint else next((p for p in fp_candidates if p.is_file()), None)
    command = make_command(reference, data, mr_dir, fp_path)
    try:
        sources = archived_source(reference)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        parser.error(f'Cannot verify historical Git revision {REVISION}: {error}')
    print(f'Reference: 67.492%, 20k, formal reconstruction {base_seconds / 60:.1f} min', flush=True)
    print(f'Historical source fingerprints verified at {REVISION[:7]}\nMR weights: {mr_weights}', flush=True)
    if args.dry_run:
        print(f'MR weights present: {mr_weights.is_file()}\n{shlex.join(command)}')
        return
    if not all((data / part).is_dir() for part in ('train', 'val')):
        parser.error(f'Missing ImageNet train/val under {data}')
    if fp_path is not None and not fp_path.is_file():
        parser.error(f'FP checkpoint missing: {fp_path}')
    if not mr_weights.is_file() or not mr_report.is_file():
        parser.error(f'Restore original MR files: {mr_weights} and {mr_report}')
    screen = read_json(mr_report)
    if (screen['status'] != 'complete' or screen['seed'] != 3408
            or screen.get('budget', 'uniform') != 'uniform' or screen['steps'] != 3000
            or screen['fp_model_sha256'] != reference['fp_model_sha256']
            or screen['data_hashes'] != reference['calibration_hashes'][1:]
            or screen['checkpoint_sha256'] != reference['mr_weights_sha256']
            or file_sha(mr_weights) != reference['mr_weights_sha256']):
        parser.error('The saved MR screen or weight hash does not match the 20k run')

    try:
        versions = {package: importlib.metadata.version(package) for package in reference['launch']['dependency_versions']}
    except importlib.metadata.PackageNotFoundError as error:
        parser.error(f'Missing dependency {error.name}. Activate the original fimaq environment.')
    if versions != reference['launch']['dependency_versions']:
        parser.error(f'Dependencies differ from the reference. Expected {reference["launch"]["dependency_versions"]}; '
                     f'found {versions}. Activate the original fimaq environment.')
    payload = dict(reference_sha256=file_sha(REFERENCE), source_revision=REVISION, command=command[1:],
                   code_sha256=reference['launch']['code_sha256'], dependency_versions=versions,
                   mr_weights_sha256=reference['mr_weights_sha256'], fp_model_sha256=reference['fp_model_sha256'])
    tag = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    run_name = f'ls10k_seed3408_{tag}'
    work = ROOT / 'checkpoints/ls_step_budget'
    work.mkdir(parents=True, exist_ok=True)
    with job_lock('ls10k_seed3408'):
        source = install_source(work, sources)

        def completed():
            return sorted((ROOT / 'checkpoints/quant_result').glob(f'*_{run_name}/method_result.json'))

        records = completed()
        if not records:
            with gpu_lock(args.gpu, args.gpu_slots) as uuid:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                           FIMAQ_RUN_NAME=run_name, FIMAQ_CAPTURE_CALIBRATION_HASH='1',
                           FIMAQ_METHOD_LAUNCH=json.dumps(payload))
                launch(command, source, env, reference['calibration_hashes'])
            records = completed()
        if not records:
            raise RuntimeError('The process returned without a completed method_result.json')
        record = records[-1]
        result = read_json(record)
        seconds = check_result(record, result, payload, reference)
        delta = result['after']['top1'] - reference['after']['top1']
        saving = (base_seconds - seconds) / base_seconds
        summary = dict(status='complete', seed=3408, top1_delta_pp=delta, within_0_2_pp_loss=delta >= -0.200000001,
                       reconstruction_time_saved_fraction=saving, timing_note='Wall times use a historical reference and may differ in GPU load.',
                       run_20k=dict(source=str(REFERENCE), metrics=reference['after'], reconstruction_seconds=base_seconds),
                       run_10k=dict(source=str(record), metrics=result['after'], reconstruction_seconds=seconds),
                       fingerprints=payload, calibration_hashes=result['calibration_hashes'])
        target = work / f'summary_seed3408_{tag}.json'
        temporary = target.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        temporary.replace(target)
        print(f'20k: 67.492%, {base_seconds / 60:.1f} min; 10k: {result["after"]["top1"]:.3f}%, {seconds / 60:.1f} min', flush=True)
        print(f'Top-1 delta {delta:+.3f} pp, observed reconstruction time saved {saving:.1%}\nSummary: {target}', flush=True)


if __name__ == '__main__':
    main()
