"""Shared snapshot and first-branch verification for paired LS experiments."""

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time

import torch


def fingerprint(value):
    digest = hashlib.sha256()

    def update(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            update((str(tensor.dtype), tuple(tensor.shape)))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            update(tuple(item.items()))
        elif isinstance(item, (list, tuple)):
            digest.update(f'[{len(item)}]'.encode())
            for part in item:
                update(part)
        else:
            digest.update(repr(item).encode() + b'\0')

    update(value)
    return digest.hexdigest()


def calibration_hashes(log_path):
    matches = re.findall(r'Calibration cache: count=(\d+) sample_sha256=([a-f0-9]+) tensor_sha256=([a-f0-9]+)',
                         Path(log_path).read_text(encoding='utf-8'))
    return [list(row) for row in matches]


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def save_snapshot(directory, checkpoint, model_sha, data_hashes, seed):
    if [row[0] for row in data_hashes] != ['128']:
        raise RuntimeError('Missing 128-image calibration fingerprint')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'calibrated.pth'
    sha = file_sha(checkpoint)
    Path(checkpoint).replace(target)
    write_json(directory / 'snapshot.json', dict(seed=seed, fp_model_sha256=model_sha,
        checkpoint_sha256=sha, calibration_hashes=data_hashes))


def read_snapshot(directory, checkpoint, model_sha, seed):
    directory = Path(directory)
    if Path(checkpoint).resolve() != directory.resolve() / 'calibrated.pth':
        raise RuntimeError('Paired branches must use the shared calibrated checkpoint')
    snapshot = json.loads((directory / 'snapshot.json').read_text(encoding='utf-8'))
    if (snapshot['seed'] != seed or snapshot['fp_model_sha256'] != model_sha
            or snapshot['checkpoint_sha256'] != file_sha(checkpoint)):
        raise RuntimeError('Shared calibration snapshot fingerprint mismatch')
    return snapshot


def verify_first_branch(directory, mode, signature, timeout=1200):
    """Publish an immutable signature and wait for the paired branch's signature."""
    directory = Path(directory)
    own = directory / f'{mode}.json'
    peer_mode = {'mix-equal': 'mix-adaptive', 'mix-adaptive': 'mix-equal'}[mode]
    peer = directory / f'{peer_mode}.json'
    if own.is_file():
        if json.loads(own.read_text(encoding='utf-8')) != signature:
            raise RuntimeError(f'First-branch state differs from the previous attempt in {directory}')
    else:
        write_json(own, signature)
    logging.info('First LS branch signature written: %s; waiting for %s', own, peer_mode)
    deadline = time.monotonic() + timeout
    while not peer.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Timed out waiting for {peer_mode} at its first LS probe')
        time.sleep(2)
    other = json.loads(peer.read_text(encoding='utf-8'))
    if other != signature:
        changed = sorted(k for k in other.keys() | signature.keys() if other.get(k) != signature.get(k))
        raise RuntimeError(f'Paired LS branches differ before adaptation: {", ".join(changed)}')
    logging.info('[PAIR VERIFIED] Shared calibration and first LS branch match')
