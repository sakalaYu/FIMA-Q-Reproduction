"""Prepare the 1,152 SUN397 images needed by the formal-budget experiment."""

import argparse
from bisect import bisect_left
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from tempfile import NamedTemporaryFile


REPO = 'tanganke/sun397'
REVISION = '9c4afd6'
TRAIN_ROWS = 19850
TRAIN_SHARDS = 18
COUNT = 1024 + 128
SAMPLE_SEED = 3408


def download_shard(filename, scratch):
    path = scratch / filename
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.parquet.part')
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
    url = f'{endpoint}/datasets/{REPO}/resolve/{REVISION}/{filename}'
    subprocess.run([
        'curl', '-4', '--fail', '--location', '--retry', '2', '--retry-delay', '2',
        '--connect-timeout', '15', '--continue-at', '-', '--output', str(partial), url,
    ], check=True)
    partial.replace(path)
    return path


def atomic_write(path, data):
    with NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
        stream.write(data)
        staged = Path(stream.name)
    staged.replace(path)


def save_json(path, data):
    atomic_write(path, (json.dumps(data, indent=2) + '\n').encode())


def subset_id():
    rows = sorted(random.Random(SAMPLE_SEED).sample(range(TRAIN_ROWS), COUNT))
    return dict(repo=REPO, revision=REVISION, split='train', population=TRAIN_ROWS, sample_seed=SAMPLE_SEED, rows=rows)


def extract_shard(path, destination, start, selected):
    import pyarrow.parquet as pq

    source = pq.ParquetFile(path)
    try:
        stop = start + source.metadata.num_rows
        indices = selected[bisect_left(selected, start):bisect_left(selected, stop)]
        images = []
        for batch in source.iter_batches(batch_size=32, columns=['image'], use_threads=False):
            end = start + batch.num_rows
            for row in indices[bisect_left(indices, start):bisect_left(indices, end)]:
                record = batch.column(0)[row - start].as_py()
                value = record['bytes']
                if not value:
                    raise ValueError(f'SUN397 row {row} has no embedded image')
                # Keep the original bytes. Pillow reads the image format from its header.
                name = f'sun_{row:06}.jpg'
                atomic_write(destination / name, value)
                images.append(dict(row=row, file=name, sha256=hashlib.sha256(value).hexdigest()))
            start = end
        if start != stop or len(images) != len(indices):
            raise RuntimeError(f'Unexpected parquet row count in {path}')
        return stop, images
    finally:
        source.close()


def verify_output(destination, state):
    images = [image for shard in state['shards'] for image in shard['images']]
    if [image['row'] for image in images] != state['identity']['rows']:
        raise RuntimeError('The prepared SUN397 rows do not match the sampling manifest')
    if {path.name for path in destination.glob('*.jpg')} != {image['file'] for image in images}:
        raise RuntimeError('The SUN397 output directory contains missing or extra JPEG files')
    for image in images:
        if hashlib.sha256((destination / image['file']).read_bytes()).hexdigest() != image['sha256']:
            raise RuntimeError(f'SUN397 image hash differs: {image["file"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path.home() / 'autodl-tmp/SUN397_train_subset')
    parser.add_argument('--tools-dir', type=Path, help='Optional separate directory containing PyArrow')
    parser.add_argument('--local-shards', type=Path, help='Read parquet files from the data/ subdirectory')
    args = parser.parse_args()
    if args.tools_dir:
        sys.path.insert(0, str(args.tools_dir.expanduser().resolve()))
    destination = args.output.expanduser().resolve()
    progress = destination / 'sun397_subset.json'
    identity = subset_id()
    state = dict(identity=identity, shards=[], complete=False)
    if progress.is_file():
        state = json.loads(progress.read_text(encoding='utf-8'))
        if state['identity'] != identity:
            parser.error(f'Different subset manifest exists in {destination}')
        if state['complete']:
            verify_output(destination, state)
            print(f'SUN397 subset ready: {destination} ({COUNT} images)', flush=True)
            return
    elif destination.exists() and any(destination.iterdir()):
        parser.error(f'Use an empty output directory: {destination}')
    try:
        import pyarrow.parquet
    except ImportError as error:
        parser.error(f'Install PyArrow in your tools directory and use the fimaq environment: {error}')

    # The split layout is fixed at REVISION. No repository listing is needed.
    files = [f'data/train-{i:05d}-of-{TRAIN_SHARDS:05d}.parquet' for i in range(TRAIN_SHARDS)]
    done = state['shards']
    if [item['file'] for item in done] != files[:len(done)]:
        raise ValueError('Saved shard list differs from the pinned SUN397 revision')
    local = args.local_shards.expanduser().resolve() if args.local_shards else None
    if local is not None:
        for filename in files[len(done):]:
            if not (local / filename).is_file():
                parser.error(f'Missing local shard: {local / filename}')
    destination.mkdir(parents=True, exist_ok=True)
    save_json(progress, state)
    offset = done[-1]['stop_row'] if done else 0
    scratch = destination / '.downloads'
    for i, filename in enumerate(files[len(done):], start=len(done) + 1):
        action = 'Reading local' if local is not None else 'Downloading'
        print(f'{action} train shard {i}/{len(files)}: {filename}', flush=True)
        try:
            path = local / filename if local is not None else download_shard(filename, scratch)
        except FileNotFoundError:
            parser.exit(1, 'curl was not found. Install curl or supply --local-shards.\n')
        except subprocess.CalledProcessError as error:
            parser.exit(1, f'Could not download {filename} (curl exit {error.returncode}). '
                        'Check the connection and rerun the same command to resume.\n')
        offset, images = extract_shard(path, destination, offset, identity['rows'])
        done.append(dict(file=filename, stop_row=offset, images=images))
        save_json(progress, state)
        if local is None:
            path.unlink()  # Discard the temporary download after saving the images and manifest.
        print(f'Saved {sum(len(item["images"]) for item in done)}/{COUNT} images', flush=True)
    if offset != TRAIN_ROWS:
        raise RuntimeError(f'Expected {TRAIN_ROWS} SUN397 train rows; found {offset}')
    verify_output(destination, state)
    state['complete'] = True
    save_json(progress, state)
    print(f'SUN397 subset ready: {destination} ({COUNT} images)', flush=True)


if __name__ == '__main__':
    main()
