"""Install the verified ViT-S weights from a local file or the Hugging Face cache."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'checkpoints/vit_raw/model.safetensors'
REPO = 'timm/vit_small_patch16_224.augreg_in21k_ft_in1k'
FILE_SHA256 = '79c03c635cdfd798a364a9d8c4e5c0b7255b975ea2c9616046d4f77ab01435aa'


def verify(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != FILE_SHA256:
        raise ValueError(f'SHA256 mismatch for {path}. Expected the original ViT-S safetensors file.')


def cached_weights():
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as error:
        raise ValueError('Use the fimaq environment to read the Hugging Face cache, or pass --source.') from error
    path = try_to_load_from_cache(REPO, DESTINATION.name)
    if not isinstance(path, str):
        raise ValueError(f'{REPO}/{DESTINATION.name} is absent from the local cache. Pass --source with a verified copy.')
    return Path(path)


def install(source, destination):
    if destination.exists() or destination.is_symlink():
        verify(destination)
        return
    if source is None:
        source = cached_weights()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, prefix='.vit-small-', delete=False) as handle:
        staged = Path(handle.name)
    try:
        shutil.copyfile(source, staged)
        verify(staged)
        try:
            os.link(staged, destination)
        except FileExistsError:
            # Another invocation may have installed it while this copy was being verified.
            verify(destination)
    finally:
        staged.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Explicit path to the original model.safetensors')
    args = parser.parse_args()
    try:
        install(args.source.expanduser() if args.source else None, DESTINATION)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f'Verified ViT-S weights: {DESTINATION}')
    print(f'File SHA256: {FILE_SHA256}')


if __name__ == '__main__':
    main()
