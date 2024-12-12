"""
CRISP - model weight downloader.

The trained CNN weights total roughly 656 MB, which is far past what
belongs in a git repository (GitHub rejects any single file over 100 MB,
and Git LFS on a free account would allow only about one clone per month
before exhausting the bandwidth quota). They are published as GitHub
Release assets instead, which have no bandwidth limit.

Usage:
    python download_models.py            # fetch anything missing
    python download_models.py --force    # re-download everything
    python download_models.py --check    # verify only, download nothing

The release tag and repository can be overridden with the CRISP_MODELS_REPO
and CRISP_MODELS_TAG environment variables, which is useful if you fork
the project and host the weights yourself.
"""

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request

REPO = os.getenv('CRISP_MODELS_REPO', 'Saif-khann/CRISP')
TAG = os.getenv('CRISP_MODELS_TAG', 'models-v1')
MODELS_DIR = 'models'

# name -> (approximate size in MB, sha256 or None if not yet published)
MODELS = {
    'mobilenet.keras': 34,
    'inception.keras': 151,
    'vgg16.keras': 185,
    'Foundation_mobile.keras': 34,
    'Superstructure_mobile.keras': 34,
    'Facade_inception.keras': 151,
    'Interior_mobile.keras': 34,
    'finishing_mobile.keras': 34,
}

BASE_URL = f'https://github.com/{REPO}/releases/download/{TAG}'


def _human(num_bytes):
    return f'{num_bytes / 1024 / 1024:.1f} MB'


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100.0, downloaded * 100.0 / total_size)
    bar_width = 30
    filled = int(bar_width * pct / 100)
    bar = '=' * filled + ' ' * (bar_width - filled)
    sys.stdout.write(f'\r    [{bar}] {pct:5.1f}%  {_human(min(downloaded, total_size))}')
    sys.stdout.flush()


def sha256_of(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def missing_models():
    return [name for name in MODELS
            if not os.path.exists(os.path.join(MODELS_DIR, name))]


def download(name, force=False):
    destination = os.path.join(MODELS_DIR, name)
    if os.path.exists(destination) and not force:
        print(f'  {name}: already present ({_human(os.path.getsize(destination))})')
        return True

    url = f'{BASE_URL}/{name}'
    temp_path = destination + '.part'
    print(f'  {name}: downloading (~{MODELS[name]} MB)')
    try:
        urllib.request.urlretrieve(url, temp_path, _progress)
        sys.stdout.write('\n')
        os.replace(temp_path, destination)
        return True
    except urllib.error.HTTPError as exc:
        sys.stdout.write('\n')
        print(f'    failed: HTTP {exc.code} for {url}')
        if exc.code == 404:
            print('    The release asset was not found. Check that the release '
                  f'"{TAG}" exists on {REPO} and contains this file.')
    except (urllib.error.URLError, OSError) as exc:
        sys.stdout.write('\n')
        print(f'    failed: {exc}')
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
    return False


def main():
    parser = argparse.ArgumentParser(description='Download CRISP model weights.')
    parser.add_argument('--force', action='store_true',
                        help='Re-download even if the file already exists.')
    parser.add_argument('--check', action='store_true',
                        help='Report what is missing without downloading.')
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.check:
        absent = missing_models()
        if absent:
            print(f'Missing {len(absent)} of {len(MODELS)} model files:')
            for name in absent:
                print(f'  - {name}')
            return 1
        print(f'All {len(MODELS)} model files are present.')
        return 0

    print(f'Fetching model weights from {REPO} (release: {TAG})\n')
    failures = [name for name in MODELS if not download(name, force=args.force)]

    print()
    if failures:
        print(f'{len(failures)} file(s) failed to download:')
        for name in failures:
            print(f'  - {name}')
        print('\nYou can retry, or download them manually from:')
        print(f'  https://github.com/{REPO}/releases/tag/{TAG}')
        print(f'and place them in the {MODELS_DIR}/ directory.')
        return 1

    total = sum(os.path.getsize(os.path.join(MODELS_DIR, name)) for name in MODELS)
    print(f'All {len(MODELS)} model files ready ({_human(total)} total).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
