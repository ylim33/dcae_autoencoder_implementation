"""Download the MATLAB datasets used by the original DCAE implementation."""
import argparse
from pathlib import Path
from urllib.request import urlopen

COMMIT = 'cffa92e9dc089ec69df227e686bbc8b9414125fe'
FILES = ('UciData.mat', 'cornData.mat')


def fetch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = directory / name
        if path.exists():
            print(f'Using existing {path}')
            continue
        url = f'https://raw.githubusercontent.com/viggin/yan-autoencoder/{COMMIT}/{name}'
        with urlopen(url, timeout=120) as response:
            content = response.read()
        path.write_bytes(content)
        print(f'Downloaded {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path,
                        default=Path(__file__).resolve().parent / 'data')
    fetch(parser.parse_args().directory)
