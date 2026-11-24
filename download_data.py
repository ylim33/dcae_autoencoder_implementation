"""Fetch the pinned authors' MATLAB datasets and verify their SHA-256."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

COMMIT = 'cffa92e9dc089ec69df227e686bbc8b9414125fe'
HASHES = {
    'UciData.mat': 'e1fa14291573fc5f547f54c59661e9f61e52a7fcd19259c18cf4bdec3178d657',
    'cornData.mat': 'b34ffc226ae8b20726594ae20172554ed5b0bdf5aa005a049d951180b5126cad',
}


def fetch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        path = directory / name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Existing file checksum mismatch: {path}; refusing overwrite')
            print(f'Verified existing {name}')
            continue
        url = f'https://raw.githubusercontent.com/viggin/yan-autoencoder/{COMMIT}/{name}'
        with urlopen(url, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'Download checksum mismatch: {name}')
        path.write_bytes(content)
        print(f'Downloaded and verified {name}')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent/'data')
    fetch(parser.parse_args().directory)
