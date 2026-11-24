"""Utilities for experiment provenance and summaries."""
import json
import platform
from pathlib import Path
import numpy as np
import scipy
import sklearn
import torch
from .data import file_hash


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def start_output(path, args, data_path):
    out = Path(path)
    if (out / 'config.json').exists():
        raise FileExistsError(f'Existing experiment: {out}. Use a new --output path.')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    info = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                python=platform.python_version(), torch=torch.__version__,
                numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                device='cpu', dtype='float64', dataset_sha256=file_hash(data_path),
                code_sha256={str(p.relative_to(root)):file_hash(p)
                             for p in [*root.glob('*.py'), *root.glob('dcae/*.py')]})
    write_json(out / 'config.json', info)
    return out
