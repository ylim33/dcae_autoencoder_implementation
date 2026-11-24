"""Experiment settings and result files."""
import json
import platform
from pathlib import Path
import numpy as np
import scipy
import sklearn
import torch


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def start_output(path, args):
    out = Path(path)
    if (out / 'config.json').exists():
        raise FileExistsError(f'Existing experiment: {out}. Use a new --output path.')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    info = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                python=platform.python_version(), torch=torch.__version__,
                numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                device='cpu', dtype='float64')
    write_json(out / 'config.json', info)
    return out
