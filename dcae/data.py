"""Dataset loading and protocol validation. MATLAB labels are converted to 0-based."""
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def as_tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def domain_rows(vector, count):
    return as_tensor(np.tile(vector, (count, 1)))


def load_uci(path, budget=10):
    if budget < 1:
        raise ValueError('Calibration budget must be positive')
    mat = loadmat(path)
    raw = list(mat['data'][0])
    labels = [a.ravel().astype(np.int64) - 1 for a in mat['label'][0]]
    scaler = StandardScaler().fit(raw[0])
    x = [as_tensor(scaler.transform(a) / 2) for a in raw]
    source = [as_tensor(scaler.transform(a[:budget]) / 2) for a in mat['Tsrc'][0]]
    target = [as_tensor(scaler.transform(a[:budget]) / 2) for a in mat['Ttar'][0]]
    masks = []
    for i, (s, t) in enumerate(zip(source, target), 1):
        # Evaluation labels do not participate in matching or fitting.
        distances = cdist(t.numpy(), x[i].numpy())
        source_distances = cdist(s.numpy(), x[0].numpy())
        if max(distances.min(1).max(), source_distances.min(1).max()) > 1e-7:
            raise ValueError('Calibration pairs do not match the supplied dataset')
        # Exclude every duplicate of a calibration row, not just the first match.
        mask = ~np.any(distances <= 1e-7, axis=0)
        masks.append(mask)
    return x, labels, source, target, masks, scaler


def load_corn(path):
    mat = loadmat(path)
    return list(mat['ftAll'][0]), mat['Y'], mat['trList'].astype(np.int64) - 1


def corn_split(transfer_indices, fold, budget=7):
    if fold not in range(4) or not 1 <= budget <= len(transfer_indices):
        raise ValueError('Invalid corn fold or calibration budget')
    test = np.arange(80) % 4 == fold
    train = ~test
    selected = transfer_indices[:budget, fold]
    if np.any(selected < 0) or np.any(selected >= train.sum()):
        raise ValueError('Invalid calibration index within source training fold')
    calibration = np.flatnonzero(train)[selected]
    return train, test, calibration
