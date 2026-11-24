"""Predict using a saved DCAE inference bundle and raw feature matrix (.npy)."""
import argparse
import numpy as np
from dcae import load_bundle, predict

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--input', required=True, help='N x 128 (UCI) or N x 700 (corn), raw features')
    p.add_argument('--output', required=True, help='Prediction .npy file')
    args = p.parse_args()
    result = predict(load_bundle(args.model), np.load(args.input, allow_pickle=False))
    np.save(args.output, result)
    print(f'Saved {len(result)} predictions to {args.output}')
