"""Corn: m5 -> mp5/mp6, independent four-fold validation, seven standards."""
import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from dcae.models import StackedAutoencoder, DCAE
from dcae.training import pretrain, fine_tune, fit_dcae
from dcae.legacy import initialize_sae, initialize_dcae, pretrain_legacy
from dcae.data import ROOT, as_tensor, load_corn, corn_split, domain_rows
from dcae.inference import save_bundle
from dcae.utils import start_output, write_json


def run(args):
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate seeds')
    if args.pretrain_steps is None:
        args.pretrain_steps = 1000 if args.fidelity == 'legacy' else 400
    out = start_output(args.output, args)
    raw, targets, transfer_indices = load_corn(args.data)
    domains = [domain_rows(np.eye(3)[i], 80) for i in range(3)]
    all_domains = torch.cat(domains)
    records = []
    for seed in args.seeds:
        predictions = np.zeros((2, 80, 4))
        for fold in range(4):
            torch.manual_seed(seed * 100 + fold)
            np.random.seed(seed * 100 + fold)
            train, test, calibration = corn_split(transfer_indices, fold, args.budget)
            # No target labels or test-fold labels enter either scaler.
            scaler = StandardScaler().fit(
                np.vstack(raw) if args.x_scaling == 'all-unlabelled' else raw[0][train])
            y_scaler = StandardScaler().fit(targets[train])
            x = [as_tensor(scaler.transform(v) / 2) for v in raw]
            y = as_tensor(y_scaler.transform(targets[train]))
            all_x = torch.cat(x)
            # Fresh network for EVERY fold: no previous-fold supervised weights.
            sae = StackedAutoencoder(700, hidden=(15,), outputs=4,
                                     activation='linear', task='regression')
            rng = np.random.RandomState(seed * 100 + fold)
            streams = initialize_sae(sae, rng) if args.fidelity == 'legacy' else None
            initial = (pretrain_legacy(sae, all_x, streams, steps=args.pretrain_steps, l2=.001)
                       if args.fidelity == 'legacy' else
                       pretrain(sae, all_x, steps=args.pretrain_steps, lr=.001, l2=.001))
            supervised = fine_tune(sae, x[0][train], y, steps=args.steps, l2=.0001)
            model = DCAE(sae, domain_dim=3, correction_hidden=None)
            if args.fidelity == 'legacy':
                initialize_dcae(model, rng)
            ps = x[0][calibration].repeat(2, 1)
            pt = torch.cat([v[calibration] for v in x[1:]])
            dps = domains[0][calibration].repeat(2, 1)
            dpt = torch.cat([v[calibration] for v in domains[1:]])
            status = fit_dcae(model, x[0][train], y, domains[0][train],
                              all_x, all_domains, ps, pt, dps, dpt, steps=args.steps,
                              supervised_weight=2., transfer_weight=16.)
            with torch.inference_mode():
                z = model.encode(x[0][train], domains[0][train]).numpy()
            estimator = Ridge(alpha=1.).fit(z, targets[train])
            for device in [1, 2]:
                with torch.inference_mode():
                    zt = model.encode(x[device][test], domains[device][test]).numpy()
                prediction = estimator.predict(zt)
                predictions[device-1, test] = prediction
                checkpoint = out / f'seed{seed}_fold{fold}_device{device}.pt'
                save_bundle(checkpoint, model, scaler, estimator, np.eye(3)[device].tolist(),
                            dict(dataset='corn', seed=seed, fold=fold, target_device=device,
                                 fidelity=args.fidelity, x_scaling=args.x_scaling))
            write_json(out / f'fit_seed{seed}_fold{fold}.json',
                       dict(pretrain=initial, supervised=supervised, dcae=status,
                            calibration_indices=calibration.tolist(),
                            test_indices=np.flatnonzero(test).tolist(),
                            y_scaler_mean=y_scaler.mean_.tolist(),
                            y_scaler_scale=y_scaler.scale_.tolist()))
            print(f'seed={seed} fold={fold+1}/4 complete', flush=True)
        rmse = np.sqrt(np.mean((predictions - targets[None])**2, axis=1))
        row = dict(seed=seed, rmse=rmse.tolist(), mean_rmse=float(rmse.mean()))
        records.append(row)
        np.savez_compressed(out / f'pred_seed{seed}.npz', prediction=predictions, truth=targets)
        write_json(out / 'metrics.json', records)
        print(f'seed={seed} RMSE={rmse.round(4).tolist()}', flush=True)
    values = np.array([r['rmse'] for r in records])
    averages = np.array([r['mean_rmse'] for r in records])
    summary = dict(mean_rmse=float(averages.mean()),
                   seed_sd=float(averages.std(ddof=1)) if len(averages)>1 else None,
                   per_device_property_rmse=values.mean(0).tolist(),
                   per_device_mean_rmse=values.mean((0,2)).tolist(),
                   target_devices=['mp5','mp6'],
                   properties=['moisture','oil','protein','starch'])
    write_json(out / 'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/cornData.mat')
    parser.add_argument('--output', type=Path, default=ROOT/'results/corn')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4])
    parser.add_argument('--budget', type=int, default=7)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--pretrain-steps', type=int, default=None)
    parser.add_argument('--fidelity', choices=['modern','legacy'], default='modern')
    parser.add_argument('--x-scaling', choices=['source-train','all-unlabelled'], default='source-train',
                        help='all-unlabelled restores author transductive X scaling; Y always train-only')
    run(parser.parse_args())
