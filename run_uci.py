"""UCI: Batch 1 -> Batches 2-10, cumulative original paired calibration."""
import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from dcae.models import StackedAutoencoder, DCAE
from dcae.training import pretrain, fine_tune, fit_dcae
from dcae.legacy import initialize_sae, initialize_dcae, pretrain_legacy
from dcae.data import ROOT, load_uci, domain_rows
from dcae.inference import save_bundle, load_bundle, predict
from dcae.utils import start_output, write_json


def run(args):
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.batches)) != len(args.batches):
        raise ValueError('Seeds and batches must not contain duplicates')
    if args.pretrain_steps is None:
        args.pretrain_steps = 1000 if args.fidelity == 'legacy' else 400
    out = start_output(args.output, args, args.data)
    x, y, ps, pt, masks, scaler, audit = load_uci(args.data, args.budget)
    write_json(out / 'calibration_audit.json', audit)
    records = []
    source_domain = domain_rows([1., 0.], len(x[0]))
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        sae = StackedAutoencoder()
        rng = np.random.RandomState(seed)
        streams = initialize_sae(sae, rng) if args.fidelity != 'modern' else None
        initial = (pretrain_legacy(sae, x[0], streams, steps=args.pretrain_steps)
                   if args.fidelity == 'legacy' else pretrain(sae, x[0], steps=args.pretrain_steps))
        supervised = fine_tune(sae, x[0], torch.tensor(y[0]), steps=args.steps)
        write_json(out / f'source_seed{seed}.json', dict(pretrain=initial, supervised=supervised))
        # Consume every batch's original initialization even when only a subset
        # is requested; this preserves author RNG order and subset consistency.
        for batch in range(2, 11):
            b = batch - 1
            torch.manual_seed(seed * 100 + batch)
            model = DCAE(sae, correction_hidden=6 if args.variant == 'cl' else None)
            if args.fidelity != 'modern':
                initialize_dcae(model, rng)
            if batch not in args.batches:
                continue
            pair_source = torch.cat(ps[:b])
            pair_target = torch.cat(pt[:b])
            dps = domain_rows([1., 0.], len(pair_source))
            dpt = torch.cat([domain_rows([1., j+1], len(pt[j])) for j in range(b)])
            status = fit_dcae(model, x[0], torch.tensor(y[0]), source_domain,
                              x[0], source_domain, pair_source, pair_target, dps, dpt,
                              steps=args.steps)
            with torch.inference_mode():
                source_z = model.encode(x[0], source_domain).numpy()
                target_z = model.encode(x[b], domain_rows([1., b], len(x[b]))).numpy()
            estimator = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=1., max_iter=2000))
            estimator.fit(source_z, y[0])
            prediction = estimator.predict(target_z)
            mask = masks[b-1]
            row = dict(seed=seed, batch=batch, calibration_pairs=len(pt[b-1]),
                       accuracy=float(accuracy_score(y[b], prediction)),
                       accuracy_excluding_calibration=float(accuracy_score(y[b][mask], prediction[mask])),
                       test_count=int(mask.sum()),
                       confusion=confusion_matrix(y[b], prediction, labels=np.arange(6)).tolist())
            records.append(row)
            checkpoint = out / f'seed{seed}_batch{batch}.pt'
            save_bundle(checkpoint, model, scaler, estimator, [1., float(b)],
                        dict(dataset='UCI', seed=seed, batch=batch, variant=args.variant, fidelity=args.fidelity))
            # Verify an actual inference roundtrip against sklearn's classifier.
            raw_probe = scaler.inverse_transform(x[b][:12].numpy() * 2)
            np.testing.assert_array_equal(predict(load_bundle(checkpoint), raw_probe), prediction[:12] + 1)
            np.savez_compressed(out / f'pred_seed{seed}_batch{batch}.npz',
                                prediction=prediction+1, truth=y[b]+1, evaluation_mask=mask)
            write_json(out / f'fit_seed{seed}_batch{batch}.json', status)
            write_json(out / 'metrics.json', records)
            print(f'seed={seed} batch={batch} accuracy={row["accuracy"]:.4%} '
                  f'excluding_calibration={row["accuracy_excluding_calibration"]:.4%}', flush=True)
    seed_means = [np.mean([r['accuracy'] for r in records if r['seed'] == s]) for s in args.seeds]
    excluded = [np.mean([r['accuracy_excluding_calibration'] for r in records if r['seed'] == s]) for s in args.seeds]
    summary = dict(mean_accuracy=float(np.mean(seed_means)),
                   seed_sd=float(np.std(seed_means, ddof=1)) if len(seed_means)>1 else None,
                   per_seed_accuracy=seed_means,
                   mean_accuracy_excluding_calibration=float(np.mean(excluded)),
                   per_batch_accuracy={str(b):float(np.mean([r['accuracy'] for r in records if r['batch']==b])) for b in args.batches})
    write_json(out / 'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/UciData.mat')
    parser.add_argument('--output', type=Path, default=ROOT/'results/uci')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4])
    parser.add_argument('--batches', type=int, nargs='+', choices=range(2,11), default=list(range(2,11)))
    parser.add_argument('--budget', type=int, default=10)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--pretrain-steps', type=int, default=None)
    parser.add_argument('--fidelity', choices=['modern','legacy-init','legacy'], default='modern',
                        help='modern=previous Adam port; legacy-init=author initialization; legacy=author initialization and stochastic CG')
    parser.add_argument('--variant', choices=['cl','basic'], default='cl')
    run(parser.parse_args())
