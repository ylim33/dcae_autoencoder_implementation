"""Recompute published aggregates from saved JSON; does not retrain models."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent / 'benchmarks'

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

def verify_uci(directories, seeds, expected):
    records = [r for d in directories for r in read(f'{d}/metrics.json')]
    keys = [(r['seed'], r['batch']) for r in records]
    wanted = {(seed, batch) for seed in seeds for batch in range(2, 11)}
    assert len(keys) == len(wanted) and set(keys) == wanted
    values = [np.mean([r['accuracy'] for r in records if r['seed'] == s]) * 100 for s in seeds]
    batches = [np.mean([r['accuracy'] for r in records if r['batch'] == b]) * 100 for b in range(2, 11)]
    np.testing.assert_allclose(values, expected['seed_means'], rtol=1e-12)
    np.testing.assert_allclose(batches, expected['per_batch'], rtol=1e-12)
    np.testing.assert_allclose([np.mean(values), np.std(values, ddof=1)],
                               [expected['mean'], expected['sd']], rtol=1e-12)
    print(f"UCI {','.join(directories)}: {np.mean(values):.4f}% +/- {np.std(values,ddof=1):.4f} pp")

def main():
    joined = read('fidelity_summary.json')
    comparisons = [
        (['uci'], range(5), '上一版 / 5 seeds'),
        (['uci','uci-modern-extra5'], range(10), '上一版 / 10 seeds'),
        (['uci-legacy-init-5'], range(5), '仅原始初始化 / 5 seeds'),
        (['uci-legacy-init-5','uci-legacy-init-extra5'], range(10), '仅原始初始化 / 10 seeds'),
        (['uci-legacy-10'], range(10), '原始流程兼容 / 10 seeds'),
    ]
    for dirs,seeds,name in comparisons:
        verify_uci(dirs,seeds,joined['uci'][name])
    for directory in ['corn','corn-legacy-all-5','corn-legacy-source-5']:
        records=read(f'{directory}/metrics.json')
        assert [r['seed'] for r in records] == list(range(5))
        rmse=np.array([r['rmse'] for r in records])
        values=rmse.mean((1,2))
        summary=read(f'{directory}/summary.json')
        np.testing.assert_allclose(values,[r['mean_rmse'] for r in records],rtol=1e-12)
        np.testing.assert_allclose([values.mean(),values.std(ddof=1)],
                                   [summary['mean_rmse'],summary['seed_sd']],rtol=1e-12)
        np.testing.assert_allclose(rmse.mean(0),summary['per_device_property_rmse'],rtol=1e-12)
        print(f"Corn {directory}: {values.mean():.6f} +/- {values.std(ddof=1):.6f}")
    hashes=read('source_artifacts_sha256.json')
    count=0
    for path,digest in hashes.items():
        if path.endswith('/config.json'):
            config=read(path)
            assert config['arguments']['data'].startswith('data/')
            assert config['publication_note']
        else:
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
            count+=1
    print(f'PASS: seed coverage, reported aggregates and {count} unchanged source JSON hashes.')
    print('This checks saved-record consistency, not independent training accuracy.')

if __name__=='__main__':
    main()
