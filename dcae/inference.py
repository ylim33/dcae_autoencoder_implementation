"""Self-contained inference bundles: weights, preprocessing and final classifier.

Bundles contain tensors and primitive metadata; loading uses weights_only=True.
No original training data or pickled sklearn estimator is needed at inference.
"""
import numpy as np
import torch
from .models import StackedAutoencoder, DCAE


def save_bundle(path, model, scaler, estimator, domain, metadata):
    if model.sae.spec['task'] == 'classification':
        weight = np.vstack([e.coef_[0] for e in estimator.estimators_])
        bias = np.array([e.intercept_[0] for e in estimator.estimators_])
    else:
        weight, bias = estimator.coef_, estimator.intercept_
    torch.save(dict(format_version=1, spec=model.spec, state_dict=model.state_dict(),
                    feature_mean=torch.as_tensor(scaler.mean_),
                    feature_scale=torch.as_tensor(scaler.scale_), divider=2.,
                    predictor_weight=torch.as_tensor(weight),
                    predictor_bias=torch.as_tensor(bias), domain=list(domain),
                    metadata=metadata), path)


def load_bundle(path):
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if bundle.get('format_version') != 1:
        raise ValueError('Unsupported DCAE bundle version')
    spec = bundle['spec']
    # Loading a model must not change the caller's random state.
    with torch.random.fork_rng(devices=[]):
        model = DCAE(StackedAutoencoder(**spec['sae']), spec['domain_dim'],
                     spec['correction_hidden'])
    model.load_state_dict(bundle['state_dict'])
    model.eval()
    return model, bundle


@torch.inference_mode()
def predict(loaded, raw_features):
    model, bundle = loaded
    x = torch.as_tensor(raw_features, dtype=torch.float64)
    if x.ndim != 2 or x.shape[1] != model.sae.spec['input_dim']:
        raise ValueError('Expected an N-by-input_dim matrix of RAW features')
    if not torch.isfinite(x).all():
        raise ValueError('Features must be finite')
    x = (x - bundle['feature_mean']) / bundle['feature_scale'] / bundle['divider']
    d = torch.tensor(bundle['domain'], dtype=torch.float64).repeat(len(x), 1)
    z = model.encode(x, d)
    output = z @ bundle['predictor_weight'].T + bundle['predictor_bias']
    if model.sae.spec['task'] == 'classification':
        # Argmax OvR decision scores = argmax normalized OvR sigmoid probabilities.
        return (output.argmax(1) + 1).numpy()  # Original MATLAB gas IDs 1..6.
    return output.numpy()  # Original moisture/oil/protein/starch units.
