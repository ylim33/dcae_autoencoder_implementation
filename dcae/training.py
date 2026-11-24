"""Training: PyTorch autograd, SciPy CG for fidelity to the original optimizer."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import minimize


def minimize_cg(parameters, objective, max_iter=1000):
    """One deterministic objective/gradient evaluation per SciPy call.

    Always restores result.x, rather than leaving a last line-search probe in
    the network. Non-convergence status is saved with the training results.
    """
    if max_iter < 1:
        raise ValueError('max_iter must be positive')
    params = [p for p in parameters if p.requires_grad]
    if any(p.device.type != 'cpu' or p.dtype != torch.float64 for p in params):
        raise ValueError('SciPy CG requires CPU float64 parameters')
    sizes = [p.numel() for p in params]

    def restore(flat):
        offset = 0
        with torch.no_grad():
            for parameter, size in zip(params, sizes):
                parameter.copy_(torch.from_numpy(flat[offset:offset + size]).reshape(parameter.shape))
                offset += size

    def evaluate(flat):
        restore(flat)
        for parameter in params:
            parameter.grad = None
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss during CG')
        loss.backward()
        gradient = np.concatenate([
            p.grad.detach().numpy().ravel() if p.grad is not None else np.zeros(p.numel())
            for p in params])
        if not np.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite gradient during CG')
        return float(loss.detach()), gradient

    initial = np.concatenate([p.detach().numpy().ravel() for p in params])
    result = minimize(evaluate, initial, jac=True, method='CG',
                      options={'maxiter': max_iter, 'gtol': 1e-5})
    restore(result.x)
    return dict(loss=float(result.fun), iterations=int(result.nit),
                evaluations=int(result.nfev), converged=bool(result.success),
                message=str(result.message))


def supervised_loss(logits, target, task):
    # Legacy CE = mean over examples AND classes of one-hot * log probability.
    if task == 'classification':
        return F.cross_entropy(logits, target) / logits.shape[1]
    return F.mse_loss(logits, target)


def pretrain(sae, x, steps=400, lr=.01, corruption=.1, l2=0.):
    """Layerwise tied denoising AE. Adam handles resampled corruption safely.

    L2 counts both encoder and tied decoder matrices, as the intended original
    formula ||W||^2+||W_prime||^2 (2||W||^2 when tied).
    """
    if steps < 1 or not 0 <= corruption < 1:
        raise ValueError('Invalid pretraining steps or corruption rate')
    x = x.detach()
    history = []
    for i, encoder in enumerate(sae.enc):
        decoder = sae.dec[len(sae.enc) - 1 - i]
        optimizer = torch.optim.Adam([encoder.weight, encoder.bias, decoder.bias], lr=lr)
        for _ in range(steps):
            noisy = x * (torch.rand_like(x) > corruption)
            h = sae.activate(encoder(noisy))
            reconstruction = sae.activate(F.linear(h, encoder.weight.T, decoder.bias))
            loss = F.mse_loss(reconstruction, x) + l2 * 2 * encoder.weight.square().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            decoder.weight.copy_(encoder.weight.T)
            x = sae.activate(encoder(x)).detach()
        history.append(float(loss.detach()))
    return dict(optimizer='Adam', steps_per_layer=steps, final_layer_losses=history,
                corruption=corruption, lr=lr, l2=l2)


def fine_tune(sae, x, y, steps=1000, l2=.001):
    def objective():
        prediction = sae(x)
        regularizer = sum(layer.weight.square().sum() for layer in [*sae.enc, sae.head])
        return supervised_loss(prediction, y, sae.spec['task']) + l2 * regularizer
    status = minimize_cg([*sae.enc.parameters(), *sae.head.parameters()], objective, steps)
    sae.initialize_decoder_from_encoder()
    return status


def fit_dcae(model, x, y, domain, reconstruction_x, reconstruction_domain,
             pair_source, pair_target, pair_source_domain, pair_target_domain,
             steps=1000, supervised_weight=1/16, transfer_weight=1/8):
    if len(pair_source) != len(pair_target) or len(pair_source) == 0:
        raise ValueError('Nonempty, matched source/target calibration pairs required')

    def objective():
        reconstruction = F.mse_loss(model.reconstruct(reconstruction_x, reconstruction_domain),
                                    reconstruction_x)
        supervised = supervised_loss(model(x, domain), y, model.sae.spec['task'])
        transfer = F.mse_loss(model.encode(pair_source, pair_source_domain),
                             model.encode(pair_target, pair_target_domain))
        return reconstruction + supervised_weight * supervised + transfer_weight * transfer
    return minimize_cg(model.parameters(), objective, steps)
