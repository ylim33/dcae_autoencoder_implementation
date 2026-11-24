"""Original NumPy initialization and stochastic-CG pretraining compatibility.

No Theano dependency. The RandomStreams seed expansion follows Theano 0.9
shared_randomstreams.py. Different BLAS/SciPy versions can change line search.
"""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import minimize


@torch.no_grad()
def _uniform(layer, rng, factor=1.):
    bound = factor * np.sqrt(6. / (layer.in_features + layer.out_features))
    # Theano stores (input, output); transpose AFTER drawing, not before.
    weight = rng.uniform(-bound, bound, (layer.in_features, layer.out_features))
    layer.weight.copy_(torch.from_numpy(weight.T))
    if layer.bias is not None:
        layer.bias.zero_()


@torch.no_grad()
def initialize_sae(sae, rng):
    streams = []
    for enc, dec in zip(sae.enc, reversed(sae.dec)):
        stream_seed = int(rng.randint(2 ** 30))
        member_seed = int(np.random.RandomState(stream_seed).randint(2 ** 30))
        streams.append(np.random.RandomState(member_seed))
        _uniform(enc, rng, factor=4.)
        dec.weight.copy_(enc.weight.T)
        dec.bias.zero_()
    # In the original this head is drawn after pretraining. Pretraining only
    # consumes member streams, so the shared initialization RNG is unchanged.
    _uniform(sae.head, rng)
    return streams


@torch.no_grad()
def initialize_dcae(model, rng):
    rng.randint(2 ** 30)  # unused DCAE RandomStreams seed, consumed upstream
    if model.spec['correction_hidden'] is None:
        _uniform(model.cor, rng)
    else:
        _uniform(model.cor0, rng)
        model.cor0.weight.zero_()  # draw first, then zero, as in author code
        _uniform(model.cor1, rng)
    _uniform(model.sae.head, rng)


def pretrain_legacy(sae, x, streams, steps=1000, corruption=.1, l2=0.):
    """Reproduce separate stochastic cost/gradient calls and cost callback.

    This mode intentionally retains noisy CG to audit implementation fidelity.
    It is not a recommendation for optimizing general denoising autoencoders.
    As elsewhere in this port, the returned optimum is explicitly restored.
    """
    if steps < 1 or not 0 <= corruption < 1 or len(streams) != len(sae.enc):
        raise ValueError('Invalid legacy pretraining configuration')
    x = x.detach()
    statuses = []
    for enc, dec, rng in zip(sae.enc, reversed(sae.dec), streams):
        params = [enc.weight, enc.bias, dec.bias]
        def pack(values):
            return np.concatenate([values[0].T.ravel(), values[1].ravel(), values[2].ravel()])
        def restore(flat):
            count = enc.weight.numel()
            with torch.no_grad():
                enc.weight.copy_(torch.from_numpy(flat[:count].reshape(enc.in_features, enc.out_features).T))
                enc.bias.copy_(torch.from_numpy(flat[count:count+enc.out_features]))
                dec.bias.copy_(torch.from_numpy(flat[count+enc.out_features:]))
        def objective():
            mask = torch.from_numpy(rng.binomial(1, 1-corruption, size=tuple(x.shape))).to(x.dtype)
            h = sae.activate(enc(x * mask))
            rec = sae.activate(F.linear(h, enc.weight.T, dec.bias))
            return F.mse_loss(rec, x) + 2 * l2 * enc.weight.square().sum()
        def cost(flat):
            restore(flat)
            with torch.no_grad():
                return float(objective())
        def gradient(flat):
            restore(flat)
            for p in params:
                p.grad = None
            objective().backward()
            return pack([p.grad.detach().numpy() for p in params])
        callbacks = []
        result = minimize(cost, pack([p.detach().numpy() for p in params]),
                          jac=gradient, method='CG', callback=lambda flat: callbacks.append(cost(flat)),
                          options={'maxiter': steps, 'gtol': 1e-5})
        restore(result.x)
        with torch.no_grad():
            dec.weight.copy_(enc.weight.T)
            x = sae.activate(enc(x)).detach()
        statuses.append(dict(loss=float(result.fun), iterations=int(result.nit),
                             evaluations=int(result.nfev), gradient_evaluations=int(result.njev),
                             converged=bool(result.success), message=str(result.message),
                             callback_losses=callbacks))
    return dict(optimizer='legacy-stochastic-CG', max_iter=steps, corruption=corruption,
                l2=l2, layers=statuses, restore_returned_solution=True)
