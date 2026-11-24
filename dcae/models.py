"""Network equations corresponding to Yan & Zhang (2016), equations (5)-(9)."""
from __future__ import annotations
import copy
import torch
from torch import nn

DTYPE = torch.float64


class StackedAutoencoder(nn.Module):
    """Tanh encoder/decoder for classification; linear for regression.

    Decoder weights are independent parameters. Training ties them during
    layerwise pretraining, then unties them for final DCAE optimization.
    Prediction heads return logits or unconstrained regression values.
    """

    def __init__(self, input_dim=128, hidden=(30, 20), outputs=6,
                 activation='tanh', task='classification'):
        super().__init__()
        if activation not in ('tanh', 'linear'):
            raise ValueError('activation must be tanh or linear')
        if task not in ('classification', 'regression'):
            raise ValueError('task must be classification or regression')
        if not hidden or min(input_dim, outputs, *hidden) < 1:
            raise ValueError('Layer dimensions must be positive')
        self.spec = dict(input_dim=input_dim, hidden=list(hidden), outputs=outputs,
                         activation=activation, task=task)
        dims = [input_dim, *hidden]
        self.enc = nn.ModuleList(nn.Linear(a, b, dtype=DTYPE)
                                 for a, b in zip(dims[:-1], dims[1:]))
        self.dec = nn.ModuleList(nn.Linear(b, a, dtype=DTYPE)
                                 for a, b in reversed(list(zip(dims[:-1], dims[1:]))))
        self.head = nn.Linear(hidden[-1], outputs, dtype=DTYPE)
        for layer in [*self.enc, *self.dec, self.head]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def activate(self, value):
        return torch.tanh(value) if self.spec['activation'] == 'tanh' else value

    def encode(self, x):
        for layer in self.enc:
            x = self.activate(layer(x))
        return x

    def forward(self, x):
        return self.head(self.encode(x))

    @torch.no_grad()
    def initialize_decoder_from_encoder(self):
        for encoder, decoder in zip(self.enc, reversed(self.dec)):
            decoder.weight.copy_(encoder.weight.T)


class DCAE(nn.Module):
    """Basic DCAE or DCAE with a nonlinear correction layer (DCAE-CL).

    Correction is added before the first encoder activation and subtracted
    BEFORE the final decoder linear map. There is no denoising in this stage.
    """

    def __init__(self, sae, domain_dim=2, correction_hidden=6):
        super().__init__()
        self.sae = copy.deepcopy(sae)
        self.spec = dict(sae=copy.deepcopy(sae.spec), domain_dim=domain_dim,
                         correction_hidden=correction_hidden)
        if correction_hidden is None:
            self.cor = nn.Linear(domain_dim, sae.enc[0].out_features,
                                 bias=False, dtype=DTYPE)
            nn.init.xavier_uniform_(self.cor.weight)
        else:
            self.cor0 = nn.Linear(domain_dim, correction_hidden, bias=False, dtype=DTYPE)
            self.cor1 = nn.Linear(correction_hidden, sae.enc[0].out_features,
                                  bias=False, dtype=DTYPE)
            nn.init.zeros_(self.cor0.weight)
            nn.init.xavier_uniform_(self.cor1.weight)
        nn.init.xavier_uniform_(self.sae.head.weight)
        nn.init.zeros_(self.sae.head.bias)

    def correction(self, domain):
        if self.spec['correction_hidden'] is None:
            return self.cor(domain)
        return self.cor1(torch.tanh(self.cor0(domain)))

    def encode(self, x, domain):
        for i, layer in enumerate(self.sae.enc):
            x = self.sae.activate(layer(x) + (self.correction(domain) if i == 0 else 0))
        return x

    def reconstruct(self, x, domain):
        x = self.encode(x, domain)
        for i, layer in enumerate(self.sae.dec):
            if i == len(self.sae.dec) - 1:
                x = x - self.correction(domain)
            x = self.sae.activate(layer(x))
        return x

    def forward(self, x, domain):
        return self.sae.head(self.encode(x, domain))
