import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from dcae.models import StackedAutoencoder, DCAE
from dcae.legacy import initialize_sae, initialize_dcae, pretrain_legacy


class LegacyTests(unittest.TestCase):
    def test_original_draw_order_and_transpose(self):
        rng = np.random.RandomState(0)
        expected = np.random.RandomState(0)
        sae = StackedAutoencoder(4, (3,2), 2)
        streams = initialize_sae(sae, rng)
        for layer, stream in zip(sae.enc, streams):
            outer = int(expected.randint(2**30))
            inner = int(np.random.RandomState(outer).randint(2**30))
            bound = 4 * np.sqrt(6 / (layer.in_features + layer.out_features))
            w = expected.uniform(-bound,bound,(layer.in_features,layer.out_features))
            np.testing.assert_array_equal(layer.weight.detach().numpy(), w.T)
            np.testing.assert_array_equal(stream.binomial(1,.9,12),np.random.RandomState(inner).binomial(1,.9,12))
            self.assertEqual(tuple(layer.bias.shape), (layer.out_features,))
        bound = np.sqrt(6 / 4)
        np.testing.assert_array_equal(sae.head.weight.detach().numpy(),expected.uniform(-bound,bound,(2,2)).T)
        model = DCAE(sae, 2, 3)
        initialize_dcae(model,rng)
        expected.randint(2**30)
        expected.uniform(-np.sqrt(6/5),np.sqrt(6/5),(2,3))
        np.testing.assert_array_equal(model.cor0.weight.detach().numpy(),np.zeros((3,2)))
        np.testing.assert_array_equal(model.cor1.weight.detach().numpy(),expected.uniform(-1.,1.,(3,3)).T)
        bound=np.sqrt(6/4)
        np.testing.assert_array_equal(model.sae.head.weight.detach().numpy(),expected.uniform(-bound,bound,(2,2)).T)
        self.assertEqual(rng.randint(2**30),expected.randint(2**30))

    def test_stochastic_calls_and_restore(self):
        sae=StackedAutoencoder(4,(3,),2)
        streams=initialize_sae(sae,np.random.RandomState(9))
        saved=sae.enc[0].weight.detach().clone()
        x=torch.arange(40,dtype=torch.float64).reshape(10,4)/40
        def fake_minimize(fun, initial, jac, callback, **kwargs):
            value=fun(initial)
            # Independent masks must produce different costs, as upstream does.
            self.assertNotEqual(value,fun(initial))
            gradient=jac(initial)
            self.assertEqual(gradient.shape,initial.shape)
            self.assertTrue(np.isfinite(gradient).all())
            callback(initial)
            fun(initial+0.1)
            return SimpleNamespace(x=initial, fun=value, nit=1, nfev=4, njev=1, success=False, message='test')
        with patch('dcae.legacy.minimize',fake_minimize):
            status=pretrain_legacy(sae,x,streams,steps=2)
        torch.testing.assert_close(sae.enc[0].weight,saved)
        self.assertEqual(len(status['layers'][0]['callback_losses']),1)
