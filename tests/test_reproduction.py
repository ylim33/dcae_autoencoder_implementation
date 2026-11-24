import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from dcae.models import StackedAutoencoder, DCAE
from dcae.training import supervised_loss, minimize_cg
from dcae.data import ROOT, load_uci, load_corn, corn_split
from dcae.inference import save_bundle, load_bundle, predict


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)

    def test_forward_matches_numpy_equations(self):
        model = DCAE(StackedAutoencoder(4, hidden=(3,2)))
        with torch.no_grad(): model.cor0.weight.normal_(0,.1)
        x = torch.randn(5,4,dtype=torch.float64)
        d = torch.randn(5,2,dtype=torch.float64)
        a = lambda t: t.detach().numpy()
        e0,e1 = model.sae.enc
        d0,d1 = model.sae.dec
        correction = np.tanh(a(d)@a(model.cor0.weight).T)@a(model.cor1.weight).T
        h0 = np.tanh(a(x)@a(e0.weight).T+a(e0.bias)+correction)
        h1 = np.tanh(h0@a(e1.weight).T+a(e1.bias))
        expected = np.tanh((np.tanh(h1@a(d0.weight).T+a(d0.bias))-correction)@a(d1.weight).T+a(d1.bias))
        np.testing.assert_allclose(a(model.encode(x,d)),h1,atol=1e-12)
        np.testing.assert_allclose(a(model.reconstruct(x,d)),expected,atol=1e-12)

    def test_linear_reconstruction_correction_cancels(self):
        model = DCAE(StackedAutoencoder(4, (3,), 2, 'linear','regression'),3,None)
        x = torch.randn(6,4,dtype=torch.float64)
        d = torch.randn(6,3,dtype=torch.float64)
        torch.testing.assert_close(model.reconstruct(x,d),model.sae.dec[0](model.sae.enc[0](x)))

    def test_gradients(self):
        model = DCAE(StackedAutoencoder(4,(3,2)))
        x = torch.randn(3,4,dtype=torch.float64,requires_grad=True)
        d = torch.randn(3,2,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(model.reconstruct,(x,d)))

    def test_legacy_cross_entropy_scale(self):
        logits = torch.randn(8,6,dtype=torch.float64)
        target = torch.arange(8)%6
        legacy = -(F.one_hot(target,6)*logits.log_softmax(1)).mean()
        torch.testing.assert_close(supervised_loss(logits,target,'classification'),legacy)

    def test_regression_head_allows_negative_values(self):
        sae = StackedAutoencoder(4,(3,),2,'linear','regression')
        with torch.no_grad():
            sae.head.weight.zero_()
            sae.head.bias.copy_(torch.tensor([-2.,3.]))
        result=sae(torch.zeros(2,4,dtype=torch.float64))
        torch.testing.assert_close(result,torch.tensor([[-2.,3.],[-2.,3.]],dtype=torch.float64))

    def test_optimizer_restores_returned_solution(self):
        p = torch.nn.Parameter(torch.tensor([8.],dtype=torch.float64))
        def fake_minimize(fun, initial, **kwargs):
            fun(np.array([99.]))  # Simulate a rejected final line-search trial.
            return SimpleNamespace(x=np.array([2.]),fun=0.,nit=1,nfev=2,success=True,message='ok')
        with patch('dcae.training.minimize',fake_minimize):
            minimize_cg([p],lambda:(p-2).square().sum())
        self.assertEqual(p.item(),2.)

    def test_calibration_protocols(self):
        x,y,ps,pt,masks,scaler=load_uci(ROOT/'data/UciData.mat')
        self.assertEqual(sum(len(v) for v in x),13910)
        self.assertEqual([len(v) for v in pt],[10,10,10,10,10,10,8,6,10])
        for batch,pairs,mask in zip(x[1:],pt,masks):
            for row in pairs:
                nearest=((batch.numpy()-row.numpy())**2).sum(1).argmin()
                self.assertFalse(mask[nearest])
        raw,labels,tr=load_corn(ROOT/'data/cornData.mat')
        for fold in range(4):
            train,test,cal=corn_split(tr,fold)
            self.assertEqual(test.sum(),20)
            self.assertFalse(test[cal].any())
            scaler=StandardScaler().fit(labels[train])
            changed=labels.copy(); changed[test]+=1e6
            np.testing.assert_array_equal(scaler.mean_,StandardScaler().fit(changed[train]).mean_)

    def test_bundle_predictors_match_sklearn(self):
        for task,outputs in [('classification',3),('regression',2)]:
            rng=np.random.default_rng(1); raw=rng.normal(size=(30,4))
            scaler=StandardScaler().fit(raw)
            model=DCAE(StackedAutoencoder(4,(3,),outputs,task=task))
            x=torch.tensor(scaler.transform(raw)/2,dtype=torch.float64)
            domain=torch.tensor([1.,1.],dtype=torch.float64).repeat(30,1)
            z=model.encode(x,domain).detach().numpy()
            if task=='classification':
                estimator=OneVsRestClassifier(LogisticRegression(solver='liblinear')).fit(z,np.arange(30)%3)
            else:
                estimator=Ridge().fit(z,rng.normal(size=(30,2)))
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'model.pt'
                save_bundle(path,model,scaler,estimator,[1.,1.],{})
                expected=estimator.predict(z)+(1 if task=='classification' else 0)
                np.testing.assert_allclose(predict(load_bundle(path),raw),expected,atol=1e-12)


if __name__=='__main__': unittest.main(verbosity=2)
