# Implementation notes

The network equations follow [Yan and Zhang (2016)](https://doi.org/10.1109/TIM.2016.2573078) and the [original implementation](https://github.com/viggin/yan-autoencoder/tree/cffa92e9dc089ec69df227e686bbc8b9414125fe).

- **Regression head:** use a linear output for corn regression. The original SAE supervised stage always constructs a softmax head.
- **Cross-validation:** train a fresh model per corn fold and fit the target scaler on training labels only.
- **Optimizer state:** restore SciPy's returned parameters when optimization finishes.
- **Bias dimensions:** initialize encoder and decoder biases with their corresponding output dimensions.
- **Regularization:** count both encoder and decoder weights in the pretraining L2 term; with tied weights this is twice the encoder norm.
- **Loss scaling:** preserve the author's elementwise MSE and class-averaged cross-entropy conventions.

`legacy` mode uses the original NumPy initialization order, the shallow AE's four-times-Xavier range, and stochastic CG pretraining. `legacy-init` retains that initialization but uses Adam pretraining. `modern` uses standard Xavier and Adam. All modes use CG for supervised and final DCAE training.

The original stochastic CG pretraining can stop early with precision loss. Reaching the iteration limit does not mean convergence; optimizer status is saved with the results. Numerical equality with an executed Python 2.7 / Theano environment is not claimed.

For corn, `source-train` fits the feature scaler on source training samples; `all-unlabelled` follows the author's transductive use of all instrument features. Neither setting uses test labels for scaling or carries supervised weights across folds.

Unit tests in `tests/` cover model equations, gradients, initialization, loss scaling, fold separation, and prediction after model loading. They run separately from training:

```bash
python -m unittest discover -s tests -v
```
