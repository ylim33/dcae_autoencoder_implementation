# DCAE — PyTorch Reproduction

An independent Python 3.11 / PyTorch reproduction of **Drift Correction Autoencoder (DCAE)** for electronic-nose sensor drift and instrument variation. This is a framework migration of the original method, not an official release by its authors.

## Paper and original code

Ke Yan and David Zhang, **“Correcting Instrumental Variation and Time-Varying Drift: A Transfer Learning Approach With Autoencoders,”** *IEEE Transactions on Instrumentation and Measurement*, 65(9), 2012–2022, 2016. [DOI: 10.1109/TIM.2016.2573078](https://doi.org/10.1109/TIM.2016.2573078).

- Original implementation: [viggin/yan-autoencoder](https://github.com/viggin/yan-autoencoder), by [Ke Yan (viggin)](https://github.com/viggin).
- Reference revision: [cffa92e](https://github.com/viggin/yan-autoencoder/tree/cffa92e9dc089ec69df227e686bbc8b9414125fe).
- Original attribution: Copyright 2016 YAN Ke, Tsinghua Univ.

Please cite the original paper when using this work. Citation files: [CITATION.cff](CITATION.cff) · [BibTeX](references.bib).

## Implementation comparison

| | Original implementation | This reproduction |
|---|---|---|
| Python | Python 2.7 | Python 3.11; tested with 3.11.17 |
| Framework | Theano; exact version not pinned upstream | PyTorch; tested with 2.15.0+cpu |
| Computation | Compiled symbolic graphs | Eager tensors and autograd |
| Pretraining | SciPy conjugate gradient (CG) with denoising | Original-style CG in `legacy` mode; Adam in `modern` / `legacy-init` |
| Supervised and final optimization | SciPy CG | SciPy CG with PyTorch gradients |
| Final prediction | scikit-learn LogisticRegression / Ridge | Same model types |

`--fidelity legacy` follows the original initialization and training flow more closely. The default is `modern`. Implementation fixes are documented in [BUGFIXES.md](BUGFIXES.md); exact tested dependencies are in [requirements-lock.txt](requirements-lock.txt).

## Datasets

| Dataset | Description | Access and redistribution |
|---|---|---|
| [UCI Gas Sensor Array Drift at Different Concentrations](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations) | 13,910 measurements, 128 features, 6 gases; used for drift classification. | Publicly downloadable. UCI lists **CC BY 4.0**, which permits redistribution with attribution, a license link, and identification of changes. However, the same page retains an older research-only restriction; we link to the source rather than mirror it pending clarification. |
| [Corn — Eigenvector Research](https://www.eigenvector.com/data/Corn/index.html) | 80 samples measured on 3 NIR instruments, 700 spectral channels and 4 properties; used for calibration transfer. | Publicly downloadable. No general redistribution license was found on the provider page, so we provide a source link rather than re-host the data. |

This reproduction uses the original authors' prepared `UciData.mat` and `cornData.mat`, including calibration selections. Their repository does not separately specify a license for these prepared files. **Datasets are not included in this repository.** To retrieve the fixed upstream files locally and verify their checksums:

```bash
python download_data.py
```

Public download availability does not by itself establish an open redistribution license. See [THIRD_PARTY.md](THIRD_PARTY.md) for data attribution and [LICENSE_STATUS.md](LICENSE_STATUS.md) for repository licensing status.
