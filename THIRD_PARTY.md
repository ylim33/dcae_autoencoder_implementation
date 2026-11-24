# Sources, attribution, and boundaries

## DCAE paper and original implementation

Ke Yan and David Zhang, *Correcting Instrumental Variation and Time-Varying Drift: A Transfer Learning Approach With Autoencoders*, IEEE Transactions on Instrumentation and Measurement **65**(9), 2012–2022, September 2016. [DOI](https://doi.org/10.1109/TIM.2016.2573078). Citation metadata was checked against the DOI's Crossref record.

- [Original author: viggin / Ke Yan](https://github.com/viggin)
- [Original Python 2.7 / Theano repository](https://github.com/viggin/yan-autoencoder)
- [Pinned source commit](https://github.com/viggin/yan-autoencoder/tree/cffa92e9dc089ec69df227e686bbc8b9414125fe)
- [Author-hosted preprint](https://raw.githubusercontent.com/viggin/viggin.github.com/master/papers/preprint_DCAE.pdf)
- [Author website listed upstream](http://yanke23.com) (historical URL; availability may vary)

Original attribution: **Copyright 2016 YAN Ke, Tsinghua Univ.** Algorithmic credit belongs to the original authors. This repository contributes the framework migration, documented fixes, compatibility implementation, tests, and measured reproduction artifacts. It is not endorsed by the original authors.

No explicit upstream license file was found at the pinned revision. The publication includes source links rather than a copy of the original toolbox. The local development audit used an unmodified upstream snapshot; statements in historical reports about `reference/*.py` refer to that snapshot. [reference/README.md](reference/README.md) maps those names to exact upstream URLs. See [LICENSE_STATUS.md](LICENSE_STATUS.md).

## Data

- [UCI Gas Sensor Array Drift Dataset at Different Concentrations](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations). Attribute the data creators and publications listed there, including Vergara et al. (2012), *Chemical gas sensor drift compensation using classifier ensembles*, and Rodriguez-Lujan et al. (2014), *On the calibration of sensor arrays for pattern recognition using the minimal number of experiments*.
- [Corn spectroscopy: Eigenvector Research](https://eigenvector.com/data/Corn/). The original DCAE repository supplies `cornData.mat` with device measurements and calibration indices.
- `download_data.py` retrieves the authors' MATLAB files from the pinned commit and verifies SHA-256. Those files are not included in this repository. Consult the original providers for reuse terms and citation requirements.
- The private breath-analysis dataset described in the paper is absent from the original public repository and is not reproduced here.

## Random-stream compatibility and dependencies

The legacy seed-expansion behavior was checked against [Theano 0.9 shared_randomstreams.py](https://github.com/Theano/Theano/blob/rel-0.9.0/theano/tensor/shared_randomstreams.py). This implementation uses NumPy RandomState for seed expansion and binomial draws; it has no runtime Theano dependency and is not claimed to be bitwise equivalent to an executed Theano run.

The software depends on PyTorch, NumPy, SciPy, and scikit-learn. These projects retain their respective licenses and attribution. No dependency source is vendored here.
