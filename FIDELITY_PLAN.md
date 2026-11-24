> 发布说明：本文保留本地复现实验记录。原代码文件名对应 [固定版本源码链接](reference/README.md)，上游源码未随本仓库转载；原 results/ 中的 JSON 发布在 [benchmarks/](benchmarks/)，模型与逐样本预测需本地重新训练生成。

# Fidelity audit: fixed design before running

2026-11-23. Preserve the previous five-seed results in results/uci.

Primary: UCI --fidelity legacy, seeds 0 through 9, 1000 CG iterations for every stage.
Comparator: extend modern mode with seeds 5 through 9 (combine with saved 0 through 4).
Attribution: legacy-init, seeds 0 through 4, unchanged Adam 400-step pretraining.
No test labels select training iterations, hyperparameters, seed subsets or checkpoints.
Report every run, both nine-batch mean and each batch, including failed CG statuses.
The main outcome is fidelity plus discrepancy, not winning an accuracy search.

Legacy changes are grounded in reference/yan_ae.py (4x Xavier and RandomStreams),
reference/yan_sae.py (shared RNG), reference/dcae.py (RNG consumption), and
Theano rel-0.9.0/theano/tensor/shared_randomstreams.py (two-level seed expansion).
Framework rounding and SciPy line-search behavior are not expected to be identical.
Return-result restoration, bias dimensions, and regression/CV fixes stay in place.

Corn extension (specified before running either new corn experiment):
legacy mode, seeds 0-4, independent four-fold training, with two X scaling settings:
source-train (previous protocol) and all-unlabelled (original transductive protocol).
Both keep train-only Y scaler, linear regression heads, and no cross-fold label reuse.
No choice of settings is based on target scores. Original per-fold RNG continuation
cannot be matched exactly while also removing cross-fold supervised weight reuse.

Additional replication after reading the prespecified UCI results:
legacy-init seeds 0-4 gave 93.1240% (SD 0.6996 percentage points).
Extend that exact configuration with seeds 5-9 for the same ten-seed count as
modern and legacy; no hyperparameter changes. This is a disclosed post-result
replication, not part of the initial primary comparison. Retain all ten seeds.
