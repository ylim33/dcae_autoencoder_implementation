# Published experiment records

These JSON files are copied from completed local runs of this PyTorch implementation, not from the paper or synthetic examples. They contain all reported seeds and batches, configs, per-stage optimizer states, and aggregate metrics. `fidelity_summary.json` joins the comparisons. `validation.log` records the ten passing tests from the fidelity audit.

| Directory | Configuration | Seeds |
|---|---|---|
| uci | modern, initial run | 0–4 |
| uci-modern-extra5 | modern, extension | 5–9 |
| uci-legacy-10 | original initialization + stochastic CG | 0–9 |
| uci-legacy-init-5 | original initialization + Adam | 0–4 |
| uci-legacy-init-extra5 | unchanged initialization + Adam replication | 5–9 |
| corn | modern, source-train X scaling | 0–4 |
| corn-legacy-all-5 | legacy, all-unlabelled X scaling | 0–4 |
| corn-legacy-source-5 | legacy, source-train X scaling | 0–4 |

The last UCI replication was added after seeing the first five results, without changing hyperparameters. Its timing is disclosed in [FIDELITY_PLAN.md](../FIDELITY_PLAN.md).

`source_artifacts_sha256.json` records hashes of the original local JSON bytes. Published `config.json` files normalize machine-specific data paths and add an explicit publication note; other experiment JSON files are unchanged. Code hashes in each config identify the code as it existed when that experiment started, not necessarily every file in this publication. The training equations and defaults were not changed for upload.

Run `python verify_benchmarks.py` to verify seed coverage and recompute the published means and standard deviations. This checks consistency of the saved JSON; it does not verify predictions from scratch. Original local model bundles and per-sample prediction files were checked during training and the fidelity audit but are not uploaded. To independently validate accuracy, download the datasets and rerun training using the recorded settings. This repository does not claim the exact random seeds or full environment used by the paper's authors are known.
