# DCAE：Python 3.11 / PyTorch 复现

**本仓库是 Ke Yan 与 David Zhang（2016）DCAE 方法的独立复现与框架迁移，不是原作者的官方实现，也不提出新的 DCAE 算法。** 原论文与算法贡献归原作者所有；本仓库记录 PyTorch 实现、必要修复、与原实现的差异及完整实验结果。

## 原论文、作者与原始代码

> Ke Yan and David Zhang. **Correcting Instrumental Variation and Time-Varying Drift: A Transfer Learning Approach With Autoencoders.** *IEEE Transactions on Instrumentation and Measurement*, 65(9): 2012–2022, September 2016. DOI: **10.1109/TIM.2016.2573078**.

- [论文 DOI / 出版社入口](https://doi.org/10.1109/TIM.2016.2573078)
- [原作者公开预印本](https://raw.githubusercontent.com/viggin/viggin.github.com/master/papers/preprint_DCAE.pdf)
- [原作者 Ke Yan 的 GitHub：viggin](https://github.com/viggin)
- [原始 Python 2.7 / Theano 实现：viggin/yan-autoencoder](https://github.com/viggin/yan-autoencoder)
- [本次核对的上游固定提交：cffa92e](https://github.com/viggin/yan-autoencoder/tree/cffa92e9dc089ec69df227e686bbc8b9414125fe)
- [原作者网站（上游列出的历史网址）](http://yanke23.com)

原代码署名：**Copyright 2016 YAN Ke, Tsinghua Univ.** 使用本仓库开展研究时，请优先引用原论文；若使用本仓库的迁移实现，再同时注明本仓库 URL 与具体 commit。GitHub 的 “Cite this repository” 通过 [CITATION.cff](CITATION.cff) 优先提供原论文引用。[references.bib](references.bib) 提供 BibTeX。

## 复现内容与范围

- Python 3.11，PyTorch 网络与自动微分；CPU / float64。
- UCI 气体传感器漂移分类：DCAE-CL，Batch 1 训练，Batch 2–10 测试。
- corn 光谱回归：DCAE-basic，m5 → mp5 / mp6，独立四折验证。
- 保留原模型结构、损失尺度、配对校准、最终 LogisticRegression / Ridge 预测器。
- 修复回归头误用 softmax、跨折监督状态残留、Y 标准化使用测试统计量、优化结束后参数恢复等问题。证据及适用范围见 [BUGFIXES.md](BUGFIXES.md)。
- 不包含论文的私有呼气分析数据集，不声称复现了该项实验；本仓库也不包含此前探索的其他算法改进。

代码迁移、文档整理和实验执行使用了 AI 编程助手。算法归属、引用、实测结果及限制均明确区分；测试通过不代表原论文结论已被独立证明。

## 实测结果

UCI 按原脚本口径：先对 Batch 2–10 等权平均，再统计十个固定种子 0–9 的均值与样本标准差。论文的 ± 按原表报告；本地种子集合不声称与作者未公开的完整集合相同。

| 配置 | UCI 平均准确率 | 与论文均值绝对差 |
|---|---:|---:|
| 原论文 DCAE-CL | 93.21% ± 0.52% | — |
| `modern`：标准 Xavier + Adam 预训练 | 92.78% ± 0.57% | 0.43 个百分点 |
| `legacy-init`：作者初始化 + Adam 预训练 | 93.00% ± 0.93% | 0.21 个百分点 |
| `legacy`：作者初始化 + 随机 CG 预训练 | 92.95% ± 0.89% | 0.26 个百分点 |

**建议以 `legacy` 作为更忠实于公开实现的复现基线。** `legacy-init` 仍改变了预训练优化器，应作为迁移差异对照。CLI 默认保留早期 `modern` 行为，运行兼容版必须显式指定参数。

corn 为种子 0–4 的独立四折结果，表内是两个设备、四个属性的平均 RMSE（越低越好）：

| 配置 | 平均 RMSE ± 种子标准差 |
|---|---:|
| 原论文 Table V | 0.1961 ± 0.0022 |
| `modern`，源训练折 X 标准化 | 0.2026 ± 0.0006 |
| `legacy`，源训练折 X 标准化 | 0.2032 ± 0.0010 |
| `legacy`，作者的全无标签 X 标准化 | 0.2030 ± 0.0014 |

corn 保留了回归与标签隔离修复，因此与原脚本并非完全相同的流程。详细逐批次、逐种子、逐属性结果、追加实验时序及停止状态见 [FIDELITY_RESULTS.md](FIDELITY_RESULTS.md)、[FIDELITY_PLAN.md](FIDELITY_PLAN.md) 和 [benchmarks/](benchmarks/)。[RESULTS.md](RESULTS.md) 是早期五种子实验的历史报告。

这些结果属于接近程度的实测报告，不是新算法优越性或统计等效性证明。未执行旧 Python 2.7 / Theano 环境做逐参数对照。兼容模式的随机 CG 预训练存在 precision loss；最终 UCI DCAE 优化达到 1000 次预算，不能称为全部收敛。

## 安装与运行

在 Python 3.11 环境中：

```bash
git clone https://github.com/YaoliMa33/dcae_autoencoder_implementation.git
cd dcae_autoencoder_implementation
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_data.py
python -m unittest discover -s tests -v
```

[requirements.txt](requirements.txt) 是兼容依赖范围；[requirements-lock.txt](requirements-lock.txt) 记录本地实测精确版本，可在相应发行版本可用时用于锁定安装。版本和数据 SHA-256 同时记录在各实验的 `config.json`。原实验环境为 Python 3.11.17 / PyTorch 2.15.0+cpu / NumPy 2.4.6 / SciPy 1.17.1 / scikit-learn 1.9.1。

```bash
# UCI：建议的原流程兼容模式，十个固定种子
python run_uci.py --fidelity legacy --seeds 0 1 2 3 4 5 6 7 8 9 --output results/uci-legacy

# UCI：保留 Adam 的初始化对照
python run_uci.py --fidelity legacy-init --seeds 0 1 2 3 4 5 6 7 8 9 --output results/uci-legacy-init

# 单种子快速检查（不等于十种子复现）
python run_uci.py --fidelity legacy --seeds 0 --output results/uci-seed0

# corn：恢复作者使用全体无标签 X 的传导式标准化
python run_corn.py --fidelity legacy --x-scaling all-unlabelled --seeds 0 1 2 3 4 --output results/corn-legacy

# 检查仓库发布的逐种子汇总是否一致，不需要下载数据
python verify_benchmarks.py
```

`legacy` 预训练上限为 1000 次随机 CG；其他模式默认 400 步 Adam。监督微调与最终 DCAE 均使用 CG、上限 1000 次。程序拒绝覆盖已有 `config.json`，重复运行请更换 `--output`。

训练会保存每个目标批次或设备/折的模型、逐样本预测、参数、代码哈希和优化状态。模型保存后自动检查加载预测与 sklearn 预测器一致。仓库中的测试覆盖网络方程、梯度、损失缩放、随机数顺序、参数恢复、数据划分和模型预测；本地 10 项测试通过，见 [验证记录](benchmarks/validation.log)。本仓库没有附带本机虚拟环境、原始数据或训练权重。

## 数据与评估协议

### UCI

- [UCI Gas Sensor Array Drift Dataset at Different Concentrations](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations)。数据贡献归 UCI 条目列出的创建者；请按其要求引用相关数据论文。
- 本实验使用原作者 `UciData.mat` 中的预选源/目标对应样本；下载脚本固定上游提交并校验 SHA-256。
- 仅 Batch 1 的 445 个标签用于网络监督训练；StandardScaler 仅拟合 Batch 1，再除以 2。
- 每批最多 10 对校准样本，按时间累积；Batch 8 / 9 实际只有 8 / 6 对。
- 校准关系利用相同气体/浓度信息，**不是完全无目标域信息的无监督适应**。
- 主指标包含目标校准样本，与作者脚本一致；另行报告排除校准样本及重复行的分数。两种口径不能混用。
- 编码器 128→30→20，tanh，6 个校正单元，域向量 `[1, batch−1]`；重构仅使用源批次；监督项 1/16、配对项 1/8，CE 保留按类别数除以 6 的尺度。

### corn

- [Eigenvector corn spectroscopy 数据来源](https://eigenvector.com/data/Corn/)，本实验使用作者整理的 `cornData.mat`。
- 80 样本、700 光谱特征、4 个连续属性；每折 60 个训练样本、20 个测试样本；每台目标设备 7 对训练折校准样本。
- 每折使用独立监督模型，Y scaler 仅拟合训练折，回归头为线性。
- `source-train` 只用源训练折拟合 X scaler；`all-unlabelled` 用全部设备无标签特征，恢复作者的传导式设置。两种协议分别报告。
- 700→15→700 线性网络，设备 one-hot 域特征；监督项 2、配对项 16；最终 Ridge(alpha=1)。

数据与上游代码的权利、来源和许可状态见 [THIRD_PARTY.md](THIRD_PARTY.md) 与 [LICENSE_STATUS.md](LICENSE_STATUS.md)。

## 使用模型预测

```bash
python predict.py --model results/uci-legacy/seed0_batch2.pt --input samples.npy --output predictions.npy
```

`samples.npy` 为未标准化的原始特征：UCI 为 N×128，corn 为 N×700。UCI 输出类别 1–6，corn 输出原始单位的四个属性值。模型包含 scaler、神经网络和最终线性预测器，适用于训练时指定的目标批次/设备；新批次需要重新校准。

## 仓库结构

- `dcae/`：模型、训练、原流程兼容、数据与推理。
- `run_uci.py` / `run_corn.py`：实验入口。
- `tests/`：10 项自动检查。
- `benchmarks/`：全部已报告实验的指标、配置与优化日志 JSON；不将最好的单次运行冒充平均结果。
- `verify_benchmarks.py`：校验公开 JSON 中的种子覆盖与聚合分数；它不能替代从原始数据重新训练。
- `reference/README.md`：原始代码固定版本链接；不打包转载上游源码。

## BibTeX

```bibtex
@article{yan2016dcae,
  author  = {Yan, Ke and Zhang, David},
  title   = {Correcting Instrumental Variation and Time-Varying Drift:
             A Transfer Learning Approach With Autoencoders},
  journal = {IEEE Transactions on Instrumentation and Measurement},
  year    = {2016},
  volume  = {65},
  number  = {9},
  pages   = {2012--2022},
  month   = sep,
  doi     = {10.1109/TIM.2016.2573078}
}
```

本仓库的迁移实现和修复不意味着原作者参与、认可或背书。请勿把本仓库署名替代原论文与原始代码作者。
