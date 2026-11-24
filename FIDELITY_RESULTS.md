> 发布说明：本文保留本地复现实验记录。原代码文件名对应 [固定版本源码链接](reference/README.md)，上游源码未随本仓库转载；原 results/ 中的 JSON 发布在 [benchmarks/](benchmarks/)，模型与逐样本预测需本地重新训练生成。

# DCAE 原流程兼容复核

实验方案在运行前记录于 FIDELITY_PLAN.md。所有预定种子、批次和停止状态均保留；没有按测试准确率选轮数、种子或 checkpoint。初始化对照在看到前五种子结果后，追加预定后五种子，使比较次数一致；该追加验证已单独记录于方案。

## UCI：总体结果

指标为 Batch 2–10 准确率等权平均。± 是种子均值的样本标准差，不是置信区间。论文报告 10 次初始化；作者脚本公开种子 0，但未提供论文全部逐次结果，因此本实验 0–9 不代表已知与作者完全相同的种子集合。

| 配置 | 均值 ± SD | 与论文均值绝对差 / 百分点 | 九批次绝对差均值 / 百分点 |
|---|---:|---:|---:|
| 论文 DCAE-CL | 93.21% ± 0.52% | — | — |
| 上一版 / 5 seeds | 92.77% ± 0.49% | 0.44 | 0.75 |
| 上一版 / 10 seeds | 92.78% ± 0.57% | 0.43 | 0.61 |
| 仅原始初始化 / 5 seeds | 93.12% ± 0.70% | 0.09 | 0.38 |
| 仅原始初始化 / 10 seeds | 93.00% ± 0.93% | 0.21 | 0.46 |
| 原始流程兼容 / 10 seeds | 92.95% ± 0.89% | 0.26 | 0.56 |

原始流程兼容模式相对论文均值的绝对差为 0.26 个百分点；上一版同为十种子时为 0.43 个百分点。兼容版九批次绝对差均值为 0.56 个百分点。

兼容模式更接近公开代码的流程，适合作为忠实性复现基线；初始化 + Adam 模式保留了预训练优化器变更，应明确作为迁移实现差异报告。所有模式的标准差均完整展示，不能只根据一个更贴近论文的均值宣称统计等效。

## UCI：逐批次比较

| Batch | 论文 | 上一版 10 seeds | 仅初始化 10 seeds | 兼容版 10 seeds |
|---|---:|---:|---:|---:|
| 2 | 97.47% | 96.88% | 97.28% | 97.30% |
| 3 | 96.13% | 95.11% | 95.56% | 96.13% |
| 4 | 90.50% | 88.57% | 89.13% | 88.76% |
| 5 | 98.58% | 98.48% | 98.58% | 98.58% |
| 6 | 94.94% | 94.71% | 95.27% | 95.34% |
| 7 | 95.47% | 95.49% | 96.05% | 95.12% |
| 8 | 92.62% | 91.80% | 91.73% | 91.19% |
| 9 | 96.40% | 97.17% | 96.55% | 96.96% |
| 10 | 76.82% | 76.79% | 76.88% | 77.18% |

兼容版排除校准样本（含重复行）后均值：92.83%。

## UCI：全部种子

| Seed | 上一版 | 仅初始化 | 兼容版 |
|---|---:|---:|---:|
| 0 | 92.915% | 92.958% | 93.720% |
| 1 | 93.323% | 92.670% | 93.832% |
| 2 | 92.061% | 92.863% | 92.105% |
| 3 | 93.020% | 92.769% | 93.832% |
| 4 | 92.522% | 94.361% | 93.896% |
| 5 | 93.292% | 92.910% | 92.008% |
| 6 | 92.261% | 93.545% | 93.202% |
| 7 | 93.139% | 93.072% | 91.480% |
| 8 | 93.410% | 90.888% | 92.992% |
| 9 | 91.843% | 94.011% | 92.439% |

## corn：固定五个种子的独立四折结果

| 配置 | mp5 RMSE | mp6 RMSE | 两设备四属性平均 ± SD |
|---|---:|---:|---:|
| 论文 | 0.194 | 0.198 | 0.1961 ± 0.0022（Table V） |
| 上一版 | 0.1998 | 0.2054 | 0.2026 ± 0.0006 |
| 兼容 + 源训练折 X 缩放 | 0.2007 | 0.2058 | 0.2032 ± 0.0010 |
| 兼容 + 作者全无标签 X 缩放 | 0.2010 | 0.2049 | 0.2030 ± 0.0014 |

corn 的 all-unlabelled X 缩放恢复作者的传导式协议：全部设备无标签特征拟合 X scaler。它不是测试标签泄漏；未来未知样本的纯归纳部署仍应另行评估。全部配置保留 train-only Y scaler、每折全新监督模型和线性回归头。为避免跨折标签污染，不恢复作者跨折沿用已监督网络的行为，故 corn 仍不是原脚本逐项复刻。

## 修改依据与边界

- 作者 yan_ae.py 的权重范围为 ±4√(6/(输入维度+输出维度))；兼容模式按其矩阵方向使用 NumPy RandomState 生成。上一版标准 Xavier 不是仅由框架版本引起的差异。
- 按作者的共享 RNG 顺序生成各层、DCAE 校正矩阵和新预测头；保留先随机生成 Wd0 再置零的随机数消耗。
- 兼容预训练恢复最多 1000 次随机 CG：损失、梯度分别采样，并保留每次 callback 的额外损失计算。Theano RandomStreams 的两级种子展开依据 0.9 官方源码。
- 确定性监督微调与最终 DCAE 仍用 CG 1000 次；损失系数、校准对和最终分类器均未调参。
- 保留 result.x 显式恢复、正确偏置维度、稳定 log-softmax；不人为重现可能留下试探参数的行为。
- 未执行原 Python 2.7 / Theano 环境，不能声称两框架逐参数对齐。SciPy 线搜索、BLAS 数值误差及原论文运行环境未完全固定。
- 更接近论文均值不等于算法显著改进，也不能用汇总均值证明统计等效。

## 优化停止状态

### 上一版 / 5 seeds

源监督阶段：{'Maximum number of iterations has been exceeded.': 1, 'Optimization terminated successfully.': 4}

最终 DCAE：{'Maximum number of iterations has been exceeded.': 45}

### 上一版 / 10 seeds

源监督阶段：{'Maximum number of iterations has been exceeded.': 4, 'Optimization terminated successfully.': 6}

最终 DCAE：{'Maximum number of iterations has been exceeded.': 90}

### 仅原始初始化 / 5 seeds

源监督阶段：{'Optimization terminated successfully.': 5}

最终 DCAE：{'Maximum number of iterations has been exceeded.': 45}

### 仅原始初始化 / 10 seeds

源监督阶段：{'Optimization terminated successfully.': 10}

最终 DCAE：{'Maximum number of iterations has been exceeded.': 90}

### 原始流程兼容 / 10 seeds

源监督阶段：{'Optimization terminated successfully.': 10}

最终 DCAE：{'Maximum number of iterations has been exceeded.': 90}

逐层随机预训练：{'Desired error not necessarily achieved due to precision loss.': 20}

随机 CG 的 precision loss 是保留原流程后的数值局限，不能写成已收敛；固定 1000 次预算耗尽也不代表满足梯度阈值。

## 运行命令

```powershell
.\.venv\Scripts\python.exe run_uci.py --fidelity legacy --seeds 0 1 2 3 4 5 6 7 8 9 --output results/uci-legacy-repeat
.\.venv\Scripts\python.exe run_corn.py --fidelity legacy --x-scaling all-unlabelled --seeds 0 1 2 3 4 --output results/corn-legacy-repeat
```

默认 --fidelity modern 保留上一版行为；legacy-init 是初始化对照而非建议按分数择优的模型。兼容模式用于忠实于公开实现的复现。原五种子报告 RESULTS.md 保留作为历史测量，本报告为后续复核。

来源：作者代码 reference/yan_ae.py、yan_sae.py、dcae.py、test_uci.py、test_corn.py；论文 DOI https://doi.org/10.1109/TIM.2016.2573078；Theano 随机流 https://github.com/Theano/Theano/blob/rel-0.9.0/theano/tensor/shared_randomstreams.py。
