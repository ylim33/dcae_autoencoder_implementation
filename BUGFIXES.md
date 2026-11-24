> 发布说明：本文保留本地复现实验记录。原代码文件名对应 [固定版本源码链接](reference/README.md)，上游源码未随本仓库转载；原 results/ 中的 JSON 发布在 [benchmarks/](benchmarks/)，模型与逐样本预测需本地重新训练生成。

# 原代码问题与移植决策

以下位置均对应 reference/ 中保存的作者代码；说明“错误”和“有意的数值/协议变更”两类情况。

| 项目 | 原代码依据 | 本实现处理 | 性质 |
|---|---|---|---|
| 回归输出误用 softmax | yan_sae.py 的 fine_tune_sup 不区分任务，总是构造 softmax 层；test_corn.py 用它做标准化连续值的 MSE 回归 | 回归头输出不受限的线性数值；分类使用 logits + 稳定 CE | 明确的任务实现错误 |
| 跨折复用已监督训练网络 | test_corn.py 在四折循环外创建 sae，循环内不断 fine_tune_sup；前一训练折可能包含下一测试折标签 | 每折重新初始化、预训练和微调；不携带上一折监督状态 | 潜在交叉验证污染，已修正 |
| 目标 Y 使用测试折统计量 | test_corn.py 在分折前 StandardScaler().fit(Y) | 每折只拟合源域训练折 Y；最终 Ridge 直接预测原始单位 | 更严格的标签隔离 |
| 输入 X 统计使用全设备全样本 | test_corn.py 对拼接的全部设备特征拟合 scaler | X scaler 改为每折源域训练数据；无标签特征仍允许参与预训练/重构 | 协议选择；全特征缩放在传导式场景并非必然错误 |
| SciPy 结束后未恢复 result.x | dcae.py / yan_ae.py / yan_mlp.py 的 train_scipy / pretrain_scipy 返回前没有显式写回 opt_res.x | 总是恢复 result.x；保存收敛状态；损失和梯度一次计算 | 潜在优化状态错误；并非每次运行必然触发 |
| AE 的 L2 笔误 | yan_ae.py 的 L2 把 W_prime 平方写了两遍 | 使用 encoder + decoder 的范数；权重绑定时为 2||W||² | 未绑定时是错误；绑定时两者数值相同 |
| 随机去噪与 CG | 去噪随机 mask 进入分别编译的损失/梯度函数，线搜索目标会变动 | modern 模式去噪预训练使用 Adam；legacy 模式保留分别重采样的随机 CG；确定性监督/最终阶段使用 CG | 明确记录的优化器变更，不声称逐数值复现 |
| CE 的尺度 | yan_utils.py 对类别轴求 mean 而非 sum | torch cross_entropy 再除以类别数（UCI 为 6） | 保留原始尺度，不把它当作需要纠正的错误 |
| 校准样本参与原始测试分数 | test_uci.py 对整个目标 batch 评分 | 保留论文口径，并额外输出排除校准行的结果；重复行一并排除 | 额外透明报告，不暗改原论文主指标 |

重构 MSE、配对误差保留原代码按元素取均值的尺度。原论文公式的欧氏范数写法与代码的维度均值不完全等价；本项目沿用作者实现与其超参数搭配。

8 项自动检查覆盖：独立 NumPy 公式、线性重构校正抵消、autograd 有限差分、CE 缩放、负值回归输出、优化器最终状态恢复、数据划分/标签标准化隔离、分类与回归 checkpoint 的 sklearn 预测一致性。完整实验还在每次保存模型后执行预测 roundtrip 检查。

当前不移植原工具箱中与这两项 DCAE 实验无关的稀疏 AE、mSDA、通用 minibatch 训练等接口；本项目复现的是论文 DCAE 及其公开实验。


## 后续忠实性审计

- 此前遗漏的实质差异：yan_ae.py 的浅层权重范围为 ±4√(6/(输入+输出))，不是标准 Xavier。legacy 模式恢复该范围、NumPy RandomState、按 (输入,输出) 生成再转置的顺序，以及整个 UCI 流程的共享随机流。不能把这一差异仅归于 PyTorch/Theano 版本。
- 原始 b / b_prime 构造尺寸与隐藏/可见层用途相反。CG 的 theta2Wb 首次执行会将全零向量重新分割成正确尺寸，因此原 CG 路径不一定因此报错；minibatch 路径存在维度风险。移植代码从构造时就使用正确尺寸。
- legacy 模式按 Theano RandomStreams 的两级种子展开生成 binomial mask，并保留独立 cost/gradient 调用和 callback 的随机数消耗。未运行旧 Theano 来验证逐比特一致；当前 SciPy/BLAS 的计算路径仍可能不同。
- 保留 result.x 恢复修复。随机 CG 的 precision loss 与固定预算达到上限如实记录，不声称优化已收敛。
- 新增两项测试验证原始初始化矩阵、转置、随机数消耗顺序、独立随机 cost/gradient 和返回参数恢复。总计 10 项测试。
- corn 增加显式的 all-unlabelled X 缩放选项，用于恢复作者允许无标签目标特征参与的传导式协议；Y 标签统计和每折监督模型仍严格隔离。

完整测量见 FIDELITY_RESULTS.md；上一版五种子结果不覆盖。
