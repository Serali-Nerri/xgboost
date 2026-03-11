# raw psi vs log(psi) 完整训练对比报告

生成时间: 2026-03-11

## 1. 目的

本次实验的目标是，在当前 `psi = Nexp / Npl` 主线下，直接比较两种训练目标定义：

- `raw psi`
- `log(psi)`

两条线都按照完整流程执行：

- 使用 `psi_over_npl` 作为建模目标模式
- 使用样本级 `regression_stratified` 切分
- 外层指标统一回到原始 `Nexp (kN)` 空间汇报
- Optuna 超参数调优轮次统一为 `200`
- 通过 CV 复合目标 `J` 选模
- 在 `train_full` 上按 CV 得到的最终轮数重训

## 2. 实验配置

### 2.1 数据与切分

- 数据文件：`data/processed/2026.3.9-11864_feature_parameters_raw.csv`
- 报告目标：`Nexp (kN)`
- 训练目标模式：`psi_over_npl`
- 测试集比例：`0.2`
- 随机种子：`42`
- 分层策略：`regression_stratified`
- 主分层轴：目标分位数 `10 bins`
- 辅助分层轴：`lambda_bar`，`3 bins`

两条实验的 `stratification_metadata` 一致：

- `n_strata = 30`
- `minimum_count_observed = 71`
- `used_auxiliary_features = [lambda_bar]`

### 2.2 配置文件

- `raw psi`：`config/experiments/raw_original_metric.yaml`
- `log(psi)`：`config/experiments/log_original_metric.yaml`

### 2.3 调优口径

两条线都采用：

- `use_optuna = true`
- `n_trials = 200`
- `optuna_metric_space = original`
- `selection_objective.metric_space = original_nexp`

复合目标为：

`J = nrmse + 2 * max(0, cov - 0.10) / 0.10 + 2 * max(0, 0.99 - r2) / 0.01`

其中：

- `nrmse = rmse / mean(y_true_original)`
- 所有 `rmse / r2 / cov` 都在原始 `Nexp` 空间计算

### 2.4 调参中心点

本次没有沿用同一个中心点强行搜索。两条线分别以各自配置文件中的 `model.params` 作为 Optuna 搜索中心点：

- `raw psi` 使用其 raw 基线参数
- `log(psi)` 使用其 log 基线参数

这满足“调参中心点按实验语义分别设置”的要求。

## 3. 结果总表

所有主指标均为原始 `Nexp (kN)` 空间，除非特别说明。

| 实验 | 训练目标 | CV J | CV RMSE | CV R² | CV COV | 最终重训轮数 | Test RMSE | Test R² | Test COV | 过拟合比 RMSE(train/test, original) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw psi | `psi` | 2.1756 | 355.5047 | 0.9818 | 0.1178 | 3756 | 262.7955 | 0.9874 | 0.0990 | 2.3023 |
| log(psi) | `ln(psi)` | 1.5051 | 333.3728 | 0.9839 | 0.1022 | 3806 | 229.1109 | 0.9904 | 0.0905 | 1.9694 |

### 3.1 测试集差值: `log(psi) - raw psi`

| 指标 | 差值 |
| --- | ---: |
| Test RMSE | -33.6846 kN |
| Test R² | +0.00303 |
| Test COV | -0.00854 |
| CV J | -0.67049 |
| CV RMSE | -22.1319 kN |
| CV R² | +0.00208 |
| CV COV | -0.01558 |

直接结论：

- `raw psi` 已经把测试集 `COV` 压到 `0.1` 以下，但 `R²` 仍停在 `0.9874`
- `log(psi)` 同时实现了 `R² > 0.99` 和 `COV < 0.1`
- 从 CV 复合目标和最终 holdout 指标看，`log(psi)` 明显优于 `raw psi`

## 4. 最终模型参数

### 4.1 raw psi

Optuna 最优参数：

```yaml
max_depth: 6
learning_rate: 0.04310181109201213
n_estimators: 4182
subsample: 0.681039288892301
colsample_bytree: 0.45139653967880256
min_child_weight: 8
reg_alpha: 0.03203247371355593
reg_lambda: 1.043759773892089
gamma: 0.0006366247439841869
```

CV 各折 `best_iteration`：

```text
[4110, 4181, 3699, 3756, 2474]
```

最终重训轮数：

```text
3756
```

### 4.2 log(psi)

Optuna 最优参数：

```yaml
max_depth: 5
learning_rate: 0.0451669400461485
n_estimators: 3970
subsample: 0.7238136617365515
colsample_bytree: 0.4212006920305448
min_child_weight: 4
reg_alpha: 0.012529841627285286
reg_lambda: 0.44957639473634237
gamma: 0.00013613870646182803
```

CV 各折 `best_iteration`：

```text
[3806, 3382, 3934, 3970, 2234]
```

最终重训轮数：

```text
3806
```

## 5. 训练集与测试集表现

### 5.1 raw psi

训练集表观拟合：

- RMSE = `114.1461 kN`
- R² = `0.9982`
- COV = `0.0436`

测试集：

- RMSE = `262.7955 kN`
- R² = `0.9874`
- COV = `0.0990`

### 5.2 log(psi)

训练集表观拟合：

- RMSE = `116.3339 kN`
- R² = `0.9981`
- COV = `0.0405`

测试集：

- RMSE = `229.1109 kN`
- R² = `0.9904`
- COV = `0.0905`

### 5.3 解释

`log(psi)` 的训练集 RMSE 略高于 `raw psi`，但测试集明显更好。这说明：

- `log(psi)` 没有单纯追求训练集更贴合
- 它在当前样本级协议下，泛化更稳
- 这种改善不是偶然的单点提升，因为 CV 指标也同步改善

## 6. Regime 分析

### 6.1 主要坏区间对比

按 `worst_rmse_group` 对比：

| Regime | raw psi | log(psi) | 变化 |
| --- | --- | --- | --- |
| axiality | axial, 267.75 kN | axial, 230.37 kN | 明显改善 |
| scale_npl | q5, 531.58 kN | q5, 431.43 kN | 明显改善 |
| eccentricity_severity | moderate_ecc, 296.96 kN | moderate_ecc, 246.31 kN | 明显改善 |
| confinement_level | low_conf, 413.46 kN | low_conf, 284.23 kN | 改善非常明显 |

### 6.2 主要观察

1. `scale_npl_q5` 仍然是两条线共同的最大风险区。
这说明高 `Npl` 端仍是最难学的区域，虽然 `log(psi)` 已经把最坏 RMSE 从 `531.58 kN` 降到 `431.43 kN`。

2. `moderate_ecc` 和 `low_conf` 是现阶段最值得继续盯的机制区间。
`log(psi)` 在这两类上的改善幅度都很明显，说明对数变换确实缓解了偏压和低约束区域的误差扩散。

3. `section_family = obround` 在 `log(psi)` 下被标成最坏截面族，但测试样本只有 `4` 个。
这个结果可以记录，但不应过度解读。它更像“小样本告警”，不是当前主结论。

4. `axiality` 下最差组始终是 `axial`。
这不是说偏压更容易，而是当前测试集中轴压样本量更大，且该组仍承受较大的总误差贡献。后续如果做轴压/偏压双模型，这一项仍然值得优先考虑。

### 6.3 最差 COV 组

- `raw psi` 的 `axiality.eccentric`：`COV = 0.1158`
- `log(psi)` 的 `axiality.eccentric`：`COV = 0.0997`

这点非常关键。`log(psi)` 不只是整体 `COV` 更低，它把偏压子集的最差 `COV` 也压回了 `0.1` 附近。

## 7. 特征重要性

两条线的前 8 个重要特征都高度一致，核心仍然是偏心与相对偏心相关量：

### 7.1 raw psi Top 8

1. `e1/e2`
2. `e_bar`
3. `e/h`
4. `e2 (mm)`
5. `e1 (mm)`
6. `ke`
7. `r0/h`
8. `lambda_bar`

### 7.2 log(psi) Top 8

1. `e1/e2`
2. `e_bar`
3. `e/h`
4. `e1 (mm)`
5. `e2 (mm)`
6. `lambda_bar`
7. `ke`
8. `r0/h`

说明当前 `psi` 主线下，模型确实主要在学习：

- 偏心形式
- 相对偏心强度
- 稳定相关参数 `lambda_bar`
- 截面几何比例项

而不是重新退回到只靠截面尺寸硬拟合 `Nexp`。

## 8. 结论

### 8.1 本轮实验结论

在当前这套 `psi = Nexp / Npl` 主线和样本级切分口径下：

- `raw psi` 是有效基线
- `log(psi)` 是当前更优主线

更具体地说：

1. `raw psi` 已经证明 `psi` 目标本身是成立的。
即使不做对数变换，测试集也已经达到：
`R² = 0.9874, COV = 0.0990`

2. `log(psi)` 在当前协议下是更强的选择。
它最终达到：
`RMSE = 229.11 kN, R² = 0.9904, COV = 0.0905`

3. 如果当前阶段的目标是尽快拿到一条满足
`R² > 0.99` 且 `COV < 0.1`
的主线结果，那么应当优先采用 `log(psi)`。

### 8.2 需要保留的保守判断

虽然 `log(psi)` 的 holdout 已经同时满足目标，但还要注意两点：

1. CV 的 `COV` 仍然是 `0.1022`，只比阈值高一点点。
这意味着它已经很接近稳定达标，但还没有在 CV 口径上形成明显安全余量。

2. 两条线都仍然显示出一定过拟合。

- `raw psi` 原始空间 RMSE 比值：`2.3023`
- `log(psi)` 原始空间 RMSE 比值：`1.9694`

`log(psi)` 已经缓和了这个问题，但没有消除。

## 9. 建议

### 9.1 当前建议

当前建议非常明确：

- 把 `log(psi)` 作为主线配置继续推进
- 把 `raw psi` 作为对照/消融基线保留

### 9.2 下一步优先级

建议按下面顺序继续：

1. 以 `log(psi)` 为默认主线，继续做特征消融和 regime 定向改进。
2. 重点盯 `scale_npl_q5`、`moderate_ecc`、`low_conf` 三类高风险区。
3. 如果后续要追求更稳的 CV 达标，优先做面向这些 regime 的针对性增强，而不是再盲目加大全局调参轮数。
4. `raw psi` 不要删。
它已经说明“纯 `psi` 主线”具备可行性，后续写作时可作为工程解释更直观的对照组。

## 10. 产物位置

- raw 结果目录：`output/psi_over_npl_raw_original_200/`
- log 结果目录：`output/psi_over_npl_log_original_200/`
- 本报告：`doc/raw_psi_vs_log_psi_full_run_20260311.md`
