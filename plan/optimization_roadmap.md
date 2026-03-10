# XGBoost 模型当前实施路线

> 当前阶段采用样本级切分，不纳入 `Ref.No.` 分组切分。
> 当前主线目标是：以 `psi = Nexp / Npl` 建模，并在 `Nexp` 空间优化与汇报。

## 当前主线

1. 报告目标固定为 `Nexp (kN)`
2. 训练目标改为 `psi = Nexp / Npl`
3. 默认训练变换使用 `log(psi)`
4. `CV` / `Optuna` 选择目标改为原始 `Nexp` 空间复合目标
5. 最终模型在 `train_full` 上重训，不再永久留出最终验证集

## 当前已落地的关键改造

- 运行时自动补算：
  - `Npl (kN)`
  - `psi`
  - `b/h`
  - `L/h`
  - `axial_flag`
  - `section_family`
- 推理阶段会读取训练元数据，并在 `psi_over_npl` 主线下自动把模型输出恢复回 `Nexp`
- `regression_stratified` 改为搜索辅助特征子集，不再只尝试前缀组合
- `regime_analysis` 改为：
  1. 在训练集上拟合 schema
  2. 对 train/test 共用同一套 schema
- experiment suite 已统一到 `psi_over_npl` 主线，并按 `CV` 复合目标排序，而不是按测试集指标选“赢家”

## 当前默认 regime 体系

- `axiality`
- `section_family`
- `slenderness_state`
- `scale_npl`
- `eccentricity_severity`
- `confinement_level`

## 当前不在本阶段范围内

- `Ref.No.` 分组切分
- 轴压 / 偏压双模型正式训练框架
- 规范公式残差学习主线

## 下一步优先级

1. 在 `psi / log(psi)` 主线上重新跑主实验
2. 用新的 `selection_objective` 重新做 Optuna
3. 做一轮 `lambda / L/h / shape-related` 小型消融
4. 用新的 regime schema 检查偏压、细长柱和不同截面族的误差短板
5. 再决定是否进入轴压 / 偏压双模型副线
