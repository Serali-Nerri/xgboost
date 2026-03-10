# CFST柱极限承载力预测 - XGBoost Pipeline

## 项目概述

本项目使用 `XGBoost` 构建一个面向混凝土填充钢管（CFST）柱极限承载力预测的机器学习流水线，覆盖以下当前已实现的能力：

- 从 CSV 加载数据并提取报告目标列
- 运行时补算 `Npl (kN)`、`psi = Nexp / Npl`、`b/h`、`L/h`、`axial_flag`、`section_family`
- 先划分训练/测试集，再进行预处理，避免预处理数据泄漏
- 按配置剔除指定特征列
- 训练 `XGBRegressor`
- 可选使用 `Optuna` 做超参数搜索
- 在交叉验证各折内部切验证集支持 `early_stopping_rounds`
- 按 `config.cv` 中的 `n_splits` / `shuffle` / `random_state` 执行交叉验证
- 以 `CV` 复合目标选择模型，并在 `train_full` 上重训最终模型
- 输出训练/测试指标、交叉验证结果、可比较的 `regime_analysis`、图表与模型产物
- 从已保存模型目录加载模型并进行 CSV 批量预测

## 当前实现的核心特性

- **模块化结构**：`src/` 下按数据加载、预处理、训练、评估、预测、可视化拆分
- **严格参数入口**：XGBoost 参数仅允许从 `config.model.params` 读取
- **上下文隔离的最优参数复用**：通过 `context_hash` 约束 `best_params.json` 的复用范围
- **目标空间分离**：训练可在 `psi` / `log(psi)` 空间进行，最终统一回到 `Nexp` 空间报告
- **多指标评估**：当前实现 `RMSE`、`MAE`、`R²`、`MAPE`、`COV`
- **可比较的 regime analysis**：先在训练集拟合 regime schema，再对 train/test 共用同一套区间
- **训练产物保存**：保存模型、预处理器、特征名、训练元数据与评估报告

## 环境要求

- Python 3.8+
- 推荐在虚拟环境中运行

依赖见 `requirements.txt`，主要包括：

- `pandas`
- `numpy`
- `xgboost`
- `scikit-learn`
- `optuna`
- `matplotlib`
- `seaborn`
- `pyyaml`
- `joblib`

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 项目结构

```text
xgboost-CFST/
├── config/
│   └── config.yaml
├── data/
│   └── raw/
├── logs/
├── output/
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── model_trainer.py
│   ├── evaluator.py
│   ├── predictor.py
│   ├── visualizer.py
│   └── utils/
├── tests/
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## 训练流程

当前训练脚本 `train.py` 的真实流程为：

1. 读取 `config/config.yaml`
2. 从 CSV 加载报告目标 `Nexp (kN)`，并按 `target_mode` 构造训练目标
3. 运行时补算 `Npl / psi / b/h / L/h / axial_flag / section_family`
4. 先划分 `train/test`
5. 在 `train_full` 上执行 `CV` / `Optuna`；各 fold 内部按 `validation_size` 切验证集用于早停
6. 使用 `CV` 复合目标选择参数，并根据各 fold `best_iteration` 选取最终 `n_estimators`
7. 仅在完整 `train_full` 上 `fit` 预处理器与最终模型，不再永久留出最终验证集
8. 统一在 `Nexp` 空间计算训练 / 测试 / CV 指标
9. 在训练集拟合 regime schema，并对 train/test 应用同一套 schema
10. 保存模型、元数据、评估报告与图表

运行训练：

```bash
python train.py --config config/config.yaml
```

指定输出目录：

```bash
python train.py --config config/config.yaml --output output/xgboost_model
```

## 配置说明

主配置文件为 `config/config.yaml`。

如果你希望查看更完整的参数释义、配置阅读建议与各字段用途说明，请参见：`doc/配置参数说明.md`。
CFST 字段释义与历史特征筛选说明请参见：`doc/CFST字段与特征说明.md`。

关键配置项：

```yaml
data:
  file_path: "data/raw/feature_parameters_unique.csv"
  target_column: "Nexp (kN)"
  target_mode: "psi_over_npl"
  target_transform:
    enabled: true
    type: "log"
  columns_to_drop:
    - "b (mm)"
    - "h (mm)"
    - "r0 (mm)"
    - "t (mm)"
    - "L (mm)"
    - "Npl (kN)"
  test_size: 0.2
  random_state: 42

model:
  params:
    objective: "reg:squarederror"
    max_depth: 5
    learning_rate: 0.05
    n_estimators: 1200
    min_child_weight: 10
    subsample: 0.8
    colsample_bytree: 0.75
    reg_alpha: 0.5
    reg_lambda: 2.0
    gamma: 0.05
    random_state: 42
    tree_method: "hist"
    device: "cpu"
    n_jobs: -1
  use_optuna: true
  n_trials: 400
  optuna_timeout: 14400
  best_params_path: "logs/best_params.json"
  early_stopping_rounds: 100
  eval_metric: "rmse"
  validation_size: 0.15
  selection_objective:
    metric_space: "original_nexp"
    rmse_normalizer: "mean_actual"
    cov_threshold: 0.10
    r2_threshold: 0.99
    cov_weight: 2.0
    r2_weight: 2.0

cv:
  n_splits: 5
  random_state: 42
  shuffle: true
```

说明：

- 更完整的参数释义文档：`doc/配置参数说明.md`
- CFST 字段释义与历史特征筛选说明：`doc/CFST字段与特征说明.md`
- 当前代码强制要求 XGBoost 参数定义在 `config.model.params`。
- `config.cv` 当前会同时控制两条路径：`Optuna` 调参时使用的交叉验证折分，以及训练阶段输出的交叉验证报告。
- `cv.n_splits` 控制折数，`cv.shuffle` 控制是否打乱样本，`cv.random_state` 在 `shuffle: true` 时控制折分复现性。
- `target_mode: psi_over_npl` 表示模型学习 `psi = Nexp / Npl`，但最终仍回到 `Nexp` 空间汇报指标。
- `target_transform` 当前作用于训练目标，而不是直接作用于报告目标。
- 当 `use_optuna: true` 时，训练会先用 `CV` 复合目标调参，再在 `train_full` 上重训最终模型。
- 当 `use_optuna: false` 时，如果 `logs/best_params.json` 与当前 `context_hash` 匹配，则会自动加载最优参数。

## 训练输出

如果运行：

```bash
python train.py --config config/config.yaml --output output/xgboost_model
```

则当前代码会在 `output/xgboost_model/` 下生成类似产物：

```text
output/xgboost_model/
├── xgboost_model.pkl
├── preprocessor.pkl
├── feature_names.json
├── training_metadata.json
├── evaluation_report.json
└── plots/
    ├── xgboost_model_train_predictions_scatter.png
    ├── xgboost_model_train_residuals.png
    ├── xgboost_model_train_error_distribution.png
    ├── xgboost_model_train_feature_importance.png
    ├── xgboost_model_train_feature_ranking.csv
    ├── xgboost_model_train_feature_ranking.txt
    ├── xgboost_model_test_predictions_scatter.png
    ├── xgboost_model_test_residuals.png
    ├── xgboost_model_test_error_distribution.png
    ├── xgboost_model_test_feature_importance.png
    ├── xgboost_model_test_feature_ranking.csv
    └── xgboost_model_test_feature_ranking.txt
```

## 评估指标解释

当前 `Evaluator` 实现了以下指标：

- `RMSE`
- `MAE`
- `R²`
- `MAPE`
- `MSE`
- `max_error`
- `COV`

其中 `COV` 基于预测值与真实值之比的均值和样本标准差计算，用于描述预测离散性。

## 预测使用手册

当前 `predict.py` 的真实行为是：

- 从模型目录加载 `xgboost_model.pkl`、`preprocessor.pkl`、`feature_names.json`、`training_metadata.json`
- 读取输入 CSV
- 按训练元数据自动补算缺失的派生列
- 当模型主线为 `psi_over_npl` 时，输入不需要提供 `psi` 或 `Nexp (kN)`；只要输入特征足够补算 `Npl (kN)` 即可
- 在需要时将模型输出从 `psi` / `log(psi)` 恢复到 `Nexp`
- 进行单条或批量预测
- 可选导出 CSV

### 批量预测

```bash
python predict.py --model output/xgboost_model --input data/raw/all.csv --output output/predictions.csv
```

参数说明：

- `--model`：模型目录路径
- `--input`：输入 CSV 路径
- `--output`：预测结果输出路径，可选
- `--single`：单条预测模式；仍然需要提供 CSV，脚本会只使用第一行
- `--verbose`：输出更详细日志

### 单条预测

当前实现不是交互式输入，而是：

```bash
python predict.py --model output/xgboost_model --input data/raw/one_row.csv --single
```

在 `--single` 模式下：

- 输入文件仍然必须是 CSV
- 如果 CSV 超过 1 行，脚本会只使用第一行
- 返回结果为单个数值预测

### 预测输出

当前导出的 CSV 默认包含：

- 原始输入特征列
- 新增的 `prediction` 列

也就是说，输出通常类似：

```csv
fc (MPa),fy (MPa),Ac (mm^2),...,prediction
40.5,350.2,10000,...,2850.7
```

## Optuna 最优参数持久化

当前项目支持：

- 将 Optuna study 持久化到 SQLite
- 将最优参数保存到 `logs/best_params.json`
- 在上下文哈希匹配时自动复用最优参数

典型流程：

```bash
python train.py --config config/config.yaml
```

当 `use_optuna: true` 时，会：

1. 在训练数据上进行调参
2. 将最优参数保存到 `logs/best_params.json`
3. 在同一轮训练中使用最优参数重训最终模型

## 已知限制

以下内容是当前代码的已知限制，写论文或报告时需要明确区分：

### 1. `psi_over_npl` 推理仍依赖足够的原始特征

当训练主线是 `psi = Nexp / Npl` 时，推理脚本会自动把模型输出恢复回 `Nexp`，但前提是输入 CSV 里至少包含足够补算 `Npl (kN)` 的特征列：

- `As (mm^2)`
- `Ac (mm^2)`
- `fy (MPa)`
- `fc (MPa)`

如果这些列缺失，推理脚本无法完成 `psi -> Nexp` 的恢复。

### 2. `sqrt` 目标变换不是当前默认主线

当前代码已经具备 `sqrt` 训练目标变换与逆变换路径，但它不是本阶段默认实验主线，且测试覆盖明显少于 `log(psi)`。

因此：

- 默认实验仍应使用 `target_mode: psi_over_npl`
- 默认训练变换仍应使用 `target_transform.type: log`
- 如需启用 `sqrt`，应先补充额外验证

### 3. 当前 CV 结果更适合作为调参参考，不宜直接作为无偏论文结论

当前实现中：

- `Optuna` 在训练集上使用交叉验证做调参
- 然后又在同一训练集上计算交叉验证结果并输出
- 这两条路径现在都会遵守 `config.cv` 中配置的 `n_splits` / `shuffle` / `random_state`

这会导致 CV 结果偏乐观。对论文写作而言，更建议：

- 以独立测试集指标作为主要泛化结果
- 如果需要更严格的泛化估计，后续采用 `nested CV`

### 4. 当前版本未加入目标定义域与边界校验

当前代码未额外检查：

- `log` 变换下目标值是否全部大于 0
- `sqrt` 变换下目标值是否全部非负
- 数据是否存在超出经验适用范围的边界工况

这部分更适合结合你的人工数据筛选规则、工程经验和后续经验公式一起处理。

## 论文使用建议

如果你打算将本项目用于论文实验，当前更稳妥的建议是：

- 优先把数据集筛选规则写清楚，尤其是是否排除了非经典工况
- 优先报告独立测试集上的 `RMSE`、`MAE`、`R²`、`COV`
- 把交叉验证结果表述为“训练阶段模型选择参考”，避免当作完全无偏的最终泛化结论
- 如果关闭目标变换并重新训练，请确保论文中的推理结果、评估指标和工程解释全部在同一物理量空间下进行

## 测试与验证

运行测试：

```bash
pytest -q
```

运行语法检查：

```bash
python -m compileall train.py predict.py src tests
```

## 后续建议（本轮未实现）

以下改动很值得作为下一轮工作：

- 将训练/验证/测试三套指标完全拆开报告
- 将当前 CV/Optuna 方案升级为更严格的 `nested CV`
- 按你的数据筛选规则增加目标定义域与边界工况校验
- 如果未来要恢复特征选择功能，建议在代码落地后再补回对应文档
