# Copilot Instructions for JML (Julia ML Project)

## 项目架构与数据流
- 本项目以 Home Credit Default Risk 竞赛数据为核心，主要处理和特征工程流程均在 `test.jl` 中实现。
- 数据目录为 `raw_data/`，所有原始 CSV 文件均存放于此。
- 主表为 `application_train.csv` 和 `application_test.csv`，以 `SK_ID_CURR` 为主键。
- 其余表通过主键或外键（如 `SK_ID_BUREAU`, `SK_ID_PREV`）与主表关联，需聚合后合并。

## 关键文件与目录
- `test.jl`：主脚本，包含数据加载、聚合、特征工程等核心逻辑。
- `raw_data/`：存放所有原始数据文件。
- `Manifest.toml` 和 `Project.toml`：Julia 包管理文件，定义依赖。

## 依赖与环境
- 主要依赖：`DataFrames`, `CSV`, `Pipe`, `Statistics`, `MLJ`, `ScientificTypes`。
- 使用 Julia 1.12.0，建议通过 `Pkg.activate()` 和 `Pkg.instantiate()`恢复环境。

## 典型开发流程
1. 数据预处理：
   - 先加载主表和所有子表。
   - 按照注释中的数据关系图，逐层聚合子表（如 `bureau_balance` → `bureau` → `application_train`）。
   - 聚合方式多用 `groupby` + `combine`，常见统计如 `mean`, `sum`, `count`, `maximum`。
2. 特征工程：
   - 聚合后特征命名采用 `表名_字段名_统计方式`（如 `BB_MONTHS_BALANCE_MEAN`）。
   - 合并表时优先用 `leftjoin`，保持主表完整。
3. 模型训练与评估：
   - 目前未见模型训练代码，后续如需集成建议用 `MLJ`。

## 项目约定与风格
- 代码风格偏向管道式（Pipe.jl），多用 `@pipe` 简化链式操作。
- 聚合与合并逻辑高度模块化，便于扩展新特征。
- 特征命名、表间关系均有详细注释，建议保持注释风格。
- 所有路径均相对项目根目录，便于迁移。

## 常用命令
- 启动 Julia REPL 并激活环境：
  ```julia
  using Pkg
  Pkg.activate(".")
  Pkg.instantiate()
  ```
- 运行主脚本：
  ```julia
  include("test.jl")
  ```

## 扩展建议
- 新增特征时，优先在子表聚合后合并到主表。
- 如需调试，建议用 `describe(df)` 或 `first(df, 5)` 查看中间结果。
- 若集成模型训练，建议新建 `model.jl`，并在 `Project.toml` 添加相关依赖。

## 参考
- 主要数据关系与特征工程流程见 `test.jl` 注释。
- 依赖包版本见 `Manifest.toml`。

---
如有不清楚或遗漏的部分，请反馈具体需求或代码片段，以便进一步完善说明。