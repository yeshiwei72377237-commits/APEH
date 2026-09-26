# Model Card · fitness_model（2026-09-10）

> 依据评审建议，对 fitness 闭环模型做**诚实降级与边界标注**（见第三方《参赛代码审计报告》
> 及本仓库 [`attributions.md`](../../../attributions.md)）。本卡以"是什么 / 不是什么"的方式
> 写明能力边界，避免被当成高精度 fitness 预测器。

---

## 一句话定位

本模型是一个**位置盲的组成基线（composition baseline）**：仅用"序列里用了哪些氨基酸+
理化性质聚合统计"来给 ProteinMPNN 候选**排序**，用于跑通推荐闭环、辅助筛序；
**不能**作为热稳定性 / 催化活性 / 生物安全的可靠预测器。

| 项目 | 结论 |
|---|---|
| 模型类型 | 梯度提升回归（GradientBoostingRegressor），蛋白特异回归 |
| 输入 | 一条完整突变序列（长度需等于训练 WT 长度） |
| 任务 | 预测相对活性分数（单位任意，仅用于排序） |
| 输出 | `scored_candidates.csv` → `topN_recommendations.csv` |
| 是否预训练权重 | 否（在本团队湿实验活性数据上从零训练） |
| 依赖 | 仅 numpy + scikit-learn + pandas + openpyxl（零外部模型依赖） |

---

## 关键特征与边界

### 特征（`featurize.py`，默认 30 维）
| 子块 | 维度 | 是否含位置信息 |
|---|---|---|
| 氨基酸组成 | 20 | **否**（全局占比，无序） |
| 理化性质（均值+标准差） | 10 | **否**（聚合统计，无序） |
| （可选）二肽组成 | 400 | 仅相邻配对占比，非空间/结构；默认关闭 |

> ⚠️ **位置盲 / 结构盲**：特征不含残基坐标、接触图、二级结构、进化序列位置或突变位点信息。
> 因此它**无法区分**构成相同但空间排布不同的序列，也无法解释"为什么这个位点这么改"。
> 较成熟的、可解释的上游特征/模型见 [`../Model/`](../Model/)（pySAR + SHAP），本闭环与其是
> 上下游关系而非同一强度。

### 蛋白特异约束
只对**与训练 WT 同源、链长一致**的蛋白有效。`score.py` 严格校验 `len(候选) == len(训练WT)`，
不符则 `score_invalid=1` 且不产出分数。当前落盘版本对应 **2UWF 356-aa 构建体**
（`artifacts/wt.txt`）；360-aa 全长前体版本备份在 `../artifacts_backup_360wt/`。
两版编号迁移规则见 [`2UWF_construct_mapping.md`](2UWF_construct_mapping.md)。

---

## 训练数据与评估

| 项目 | 数值 |
|---|---|
| 训练集（当前默认） | `data/raw/Supplementary-Material-2_2UWF.xlsx`，126 条复突变活性标签 |
| 旧训练集（可切换） | `data/raw/enzyme_activity.xlsx`，223 → 去重 204 条（≠本仓库 raw 文件的实际行数，属"清洗后有效样本"计数；口径见 `data/README.md` 数字口径一览） |
| 交叉验证 | 10-fold KFold，报告 R² / MAE / RMSE / Spearman |
| 当前 CV 指标 | R² ≈ **0.31–0.34**、Spearman ≈ **0.64–0.65**（126 条、测量噪声大） |

> 该指标来自**同一蛋白小样本（126 条）**、测量噪声较大的湿实验标签，属"能跑通闭环"的
> 入门水平。**不要**据此推断它在别的蛋白、别的标签定义或外推场景下仍有此精度。

### ⚠️ 该指标已查明主要由「突变个数」混淆项贡献（2026-09-10）

对 R²≈0.31–0.34 做了混淆控制后，**本模型的真实能力比上表更低**：

1. 本数据集标签与**突变个数**强相关（`pearson(突变个数, activity) = 0.743`；
   52/126 为单点突变，其余是 2~9 位点的理性叠加组合，标签均值随个数单调上升）。
2. **仅用「突变个数」这 1 个整数特征**，同一套 10 折 CV 就得 **R² = 0.560 / Spearman = 0.674**
   ——**高于本模型的 0.339**。
3. 在**突变个数恒定的 52 条单点突变**子集上（混淆项被消除），本模型：

   | 方法（n=52 单点突变） | R² | Spearman |
   |---|---|---|
   | **本模型 30 维 + GBR** | **−0.527** | **+0.128** |
   | 本模型 30 维 + Ridge | −0.033 | −0.386 |
   | 对照：ESM-2 1280 维 + GBR | +0.305 | +0.476 |

   **R² = −0.527 意味着差于直接预测均值；Spearman = +0.128 与 0 无异。**
   即本模型**基本不具备可用的单点突变效应预测能力**，其全样本 R² 主要在拟合"改了几个位点"。

**因此**：本模型只应在"已验证能在同源范围内跑通排序闭环"的意义上被引用；
**不得**把 R²=0.339 / Spearman=0.654 当作单点突变效应的预测精度。

- 复跑：`python src/code/fitness_model/baseline_esm2_controls.py`
- 结果：`results/baseline_esm2/controls.json`、`results/baseline_esm2/metrics.json`
- 经过：`wiki/AI-Computational-Methods.md` §4.3

---

## 已知局限 / Known Limitations

1. **位置盲**：如上，无法感知残基相对位置与空间结构，同"组成"不同"排布"无法区分。
   **已定量证实**：在突变个数恒定的 52 条单点突变上 R²=−0.527 / Spearman=+0.128
   （见上方「该指标已查明主要由『突变个数』混淆项贡献」一节）。
2. **外推差**：对训练样本稀少的小众亚家族、大幅跨家族重构、完全脱离进化背景的人工蛋白，
   预测置信度显著下降。建议仅在"贴近训练蛋白的同源改造"范围内使用并人工复核。
3. **量纲任意**：分数是活性标签的任意线性映射，只用于**排序**，不可当作绝对活性值。
4. **不能评估安全性**：本模型不包含 BLAST / 毒性 / 抗菌 / 催化位点等安全检查。
   本仓库**没有**任何生物安全筛查（见 [`safety/ai_model_safety_report.md`](../../../safety/ai_model_safety_report.md) §4）。
   下游门控**曾**因两套词表不一致而失效（门控只匹配人工复核词表 `HOLD`/`BLOCK`，而自动审核
   写出的是 `HOLD_FOR_STRUCTURE_REVIEW`），该缺陷已由第三方审计报告指出；现已抽出
   `src/code/fitness_model/safety_gate.py` 统一词表并**修复**——命中项禁止打分（`score_invalid=2`）
   并排除出 TopN。本次 golden run 因生成侧已用 `--omit-aas PC`，19 条候选全部为
   `PASS_TO_STRUCTURE_MODEL`，门控未被实际命中（详见该报告 §3.2）。
5. **小样本过拟合风险**：默认关闭二肽组成（400 维对 126–223 条样本易过拟合）；`train.py`
   默认参数仍指向 360 全长前体，若直接用默认参数重训会回到旧版本，需显式指定 2UWF 新集。

---

## 与上游 Model（pySAR）的关系

- 本闭环：轻量、自包含、`predict()` 秒级、面向 ProteinMPNN 候选打分。
- 上游 `Model/`：pySAR（第三方 MIT，AJ McKenna）提供数百维 AAindex 特征 + SHAP 可解释性，
  标签更多后可替换本闭环的 `featurize.py`（train/score/select 无需改动，只换特征维度）。

---

## 复现 / 回滚

- 训练：`python train.py`（⚠️ 默认参数仍指向 **360-aa 全长前体** `Supplementary-Material-2.xlsx`；
  要重建部署的 356-aa 版必须显式指定 `--xlsx data/raw/Supplementary-Material-2_2UWF.xlsx
  --wt-file src/code/fitness_model/artifacts/wt.txt`，否则会覆盖 `artifacts/`，
  详见 [`2UWF_construct_mapping.md`](./2UWF_construct_mapping.md) §5）。
- 打分：`python score.py --dir <run_dir> --artifacts artifacts`
- 选择：`python select_top_mutants.py --input_dir <run_dir> --output_csv <out.csv>`
  （参数名为**下划线**，且 `--output_csv` 为必填）
- 回滚：旧模型在 `artifacts_backup_360wt/`。

*最后更新：2026-09-10*
