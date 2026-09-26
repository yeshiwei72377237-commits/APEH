# Model · 上游特征与回归模型引擎（pySAR）

> 本文件夹由原仓库根目录的 `Software/` 整体封装而来，是 APEH 的**上游计算内核**：把一条 / 一批突变序列编码成高维理化特征，并按「活性 / 热稳定性」两套任务训练回归模型，输出单模型与集成结果。
>
> 它与下游的 [`../fitness_model/`](../fitness_model/) 是**上下游协作**关系：本引擎提供更丰富、可解释（SHAP）的特征与模型；`fitness_model` 则是一个轻量、零依赖、面向 ProteinMPNN 候选的快速打分闭环。

> ⚠️ **第三方归属（重要）**：本目录的 `pySAR/` 是来自开源项目 **pySAR v2.3.4**（作者 **AJ McKenna**，`https://github.com/amckenna41/pySAR`，**MIT 协议**），**不是本团队自研的代码**。本团队做的是：将该第三方库整体纳入本仓库的 `Model/` 并做封装、适配与调用（特征选择 `Trained/`、入口 `run_prediction_pipeline.py` 等为本团队所写）。如需精确区分「我们写的」与「上游的」，请以此为准，并见根目录 [`attributions.md`](../../../attributions.md)。

---

## 1. 定位 / Role

APEH 的 AI 计算分为 **上游特征 / 模型引擎** 与 **下游打分 / 推荐闭环** 两层。本文件夹承担前者。

| | 上游 Model（本文件夹） | 下游 fitness_model |
|---|---|---|
| 定位 | 特征工程 + 回归模型引擎 | 候选打分 + TopN 推荐闭环 |
| 特征 | AAindex 理化指数 + 氨基酸组成 + 二肽组成 + 联合三元组（conjoint triad） | 30 维确定性编码（20 氨基酸组成 + 10 理化统计） |
| 特征维度 | 数百维（经 SFS + SHAP 选出核心特征） | 30 维 |
| 算法 | Bagging / GradientBoosting / XGBoost（活性 A + 热稳定性 H 两套） | GradientBoostingRegressor（单一 fitness 回归） |
| 训练方式 | 每次运行**重新训练**（特征选择配置在 `Trained/`） | 一次性训练并落盘 `artifacts/model.pkl` |
| 用时 | 较重（需 pySAR + 每次重训） | 轻（`predict()` 秒级，只依赖 numpy/sklearn/pandas） |
| 可解释性 | 有（SHAP + 特征重要性排序） | 无 |

> 一句话概括：**Model 回答「这个改法理论上活性 / 稳定性多好、为什么（SHAP）」；fitness_model 回答「这批候选里该保留哪几条」。**

---

## 2. 目录结构 / Layout

```text
src/code/Model/
├── pySAR/                        # pySAR 特征 / 模型库（第三方 MIT 开源，AJ McKenna；本团队封装适配）
│   ├── pySAR.py                  #   主类 PySAR：特征编码、模型构建、SHAP/SFS
│   ├── descriptors.py            #   描述符计算
│   ├── encoding.py               #   序列编码
│   ├── model.py                  #   回归模型（Bagging/GBR/XGB 等）
│   └── ...
├── run_prediction_pipeline.py    # ★ 统一入口：输入序列 → 单模型结果 → 集成结果
├── pred_software.py              #   单模型预测：encode_aai_descriptor → predict
├── ensemble_result.py            #   三种算法集成
├── input_parser_txt.py           #   文本/序列输入 → 标准预测 txt
├── input_parser_csv.py           #   CSV 输入解析
├── data.json                     #   训练代码配置（dataset/model/descriptors）
├── data_Pred.json                #   预测代码配置
├── test_activity_1.csv           #   训练活性数据集（Sequence + Fitness）
├── Trained/                      # 特征选择配置（不是训练好的模型权重）
│   ├── A_shap_sfs/               #   活性(A)：SFS 选中的特征列号
│   ├── A_shapvalues_rank/        #   活性(A)：SHAP 排序（特征名→原始列号）
│   ├── H_shap_sfs/               #   热稳定性(H)：SFS
│   └── H_shapvalues_rank/        #   热稳定性(H)：SHAP 排序
├── Test_Sample/                  # 示例输入（TXT/CSV/XLSX/FASTA）
└── 脚本功能说明.docx             # Software 时期的功能说明文档
```

---

## 3. 与 fitness_model 的上下游 / Upstream–Downstream

### 3.1 数据链路

```text
     上游 Model(pySAR, 数百维特征)                    下游 fitness_model(30维轻量基线)
        │                                                  │
        │  提供特征选择配置 / 训练产物                     │  面向 ProteinMPNN 候选打分
        │  提供可解释的 SHAP                                │  输出 topN_recommendations
        └──────────── 特征可复用 ◄────────────∴────────────┘
                        （标签更多后，可用本引擎更丰富的特征替换 fitness_model 的 featurize.py）
```

- **上游（Model）**：从突变序列重建出**完整突变序列**，经 pySAR 编码为高维特征，用 `Trained/` 里的 SFS / SHAP 配置筛选核心特征，并对活性、热稳定性**分别**训练三种回归模型。特征更全、可解释，但依赖 pySAR、每次预测需要重训，无法直接加载落盘模型。
- **下游（fitness_model）**：一条 `model.pkl` 直接 → `predict()`，把任意长度突变序列映射到固定 30 维向量，给 ProteinMPNN 候选序列打分，`score.py` → `select_top_mutants.py` → `topN_recommendations.csv`，并已挂进 `run_all.py` 的 Step 10。

### 3.2 为什么两个都要

`fitness_model` 的 [`README.md`](../fitness_model/README.md) 明确标注其定位是**「能跑通闭环的入门基线」**（126 条标签、CV R²≈0.31 / Spearman≈0.64，360-aa 全长前体版；2UWF 356-aa 部署版为 R²=0.339 / Spearman=0.654），并用一句话点出升级方向：

> 更多标签后用 pySAR 完整版替换 `featurize.py` 即可，train/score/select 无需改动（只换特征维度）。

也就是说：下游闭环的**流水线骨架**是稳定的，未来的精度提升主要来自**替换上游特征**。本 Model 引擎正提供了那个「更完整的特征来源」——两者不是竞争关系，而是同一目标的两代实现，**上游出特征与模型，下游跑推荐闭环**。

### 3.3 精度瓶颈提示

当前（126 条、测量噪声大）瓶颈在**数据量而不在方法**，本引擎的多维特征在小样本下相对 30 维基线的提升有限。等湿实验标签扩充后，再把 `featurize.py` 替换为 pySAR 特征，收益才会显现。

---

## 4. 快速开始 / Quick Start

> 注意：脚本使用相对路径（`data.json`、`Trained/`、`software_temp/`、`software_output/`），请在**本目录下**运行。

```bat
:: 同步预测活性 + 热稳定性（task=1）；只预测活性 task=2；只预测热稳定性 task=3
python run_prediction_pipeline.py
```

主入口为 `run_prediction_pipeline(inputs, wt_seq, task, user_save_dir, ...)`：

- `inputs`：文件路径 / 路径列表 / 序列字符串 / FASTA 文本；
- `wt_seq`：野生型序列（预测突变需先应用突变）；
- `task`：`1` 活+热 / `2` 活性 / `3` 热稳定性；
- `user_save_dir`：最终集成结果保存目录（脚本自动建「活性」「热稳定性」子目录）。

输出约定：
- 单模型结果：`software_output/<属性>_<输入名>/*.csv`
- 最终集成：`<user_save_dir>/<属性>/<属性>_<输入名>_Ensemble_Result.csv`

---

## 5. Trained · 特征选择配置

`Trained/` 下**不是训练好的模型权重**，而是特征选择的配置产物：

| 子目录 | 含义 |
|---|---|
| `*_shap_sfs/*.csv` | SFS 顺序前向选择：第一行数字＝保留特征的列号 |
| `*_shapvalues_rank/*.csv` | SHAP 特征重要性排序：特征名 → 原始特征矩阵列号 |

配套逻辑见 [`pred_software.py`](pred_software.py) 的 `get_original_index()`：它依据 rank 文件的编号规则（全局/局部两种），把 SFS 选出的列号还原成 `encode_aai_descriptor()` 需要的原始特征索引。

---

## 6. 局限与说明 / Limitations

- **每次运行重新训练**：本引擎没有持久化的 `model.pkl`，性能与实验配置（`data.json` 数据集、`Trained/` 特征选择）强绑定。
- **依赖 pySAR**：与下游 `fitness_model` 的零依赖不同，本引擎依赖 `pySAR/` 库与 `data.json` 工作配置。
- **训练集规模（容易混淆，务必区分）**：本引擎随 `data.json` 配置的当前数据集为 `test_activity_1.csv`（SI 约 48 条）；这与下游 fitness 闭环所用集（当前为 `data/raw/Supplementary-Material-2.xlsx`，126 条；另存旧集 `enzyme_activity.xlsx` 204 条）**不同源、不同规模**。本文档中出现的下游 fitness_model 指标为「126 条 / R²≈0.31 / Spearman≈0.64」（360-aa 全长前体版）与「126 条 / R²=0.339 / Spearman=0.654」（2UWF 356-aa 部署版）；本引擎的评估数字取决于 `data.json` 数据集与 `Trained/` 特征选择配置，两者不要混读。
- **蛋白特异**：模型/特征只对训练的某一个木聚糖酶（360-aa）有效，换蛋白需换对应数据集与特征选择配置。
- **与 `../fitness_model/` 的关系**：下游是自包含的轻量基线；本引擎是更重、更可解释、可落地上游特征的版本，二者按 3.2 的方向协作。

---

*最后更新：2026-09-07*
