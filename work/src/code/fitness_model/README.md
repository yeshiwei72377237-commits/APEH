# fitness_model — 训练 + 评分闭环（自包含 · 组成基线 / composition baseline）

> ⚠️ **诚实定位（重要，请先读）**：本模型是一个**位置盲的组成基线**（composition
> baseline）。它的 30 维特征只含**氨基酸组成 + 理化性质的聚合统计（均值/标准差）**，
> **不含任何残基位置、接触、结构或上下游上下文**。因此它**无法区分**只是"序列里用了哪些
> 氨基酸"相同、但**空间排布不同**的序列，也**不能**对具体位点的改动给出可靠归因。它只是
> 「能跑通 GROMACS→MPNN→fitness 闭环」的**入门排序基线**，**不代表**对热稳定性/活性/安全性的
> 准确预测。详细能力边界、交叉验证与已知局限见
> [`MODEL_CARD.md`](MODEL_CARD.md)。
>
> ⚠️ **2026-09-10 实测更新**：本模型的全样本 CV R²≈0.339 **已查明主要由「突变个数」混淆项贡献**
> （仅用突变个数一个整数特征即得 R²=0.560，高于本模型）；在突变个数恒定的 52 条单点突变上，
> 本模型 R²=**−0.527** / Spearman=+0.128，**基本无单点效应预测能力**。
> 近三年基线 ESM-2（Science 2023）在同一子集上为 R²=+0.305，**优于本模型**。
> 复跑：`python baseline_esm2_controls.py`；详见 [`MODEL_CARD.md`](MODEL_CARD.md) 与
> [wiki/AI-Computational-Methods.md §4.3](../../../wiki/AI-Computational-Methods.md)。

在一个**湿实验活性数据集**上训练一个 fitness 回归模型，再用它给 ProteinMPNN 候选
打分，并通过多样化/长度约束选出 **TopN 推荐**。这是「MPNN 提出改法 → 模型评活性 →
选择器保留」的推荐闭环。全部代码零外部依赖（仅 numpy + scikit-learn + pandas +
openpyxl），不依赖 `src/code/Model/` 的 pySAR 链路，加载已固化的 `artifacts/model.pkl` 即可打分。

已接入 `src/code/run_all.py` **Step 10**：ProteinMPNN (Step 9) 成功后自动调用本闭环，
产出 `scored_candidates.csv` → `topN_recommendations.csv`。

---

## 模型原理

本模型是一个**蛋白特异性 fitness 回归模型**：输入一条**完整突变残基序列**，输出一个
**预测活性分数**。分两个阶段。

### 1. 特征编码（`featurize.py`）

序列 → 固定长度向量（默认 **30 维**），完全确定性、零外部依赖、任何长度序列都映射到
同一维度（这是它能接 MPNN 随机长度候选的关键）：

| 子块 | 维度 | 说明 |
|---|---|---|
| 氨基酸组成 | 20 | 20 种标准 AA 在序列中的占比 |
| 理化性质 | 10 | 5 个经典性质量表（疏水/极性/体积/电荷/分子量）各取 (均值 + 标准差) |
| （可选）二肽组成 | 400 | 相邻 AA 对占比；默认**关闭**（样本量小，易过拟合） |

### 2. 训练阶段（`train.py`）

```
xlsx (突变位点 + 活性) → token 应用到 WT → 完整突变序列 → encode → 30维向量 → GBM 回归
```
- 目标值 = 「activity」（float）
- 模型 = `GradientBoostingRegressor(random_state=37, n_estimators=300, max_depth=3, lr=0.05, subsample=0.9)`
- `KFold(10)` 交叉验证报告 R² / MAE / RMSE / Spearman
- 落盘 `artifacts/`：`model.pkl`、`featurizer_meta.json`（重建特征器）、`wt.txt`（训练 WT，供校验）、`target_meta.json`

### 3. 推荐阶段（`score.py` + `select_top_mutants.py`，`run_loop.py` 一键串联）

```
MPNN 候选序列 → encode → model.predict() → Fitness_prediction
            → PDB残基号 → 序列位置翻译 (resid_mapping)
            → 蛋白特异校验 (长度 == 训练WT)
            → select_top_mutants 按 min_pred 过滤 / 长度聚焦 / Jaccard 多样化 / 每长度上限
            → topN_recommendations.csv
```
- **PDB 残基号翻译**：`A:S20W`(PDB) → `S18W`(序列位置)，经 `resid_mapping.json` 反查；
  去链名以匹配 `select_top_mutants` 的逗号分词。
- **蛋白特异校验**：候选序列长度 ≠ 训练 WT（见 `artifacts/wt.txt`，当前为 **356 构建体**；
  旧 360 备份在 `../artifacts_backup_360wt/`）→ 标 `score_invalid=1`、不产出分数。
- **TopN 选择**：`select_top_mutants.py`（从 `Model/` 复制的干净 argparse 版）按
  多样性 + 长度聚焦输出最终推荐。

一句话：**MPNN 给出「序列怎么改」，模型给出「改完活性多高」，选择器给出「保留哪几条」。**

---

## 训练集（模型学到的数据）

> **当前落盘版本（2026-09-08）**是 **2UWF 356-aa 构建体**模型：训练集为 **`data/raw/Supplementary-Material-2_2UWF.xlsx`**、WT 为
> **`artifacts/wt.txt`（356 aa，含 C 端 His-tag）**。原因与编号迁移规则见
> [`2UWF_construct_mapping.md`](2UWF_construct_mapping.md)。

来源：**`data/raw/Supplementary-Material-2.xlsx`**，sheet「Sheet1」，目标列「**activity**」。

| 项 | 值 |
|---|---|
| 标签行数 | **126 条**（`ID` 唯一，无需去重；重训为 2UWF 构建体编号后同样 126/126 全迁移，0 条丢失） |
| 目标列 | `activity`（均值约 1.68，范围 0.13–3.50）；同表另有 `fitness` 列（与 activity 相关性仅 0.535，因含义不同，本次**未采用**） |
| 突变格式 | 组合式 token，如 `A127W`、`A127W-E145L-E27D`，**序列内 1 起始位置** |
| 每条突变数 | 均值约 3.03 个，最多 9 个 |
| 参考 WT | **360-aa 全长前体**（`GVFGENQKRN…`，与 `train.py` 内嵌 `DEFAULT_WT` 一致）；重训为 2UWF 构建体后按「构建体坐标 = PDB 编号 − 10」迁移 |
| 特征维度 | 30 |

> 该数据集的本质是**湿实验测得的突变活性表**，因此模型**只认这一个蛋白**。
> 旧训练集 `data/raw/enzyme_activity.xlsx`（223 → 去重 204 条）**仍在仓库保留**，可随时用 `--xlsx` 切回。
> （"223→204"为清洗后有效样本计数，与 raw 文件实际行数不同；数据口径见 [`data/README.md`](../../../data/README.md) 的数字口径一览。）
> 另注：`train.py` 默认参数仍指向 360 全长前体（`Supplementary-Material-2.xlsx`）；
> 要用 356 版本请显式传 `--xlsx data/raw/Supplementary-Material-2_2UWF.xlsx --wt-file artifacts/wt.txt`（见映射文档 §5）。

## 数据流

```
Supplementary-Material-2.xlsx  (126 条突变活性标签)
   │  train.py  [突变应用到 WT → 序列特征 → 训练 GBM → 落盘 artifacts/]
   ▼
artifacts/{model.pkl, featurizer_meta.json, wt.txt, target_meta.json}
   │
ProteinMPNN 运行目录 (audited_candidates.csv + resid_mapping.json + seqs/<name>.fa)
   │  score.py  [PDB残基号 → 序列位置翻译 → 编码 → predict → scored_candidates.csv]
   ▼
scored_candidates.csv
   │  select_top_mutants.py  [Fitness>min_pred, 长度聚焦4–7(偏好,非硬过滤), Jaccard多样, per-value cap]
   ▼
topN_recommendations.csv
```

## 步骤

```bat
:: 1) 训练（126 条样本, CV R²=0.339 / Spearman=0.654 的部署版）
:: ⚠️ train.py 的默认参数仍指向 360-aa 全长前体（Supplementary-Material-2.xlsx），
::    必须显式指定 _2UWF.xlsx 与 356-aa 的 wt.txt，否则会训练出 360 版并覆盖 artifacts/！
python src\code\fitness_model\train.py ^
  --xlsx data\raw\Supplementary-Material-2_2UWF.xlsx ^
  --sheet Sheet1 --target activity ^
  --wt-file src\code\fitness_model\artifacts\wt.txt
:: 若想切回旧的 204 条酶活训练集（可选）：
::   python src\code\fitness_model\train.py --xlsx data\raw\enzyme_activity.xlsx --sheet 原始数据 --target "70℃ pH 9.0 酶活性"

:: 2) 闭环保 × ProteinMPNN 候选目录（一条命令：打分 → TopN 选择）
python src\code\fitness_model\run_loop.py --dir "<run_dir>" --artifacts src\code\fitness_model\artifacts --chain A --top-k 20 --min-pred 0
:: 可单独跑:  score.py --dir <run_dir> --artifacts <artifacts_dir>
::            select_top_mutants.py --input_dir <run_dir> --output_csv <out.csv>
::            （两个参数都是必填；名称用下划线 --input_dir / --output_csv）
```

### 通过 run_all.py 一键打通（推荐）

`run_all.py` 的 Step 10 默认自动接上本闭环。只需按现有方式跑 MD → ProteinMPNN，
Step 9 成功后即自动打分并输出 TopN；`--no-fitness` 可关闭。

```bat
python src\code\run_all.py --proteinmpnn-dir D:\ProteinMPNN\ProteinMPNN-main ...   :: 仓库内 ProteinMPNN-main 就绪时只需 --gmx，自动串联 8/9/10
:: 报告含 "Step 10: fitness_model — MPNN候选打分 → TopN推荐"
```

选中 `run_dir` 需含：`<name>.pdb`、`seqs/<name>.fa`、`resid_mapping.json`、
`audited_candidates.csv`（或 `candidates.csv`）——即 ProteinMPNN wrapper 的“单运行目录”产物。

## 关键参数
- `train.py --target`：预测目标列（默认 `activity`，见下）。
- `--wt / --wt-file`：训练 WT；默认内嵌 360-aa 序列。
- `score.py`：`--chain A` 设计链；`--wt` 可覆盖校验序列。
- `select_top_mutants.py`：`--top-k`、`--min-pred`、`--focus-lengths`(默认4-7)、
  `--priority-positions`、`--similarity-threshold`(Jaccard)。
- `run_loop.py`：透传以上参数，一条命令串联 score + select。
- `run_all.py --fitness-*`：Step 10 的同名参数（`--fitness-top-k`、`--fitness-min-pred` 等）。

## ⚠️ 蛋白特异约束
模型是**蛋白特异**的：只对「与训练 WT 同源、链长一致的蛋白」有效。`score.py` 对每条候选
做 `len(序列) == len(训练WT)` 校验，不符则 `score_invalid=1` 且不产出分数。若要把整个
GROMACS→MPNN→fitness 闭环端到端跑到**同一个蛋白**，请确保 MPNN 输入的 PDB 与训练活性数据
对应同一木聚糖酶。当前落盘版本对应 **2UWF 356-aa 构建体**（`artifacts/wt.txt`）；
若你输入的是 **360-aa 全长前体**，也要用 360 版模型（备份在 `../artifacts_backup_360wt/`，
或用默认参数重训）。两种 WT 的关系与迁移规则见
[`2UWF_construct_mapping.md`](2UWF_construct_mapping.md)。

## 说明
- 基线以 126 条标签、测量噪声大，`R²≈0.31 / Spearman≈0.64`（360-aa 全长前体版；**2UWF 356-aa 部署版为 `R²=0.339 / Spearman=0.654`**，见 [`2UWF_construct_mapping.md`](2UWF_construct_mapping.md)）属「能跑通闭环」的入门基线，
  不代表高精度 fitness。更多标签后用 pySAR 完整版替换 `featurize.py` 即可，train/score/select
  无需改动（只换特征维度）。
- 新旧训练集均在 `data/raw/`：新集为默认（`Supplementary-Material-2.xlsx`，126 条）；旧集
  （`enzyme_activity.xlsx`，223→204 条）仍在，用 `--xlsx --sheet --target` 可随时切回。
- 目标列名现为英文 `activity`（旧集的中文「70℃ pH 9.0 酶活性」含温度符号）；Windows 控制台
  可能 GBK 乱码，脚本已设置 UTF-8 重配置。
