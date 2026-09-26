# fitness_model — 2UWF 构建体编号对齐与重训说明

> 本文档说明为什么要做这次「355/356 vs 360」的重训、编号迁移规则、以及如何复现 / 回滚。
> 它是 v1.0（2026-09-08）迭代的**对齐 / 映射核对记录**，也是唯一保留的「改造过程」工作文件。

---

## 1. 一句话背景

`score.py` 只对「候选序列长度 == 训练 WT 长度」打分（蛋白特异性长度校核）。旧模型的训练
WT 是 **360-aa 全长前体**，而实际输入的 RCSB **`2UWF`** 是 **356-aa 成熟结构域构建体**，
两个长度不相等 → 所有候选被判 `score_invalid=1`，`topN` 为空。

本次把 fitness 模型的 WT 换成与 2UWF 构建体**同长同编号**的序列重训，候选长度 = 356 通过校核。

---

## 2. 2UWF 结构到底是多少 aa

| 项 | 值 |
|---|---|
| PDB | **2UWF**（Bacillus halodurans GH10 碱性内切木聚糖酶） |
| UniProt | **Q17TM8** |
| 旧训练 WT | 全长前体 **360 aa**（`GVFGENQKRN…`） |
| 2UWF 构建体 | **356 aa 成熟域表达构建体** = 催化域（UniProt 残基 47–396，350 残基）+ **C 端 6×His** |
| PDB 编号 | 残基号 **11–366**，`SEQRES` 总长 **356** |
| 新训练 WT | **356 aa**（`NQPFAWQVAS…HHHHHH`，含 C 端 His-tag），即 `artifacts/wt.txt` |

> 因此「356 vs 360」的差异来源：RCSB 下载的是**成熟域构建体**（去掉了 N 端信号/前导序列、
> 短了约 10 个残基，并在 C 端加 6×His），而不是 360-aa 全长前体本身。

---

## 3. 编号迁移规则（把 360 标签映射到 356 构建体）

用全局序列比对（Bio.Align PairwiseAligner，match=1 / mismatch=0 / open_gap=-1 / extend=-0.5）
得到两者的对应关系：

- **同源同一蛋白**（全局比对约 **95% 一致**）；差异集中在两端与个别位点。
- **N 端**：旧 WT 开头的 `GVFGENQKRN`（旧坐标 1–9，以及 11）在构建体中被删除；旧坐标 **10**
  保留。因此旧坐标 **12–360** 与构建体坐标 **2–350** 为 **1:1 编号**。
- **编号平移**：构造体坐标 = 旧坐标 − 10；同时 PDB 残基号 = 构造体坐标 + 10。
  - 例：旧标签 `W241Y` → 构造体坐标 231 → PDB 残基号 241；`Q28Y` → 构造体坐标 18 → PDB 18。
- **N 端前导区**（旧坐标 1–11）：无训练标签（已被删）。
- **C 端**：构建体坐标 351–356 为 His-tag（PDB `H361-H366`），无训练标签。
- **一处内部替换**：旧坐标 164 处旧 WT 与构建体有一个残基差异（旧 G vs 构建体 D），比对按
  mismatch 处理，不影响整体 1:1 映射规则。

> 迁移时对每条复突变标签的**每一个 token** 都做上述平移（不是只看第一个 token），
> 因此 **126 条标签全部迁移、0 条丢失**。

---

## 4. 这次改了什么（重训前后对比）

| 项 | 旧 | 新 |
|---|---|---|
| 训练 WT | 全长前体 360 aa | **2UWF 构建体 356 aa**（`wt.txt` 含 C 端 His-tag） |
| 训练集 | `data/raw/Supplementary-Material-2.xlsx`（360 编号） | `data/raw/Supplementary-Material-2_2UWF.xlsx`（构建体编号） |
| 标签坐标 | 360-前体编号 | **构建体编号 = 旧编号 − 10**，126 条复突变全部迁移，0 条丢失 |
| 交叉验证 | R²=0.312 / Spearman=0.642 | **R²=0.339 / Spearman=0.654** |
| 打分校核 | `len(候选)==360` 才通过 | `len(候选)==356` 才通过（与构建体一致） |

---

## 5. 复现命令

`train.py` 的**默认参数仍指向 360 全长前体**（内嵌 `DEFAULT_WT` + `--xlsx
data/raw/Supplementary-Material-2.xlsx`）。要重建本 356 版本，需显式传入：

```bash
python src/code/fitness_model/train.py \
  --xlsx data/raw/Supplementary-Material-2_2UWF.xlsx \
  --sheet Sheet1 --target activity \
  --wt-file src/code/fitness_model/artifacts/wt.txt \
  --out src/code/fitness_model/artifacts
```

> 注意：**不要用默认参数裸跑 `train.py`**，否则会把 artifacts 重置回 360 版本。

---

## 6. 回滚

旧 **360-aa** 模型备份在 **`../artifacts_backup_360wt/`**（wt.txt=360、model.pkl=378894 字节）。
确认 356 版本可用后可删除，或想回到 360 时用备份覆盖 `artifacts/` 即可。

---

## 7. 结论

本次重训让 `GROMACS → ProteinMPNN → fitness` 能对实际输入的 2UWF 构建体**端到端跑通**：
候选长度 = 356 = 训练 WT，`score.py` 校核通过，输出 TopN 推荐。

**实测记录（2026-09-10，GROMACS 2026.2 + RTX 4060 Laptop）**：
整条流水线跑完并留档于 [`results/golden_run/`](../../../results/golden_run/README.md)，
20 条候选经审核得 19 条，**`seq_len` 全部 = 356、`score_invalid` 全部 = 0**，TopN 输出 **16 条**。

> 本文档早前记载的「13 valid / 0 invalid」缺少可核查的运行记录
> （`results/gromacs_proteinmpnn/` 当时为空目录）。以上为可复跑、有完整日志与
> SHA-256 的实测值，以本次为准。
