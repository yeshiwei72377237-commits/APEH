# src/code — 可运行计算代码

> 本目录是 APEH 端到端可运行代码的存放处：GROMACS MD → 候选位置推荐 → ProteinMPNN
> → fitness 评分。主控入口是下面的 `run_all.py`。模块级说明见各自子目录的 README：
> `GROMACS/README.md`、`ProteinMPNN/`、`Model/README.md`、`fitness_model/README.md`。

## 目录结构

```
src/code/
├── run_all.py               # ★ 端到端主控：Step 0~10
├── GROMACS/                 #   分子动力学步骤脚本 01~07（含 input_pdb 入口）
├── ProteinMPNN/             #   位置推荐器 + ProteinMPNN 封装
│   └── ProteinMPNN-main/    #   （约定位置）第三方 ProteinMPNN 仓库，见下方说明
├── Model/                   #   上游特征 / 回归模型引擎（pySAR，满血版）
└── fitness_model/           #   下游 fitness 评分闭环（瘦身版）
```

---

## 1. 主控：`run_all.py`

`run_all.py` 是**唯一的端到端入口**。它用 `Path(__file__)` 定位本目录并切换工作目录，
因此**从仓库根目录运行**即可，无需手动 `cd` 进 GROMACS。

### 步骤清单

| 步骤 | 内容 | 状态 |
|---|---|---|
| Step 0 | PDB 改名（`input_pdb/` 第一个 `.pdb` → `protein.pdb`） | 必需 |
| Step 1~5 | GROMACS MD：准备 → EM → NVT → NPT → 成品模拟 | 必需 |
| Step 6~7 | 清理归档 + 后处理分析（RMSD/Rg/RMSF/SASA/能量分解） | 必需 |
| Step 8 | 位置推荐（逐残基 RMSF + SASA → 前 N 候选） | **可选/自动** |
| Step 9 | ProteinMPNN 生成候选序列 | **可选/自动** |
| Step 10 | fitness 评分 → TopN 推荐 | **可选/自动** |

Steps 0~7 是每个目标蛋白都必需的 MD 流程。Steps 8/9/10 是下游**设计/打分**环节，
默认不传参数时**不会运行**——只有提供了有效 ProteinMPNN 目录（自动或显式）才会串联执行。

---

## 2. 零参数自动串联（推荐）

**核心行为**：`run_all.py` 会先在仓库内约定的
`src/code/ProteinMPNN/ProteinMPNN-main` 查找完整目录。**找到则 MD 成功后自动执行**
Step 8 位置推荐 → Step 9 ProteinMPNN（`--from-recommendations`，直接用推荐位置）
→ Step 10 fitness。找不到或目录不完整则只跑 Step 0~7 并打印警告，**不会报错**。

```bash
# 情形 A：仓库内已放好 src/code/ProteinMPNN/ProteinMPNN-main（含权重）→ 只需 --gmx
python src/code/run_all.py --gmx gmx

# 情形 B：ProteinMPNN 在其它位置 → 用 --proteinmpnn-dir 指定完整目录
python src/code/run_all.py --gmx gmx --proteinmpnn-dir "D:\ProteinMPNN\ProteinMPNN-main"
```

> 默认不一定自动串联的原因：ProteinMPNN 是**独立的第三方仓库**，且**模型权重
> `vanilla_model_weights/*.pt` 不在本仓库内**（未入库、不在克隆/zip 中）。若
> `src/code/ProteinMPNN/ProteinMPNN-main` 缺失、或缺少 `protein_mpnn_run.py` /
> 模型权重 / `helper_scripts` 下的必要脚本，程序会打印警告并跳过 Step 8/9/10。

### 解析优先级

`run_all.py` 按以下顺序确定 ProteinMPNN 目录（见 `_resolve_proteinmpnn_dir`）：

1. **显式 `--proteinmpnn-dir`**（若填了但无效 → 启动即报错）；
2. **仓库内约定位置** `src/code/ProteinMPNN/ProteinMPNN-main`（与 `proteinmpnn_wrapper.py`
   ⽅式的默认路径一致），并校验其完整性；
3. 两者都不成立 → 未启用 ProteinMPNN。

`_validate_proteinmpnn_dir` 会校验：`protein_mpnn_run.py` 主程序、`vanilla_model_weights/
v_48_020.pt` 模型权重、以及 `helper_scripts/` 下的 `parse_multiple_chains.py`、
`assign_fixed_chains.py`、`make_fixed_positions_dict.py`。

### 如何准备仓库内的那份 ProteinMPNN

把官方 ProteinMPNN 仓库克隆/解压到 `src/code/ProteinMPNN/ProteinMPNN-main`，再从官方
发行版下载 `vanilla_model_weights/v_48_020.pt` 放入其 `vanilla_model_weights/` 目录。
（也可完全跳过，改用上面的情形 B，指向任意完整目录。）

---

## 3. 常用入参

| 参数 | 说明 |
|---|---|
| `--gmx` | GROMACS 命令名，默认 `gmx` |
| `--proteinmpnn-dir` | ProteinMPNN 仓库目录；**单独填写即自动串联 推荐→MPNN→fitness**，无需再指定 `--design-positions` |
| `--design-positions` | 可选：覆盖推荐位置，直接指定设计位置，如 `"A35 A68 A102"`；不填则自动用 Step 8 推荐位置 |
| `--recommend-positions` | 只跑位置推荐（无 ProteinMPNN 目录时的独立场景）；自动串联时隐含执行，无需再填 |
| `--existing-stats-dir` | 复用已有 `output_statistics/<批次>` 进行推荐/MPNN，跳过 MD 模拟 |
| `--num-sequences` | ProteinMPNN 候选数量，默认 20 |
| `--sampling-temp` | 采样温度，默认 0.1 |
| `--omit-aas` | 禁止引入的氨基酸，如 `"PC"`；默认不禁止 |
| `--seed` | 随机种子，默认 37 |
| `--proteinmpnn-output` | ProteinMPNN 输出目录；默认写入 `results/gromacs_proteinmpnn/<批次>` |
| `--fitness` | 是否启用 Step 10 fitness 评分闭环 | 

> `--design-positions` 与 `--from-recommendations` 互斥：wrapper 同一时刻只会收到其一。
> 显式指定设计位置时跳过推荐；否则自动走推荐表。

---

## 4. 单运行目录约定

位置推荐与 ProteinMPNN 都遵循**单运行目录**约定：每个目标蛋白对应一个自包含目录，所有
读取与输出都相对该目录解析，官方 ProteinMPNN 文件零改动。`run_all.py` 会自动完成末态结构
提取与运行目录装配，回到 `results/gromacs_proteinmpnn/<批次>/`：

```text
<run_dir>/
├── final_protein_chainA.pdb     # 输入结构（目录内唯一 *.pdb）
├── rmsf.csv / residue_sasa.csv  # GROMACS 逐残基 RMSF / SASA
├── position_scores.csv          # 推荐器：全残基评分
├── recommended_positions.csv    # 推荐器：前 N 个候选位置
├── resid_mapping.json           # 链内序号 → PDB 残基号映射
├── seqs/<name>.fa               # ProteinMPNN 原始 FASTA
├── candidates.csv               # 候选序列 + 突变列表（PDB 残基号）
├── audited_candidates.csv       # 去重 + 风险审核
└── proteinmpnn.log              # 完整运行日志
```

两个封装脚本均可独立使用：`src/code/ProteinMPNN/mutation_position_recommender.py` 与
`src/code/ProteinMPNN/proteinmpnn_wrapper.py`，统一通过 `--dir <run_dir>` 指定运行目录。

---

## 5. 更多

- GROMACS 输入 PDB 入口与 MD 步骤细节见 [`GROMACS/README.md`](GROMACS/README.md)。
- 端到端流程、参数、验证记录见根目录 [`../../README.md`](../../README.md) §3.3。
