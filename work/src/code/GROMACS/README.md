# GROMACS 子目录 — MD 全流程

> 本目录是 `run_all.py` 中 Step 0~7 的实际工作目录：存放输入 PDB、MDP 参数、
> MD 步骤脚本，以及逐步骤产物。主入口脚本是上一级的 `../run_all.py`。

---

## 1. PDB 输入入口（关键）

**把你要改造的木聚糖酶 PDB 放这里：**

```
src/code/GROMACS/
└── input_pdb/           ← ★ 你的 PDB 结构文件放到这个目录
    └── <名字>.pdb
```

- run_all 的 **Step 0** 会扫描 `input_pdb/`，把里面的**第一个** `.pdb` 重命名为
  `protein.pdb`，并复制一份到 `input_rename/`（见 `renamepdb.py`）。
- `input_rename/` 由 `renamepdb.py` 自动创建；本目录已预置占位。
- ⚠️ 目录里**至少要有一个 `.pdb`**，且目录本身必须存在（否则 Step 0 会报
  「input_pdb/ 目录不存在」或「没有可用的 .pdb」）。

> 该目录已用 `.gitkeep` 入库：即使克隆/复制仓库，`input_pdb/` 入口也会保留，
> 你只需把自己的 PDB 文件放进去即可，无需新建目录。

## 2. 目录结构

```text
src/code/GROMACS/
├── input_pdb/        # ★ 输入：放你的 PDB（必备）
├── input_rename/     # Step 0 产物：input_pdb 里第一个 pdb → protein.pdb
├── mdp_files/        # GROMACS MDP 模板（em/nvt/npt/md/ions.mdp）
├── scripts/           # MD 步骤脚本 01~07（由 run_all.py 动态 import）
│   ├── 01_pdb2gmx.py
│   ├── 02_em.py
│   ├── 03_nvt.py
│   ├── 04_npt.py
│   ├── 05_md.py
│   ├── 06_cleanup.py
│   ├── 07_analysis.py
│   └── _gmx_utils.py
├── renamepdb.py      # Step 0 调用的改名脚本
├── MD_result/        # 运行产物（轨迹/报告，gitignore）
├── output_statistics/ # 统计输出（gitignore）
└── temporary_files/   # 中间文件（gitignore）
```

## 3. 快速运行

```bash
# 先放一个 PDB 到 input_pdb/，然后从仓库根目录跑（方向别反了）：
cd <仓库根目录>
python src/code/run_all.py --dry-run          # 预览，确认结构/输入 OK
python src/code/run_all.py --gmx gmx          # 只跑 Step 0~7（GROMACS MD）
```

**让下游自动串联**：不传额外参数时，`run_all.py` 会先在仓库内约定的
`src/code/ProteinMPNN/ProteinMPNN-main` 查找完整目录；找到则 MD 完成后**自动**执行
Step 8 位置推荐 → Step 9 ProteinMPNN → Step 10 fitness（Step 8/9/10）。

```bash
# 情形 A：仓库里已经放好了 src/code/ProteinMPNN/ProteinMPNN-main（含权重）→ 只需 --gmx
python src/code/run_all.py --gmx gmx

# 情形 B：ProteinMPNN 在你机器上的其它位置 → 用 --proteinmpnn-dir 指定
python src/code/run_all.py --gmx gmx --proteinmpnn-dir "D:\ProteinMPNN\ProteinMPNN-main"
```

> **为什么默认不一定自动串联**：ProteinMPNN 是独立第三方仓库，且**模型权重
> `vanilla_model_weights/*.pt` 不在本仓库内**（未入库、不在克隆/zip 中）。如果
> `src/code/ProteinMPNN/ProteinMPNN-main` 目录不存在、或缺少 `protein_mpnn_run.py` /
> 模型权重 / `helper_scripts` 下的必要脚本，程序会打印警告并只跑 Step 0~7，**不会报错**，
> 也不会误连一个残缺目录。补齐文件后重跑即可自动串联。

**如何准备仓库内的那份 ProteinMPNN**：把官方 ProteinMPNN 仓库克隆/解压到
`src/code/ProteinMPNN/ProteinMPNN-main`，再从官方发行版下载
`vanilla_model_weights/v_48_020.pt` 放入其 `vanilla_model_weights/` 目录。
（也可完全跳过这一步，改用上面的情形 B，指向任意完整目录。）

主控脚本用 `Path(__file__)` 定位本目录并切换工作目录；因此**从仓库根目录运行**
即可，无需手动 cd 进 GROMACS。更多入参（蛋白质位图/MPNN/fitness）见
[`../README.md`](../README.md) §3.3。
