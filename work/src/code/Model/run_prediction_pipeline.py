"""
run_prediction_pipeline.py

统一预测软件入口（单模型结果与最终集成结果分离输出）：

1. 调用 input_parser_txt.py，将用户上传文件或手动输入序列转换为 software_temp 下的标准预测 txt；
2. 根据 task=1/2/3 选择预测属性：
   - 1：同时预测活性和热稳定性
   - 2：只预测活性
   - 3：只预测热稳定性
3. 自动扫描 Trained 下对应的 SFS 文件和 rank 文件；
4. 对每一个标准输入文件，分别调用 pred_software.py 中的 generate_desc_pred()；
5. 单模型预测结果统一保存在 software_output 下，文件夹命名为“活性_输入名”或“热稳定性_输入名”；
6. 调用 ensemble_result.py 中的 run_ensemble()，把三种算法结果集成；
7. 最终集成结果只保存到用户指定目录下的“活性”或“热稳定性”文件夹中。

输出结构示例（task=2，只预测活性，输入 seq1.txt、seq2.csv）：

software_output/
  活性_seq1/
    baggingregressor_MANP780101.csv
    gradientboostingregressor_ROSG850101.csv
    xgb_JANJ780102.csv
  活性_seq2/
    baggingregressor_MANP780101.csv
    gradientboostingregressor_ROSG850101.csv
    xgb_JANJ780102.csv

用户指定保存目录/
  活性/
    活性_seq1_Ensemble_Result.csv
    活性_seq2_Ensemble_Result.csv

输出结构示例（task=1，同时预测活性和热稳定性）：

software_output/
  活性_seq1/
    三种活性模型单模型结果
  热稳定性_seq1/
    三种热稳定性模型单模型结果

用户指定保存目录/
  活性/
    活性_seq1_Ensemble_Result.csv

  热稳定性/
    热稳定性_seq1_Ensemble_Result.csv

注意：
- pred_software.py 底部的示例运行代码必须放到 if __name__ == "__main__": 中，
  否则本脚本 import pred_software 时会自动执行 pred_software 自己的测试预测。
- input_parser_txt.py 需要包含 convert_user_inputs_to_prediction_files()。
- ensemble_result.py 需要包含 run_ensemble(input_dir=..., output_dir=..., output_prefix=...)。
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from pySAR.pySAR import PySAR

try:
    from input_parser_txt import convert_user_inputs_to_prediction_files
except ImportError as e:
    raise ImportError(
        "未能导入 input_parser_txt.py。请确认文件名为 input_parser_txt.py，"
        "并且其中包含 convert_user_inputs_to_prediction_files()。"
    ) from e

try:
    from pred_software import generate_desc_pred
except ImportError as e:
    raise ImportError(
        "未能导入 pred_software.py。请确认文件名为 pred_software.py，"
        "并且其中包含 generate_desc_pred()。"
    ) from e

try:
    from ensemble_result import run_ensemble
except ImportError as e:
    raise ImportError(
        "未能导入 ensemble_result.py。请确认你已将 ensemble_result_dir_mode_single_output.py "
        "重命名为 ensemble_result.py，并且其中包含 run_ensemble()。"
    ) from e


PROPERTY_CONFIG = {
    "A": {
        "zh_label": "活性",
        "sfs_dir": Path("Trained/A_shap_sfs"),
        "rank_dir": Path("Trained/A_shapvalues_rank"),
    },
    "H": {
        "zh_label": "热稳定性",
        "sfs_dir": Path("Trained/H_shap_sfs"),
        "rank_dir": Path("Trained/H_shapvalues_rank"),
    },
}

TASK_MAP = {
    1: ["A", "H"],  # 同时预测活性和热稳定性
    2: ["A"],       # 只预测活性
    3: ["H"],       # 只预测热稳定性
}

SUPPORTED_MODEL_SUFFIX = ".csv"
DESCRIPTOR_NAMES = [
    "amino_acid_composition",
    "dipeptide_composition",
    "conjoint_triad",
]


def sanitize_name(name: str, fallback: str = "item") -> str:
    """将字符串转换为安全的文件夹/文件名，保留中文、字母、数字、下划线和短横线。"""
    name = str(name).strip()
    name = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", name)
    name = name.strip("_")
    return name or fallback


def clean_directory(directory: Union[str, Path]) -> None:
    """清空目录中的所有文件和子目录；如果目录不存在则创建。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for item in directory.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def cleanup_runtime_artifacts(
    artifact_root: Union[str, Path] = ".",
    directories: Optional[Sequence[str]] = None,
    files: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    """
    删除 pred_software / PySAR 在运行过程中生成的过程文件夹和临时文件。

    默认删除：
    - desc2/
    - outputs/
    - desc2_save/
    - data/
    - X_column_names.txt

    返回删除记录，方便日志或调试。
    """
    artifact_root = Path(artifact_root)

    if directories is None:
        directories = ["desc2", "outputs", "desc2_save", "data"]

    if files is None:
        files = ["X_column_names.txt"]

    removed = {"directories": [], "files": []}

    for dirname in directories:
        target_dir = artifact_root / dirname
        if target_dir.exists():
            if target_dir.is_dir():
                shutil.rmtree(target_dir)
                removed["directories"].append(str(target_dir))
            else:
                target_dir.unlink()
                removed["files"].append(str(target_dir))

    for filename in files:
        target_file = artifact_root / filename
        if target_file.exists():
            if target_file.is_dir():
                shutil.rmtree(target_file)
                removed["directories"].append(str(target_file))
            else:
                target_file.unlink()
                removed["files"].append(str(target_file))

    return removed


def clear_directory_contents(directory: Union[str, Path]) -> List[str]:
    """
    清空指定目录中的所有文件和子目录，但保留目录本身。

    用途：流程结束后清空 software_temp 和 software_output，
    避免过程文件残留，同时不删除目录本身，方便前端或后续流程继续使用固定路径。

    返回被删除项目路径列表，方便日志输出。
    """
    directory = Path(directory)
    removed_items: List[str] = []

    if not directory.exists():
        return removed_items

    for item in directory.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
            removed_items.append(str(item))
        elif item.is_dir():
            shutil.rmtree(item)
            removed_items.append(str(item))

    return removed_items


def normalize_model_key(file_path: Union[str, Path]) -> str:
    """
    将 sfs 文件名和 rank 文件名归一化为同一个 key，方便配对。

    示例：
    baggingregressor_MANP780101dipeptide_compositionconjoint_triad_sfs.csv
      -> baggingregressor_MANP780101dipeptide_compositionconjoint_triad

    rank_baggingregressor_MANP780101dipeptide_compositionconjoint_triad.csv
      -> baggingregressor_MANP780101dipeptide_compositionconjoint_triad
    """
    stem = Path(file_path).stem

    if stem.startswith("rank_"):
        stem = stem[len("rank_"):]

    if stem.endswith("_sfs"):
        stem = stem[:-len("_sfs")]

    return stem


def model_result_name_from_key(model_key: str) -> str:
    """
    从模型 key 中提取“算法_氨基酸索引”，用作单模型结果 CSV 文件名主体。

    示例：
    baggingregressor_MANP780101dipeptide_compositionconjoint_triad
      -> baggingregressor_MANP780101

    gradientboostingregressor_ROSG850101amino_acid_compositiondipeptide_composition
      -> gradientboostingregressor_ROSG850101
    """
    if "_" not in model_key:
        return sanitize_name(model_key, fallback="model")

    algorithm, rest = model_key.split("_", 1)

    descriptor_positions = [
        rest.find(desc)
        for desc in DESCRIPTOR_NAMES
        if rest.find(desc) != -1
    ]

    if descriptor_positions:
        first_desc_pos = min(descriptor_positions)
        aa_index = rest[:first_desc_pos]
    else:
        aa_index = rest

    result_name = f"{algorithm}_{aa_index}" if aa_index else algorithm
    return sanitize_name(result_name, fallback="model")


def discover_model_pairs(sfs_dir: Union[str, Path], rank_dir: Union[str, Path]) -> List[Tuple[str, Path, Path]]:
    """
    扫描一个属性目录下的模型文件，并自动配对 sfs 与 rank 文件。

    返回：
    [
      (model_key, sfs_path, rank_path),
      ...
    ]
    """
    sfs_dir = Path(sfs_dir)
    rank_dir = Path(rank_dir)

    if not sfs_dir.exists():
        raise FileNotFoundError(f"SFS 目录不存在：{sfs_dir}")
    if not rank_dir.exists():
        raise FileNotFoundError(f"rank 目录不存在：{rank_dir}")

    sfs_files = sorted(p for p in sfs_dir.iterdir() if p.is_file() and p.suffix.lower() == SUPPORTED_MODEL_SUFFIX)
    rank_files = sorted(p for p in rank_dir.iterdir() if p.is_file() and p.suffix.lower() == SUPPORTED_MODEL_SUFFIX)

    rank_by_key: Dict[str, Path] = {
        normalize_model_key(p): p
        for p in rank_files
    }

    model_pairs: List[Tuple[str, Path, Path]] = []
    unmatched_sfs: List[Path] = []

    for sfs_path in sfs_files:
        key = normalize_model_key(sfs_path)
        rank_path = rank_by_key.get(key)
        if rank_path is None:
            unmatched_sfs.append(sfs_path)
            continue
        model_pairs.append((key, sfs_path, rank_path))

    if unmatched_sfs:
        msg = "以下 SFS 文件未找到对应 rank 文件：\n" + "\n".join(f"- {p}" for p in unmatched_sfs)
        raise FileNotFoundError(msg)

    if not model_pairs:
        raise FileNotFoundError(
            f"未在目录中找到可配对的模型文件：\nSFS: {sfs_dir}\nrank: {rank_dir}"
        )

    return model_pairs


def prepare_prediction_inputs(
    inputs: Union[str, Path, Sequence[Union[str, Path]]],
    wt_seq: str,
    temp_dir: Union[str, Path] = "software_temp",
    clean_temp: bool = True,
    table_has_header: Optional[bool] = None,
    allow_unknown: bool = False,
    allow_indel: bool = False,
    fitness_placeholder: float = 1.0,
    use_mutation_name_as_name: bool = True,
) -> List[Dict[str, object]]:
    """调用 input_parser_txt.py，将原始用户输入转换成标准预测文件。"""
    temp_dir = Path(temp_dir)

    if clean_temp:
        clean_directory(temp_dir)
    else:
        temp_dir.mkdir(parents=True, exist_ok=True)

    return convert_user_inputs_to_prediction_files(
        inputs=inputs,
        wt_seq=wt_seq,
        output_dir=temp_dir,
        table_has_header=table_has_header,
        allow_unknown=allow_unknown,
        allow_indel=allow_indel,
        fitness_placeholder=fitness_placeholder,
        use_mutation_name_as_name=use_mutation_name_as_name,
        preserve_file_stem=True,
        avoid_overwrite=False,
    )


def prepare_single_input_workspace(input_file: Union[str, Path], workspace_root: Union[str, Path]) -> Path:
    """
    为单个标准预测输入文件创建独立工作目录。

    generate_desc_pred() 的接口是处理一个文件夹，所以这里为每个输入文件建立临时目录，
    目录中只放当前这一个 txt 文件，避免它一次性处理 software_temp 中的全部输入。
    """
    input_file = Path(input_file)
    workspace_root = Path(workspace_root)
    single_dir = workspace_root / sanitize_name(input_file.stem, fallback="input")

    if single_dir.exists():
        shutil.rmtree(single_dir)
    single_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(input_file, single_dir / input_file.name)
    return single_dir


def list_csv_files(directory: Union[str, Path]) -> List[Path]:
    """列出目录中的 CSV 文件。"""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".csv")


def make_destination_path(
    output_dir: Union[str, Path],
    result_stem: str,
    overwrite_existing: bool = True,
) -> Path:
    """
    生成单模型结果路径。

    默认 overwrite_existing=True：
      gradientboostingregressor_ROSG850101.csv 已存在时直接覆盖，
      保证每个输入文件夹内固定三个模型结果文件。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_stem = sanitize_name(result_stem, fallback="model")
    target = output_dir / f"{result_stem}.csv"

    if overwrite_existing or not target.exists():
        return target

    counter = 1
    while True:
        candidate = output_dir / f"{result_stem}_{counter}.csv"
        if not candidate.exists():
            return candidate
        counter += 1


def run_one_model_for_one_input(
    input_file: Union[str, Path],
    final_output_dir: Union[str, Path],
    model_result_stem: str,
    sfs_path: Union[str, Path],
    rank_path: Union[str, Path],
    data_json: Union[str, Path],
    workspace_root: Union[str, Path],
    overwrite_existing: bool = True,
) -> Path:
    """
    对单个输入文件调用单个模型，并将 pred_software 生成的结果重命名为“算法_氨基酸索引.csv”。
    """
    input_file = Path(input_file)
    final_output_dir = Path(final_output_dir)
    workspace_root = Path(workspace_root)

    single_input_dir = prepare_single_input_workspace(input_file, workspace_root / "single_inputs")

    raw_output_dir = (
        workspace_root
        / "raw_outputs"
        / sanitize_name(input_file.stem, fallback="input")
        / sanitize_name(model_result_stem, fallback="model")
    )
    if raw_output_dir.exists():
        shutil.rmtree(raw_output_dir)
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    before_csvs = set(list_csv_files(raw_output_dir))

    pys = PySAR(str(data_json))
    generate_desc_pred(
        folder_path=str(single_input_dir),
        output_path=str(raw_output_dir),
        sfs_path=str(sfs_path),
        sorted_path=str(rank_path),
        pys=pys,
    )

    after_csvs = set(list_csv_files(raw_output_dir))
    new_csvs = sorted(after_csvs - before_csvs)

    # raw_output_dir 是每次新建的，正常情况下 after_csvs 就是本次结果。
    if not new_csvs:
        new_csvs = sorted(after_csvs)

    if len(new_csvs) != 1:
        raise RuntimeError(
            f"模型预测完成后，未能唯一确定结果文件。\n"
            f"输入文件：{input_file}\n"
            f"模型：{model_result_stem}\n"
            f"原始输出目录：{raw_output_dir}\n"
            f"检测到 CSV 文件数量：{len(new_csvs)}\n"
            f"CSV 文件列表：{[str(p) for p in new_csvs]}"
        )

    src = new_csvs[0]
    dst = make_destination_path(final_output_dir, model_result_stem, overwrite_existing=overwrite_existing)

    if dst.exists() and overwrite_existing:
        dst.unlink()
    shutil.move(str(src), str(dst))

    return dst


def run_prediction_pipeline(
    inputs: Union[str, Path, Sequence[Union[str, Path]]],
    wt_seq: str,
    task: int,
    user_save_dir: Union[str, Path],
    temp_dir: Union[str, Path] = "software_temp",
    software_output_dir: Union[str, Path] = "software_output",
    clean_temp: bool = True,
    clean_save_dir: bool = False,
    clean_software_output: bool = False,
    cleanup_workspace: bool = True,
    cleanup_process_artifacts: bool = True,
    cleanup_temp_and_output: bool = True,
    process_artifact_root: Union[str, Path] = ".",
    table_has_header: Optional[bool] = None,
    data_json: Union[str, Path] = "data.json",
    overwrite_existing: bool = True,
) -> Dict[str, object]:
    """
    总入口函数：单模型预测结果与最终集成结果分离保存。

    输出约定：
    - input_parser_txt 生成的标准预测 txt 和检查报告保存在 temp_dir，默认 software_temp；
    - pred_software 生成的三种单模型预测结果保存在 software_output_dir，默认 software_output；
      例如：software_output/活性_seq1/*.csv；
    - ensemble_result 生成的最终集成结果保存在 user_save_dir 下的“活性”或“热稳定性”文件夹；
      例如：用户保存目录/活性/活性_seq1_Ensemble_Result.csv。

    参数：
    - inputs:
        文件路径、文件路径列表、手动序列字符串、手动序列列表或 FASTA 文本。
    - wt_seq:
        野生型序列。
    - task:
        1 = 同时预测活性和热稳定性；
        2 = 只预测活性；
        3 = 只预测热稳定性。
    - user_save_dir:
        用户手动选择的最终结果保存目录。脚本会在该目录下自动创建“活性”和/或“热稳定性”文件夹。
    - temp_dir:
        input_parser 生成标准 txt 的临时目录，默认 software_temp。
    - software_output_dir:
        pred_software 产生的单模型结果保存目录，默认 software_output。
    - clean_temp:
        是否在每次运行前清空 temp_dir，建议 True。
    - clean_save_dir:
        是否在每次运行前清空用户保存目录，默认 False，避免误删历史最终结果。
    - clean_software_output:
        是否在每次运行前清空 software_output_dir，默认 False，避免误删历史单模型结果。
    - cleanup_workspace:
        是否在流程结束后删除 temp_dir/_pipeline_work 临时工作目录，默认 True。
    - cleanup_process_artifacts:
        是否在最终结果生成后删除 desc2、outputs、desc2_save、data 过程文件夹以及 X_column_names.txt，
        默认 True。
    - cleanup_temp_and_output:
        是否在最终结果生成后清空 temp_dir 和 software_output_dir 中的所有内容，
        但保留 software_temp / software_output 目录本身，默认 True。
        如需保留标准输入文件、检查报告和单模型预测结果用于调试，可手动设为 False。
    - process_artifact_root:
        过程文件所在根目录，默认当前工作目录 "."。
    - data_json:
        PySAR 训练配置文件，默认 data.json。
    - overwrite_existing:
        True：同名模型结果文件已存在时覆盖，保证每个单模型结果文件夹内固定三个模型结果文件；
        False：同名模型结果文件已存在时追加 _1、_2。
    """
    if task not in TASK_MAP:
        raise ValueError("task 只能是 1、2 或 3：1=两种属性，2=活性，3=热稳定性。")

    temp_dir = Path(temp_dir)
    user_save_dir = Path(user_save_dir)
    software_output_dir = Path(software_output_dir)

    if clean_save_dir:
        clean_directory(user_save_dir)
    else:
        user_save_dir.mkdir(parents=True, exist_ok=True)

    if clean_software_output:
        clean_directory(software_output_dir)
    else:
        software_output_dir.mkdir(parents=True, exist_ok=True)

    print("========== Step 1: 标准化用户输入 ==========")
    input_results = prepare_prediction_inputs(
        inputs=inputs,
        wt_seq=wt_seq,
        temp_dir=temp_dir,
        clean_temp=clean_temp,
        table_has_header=table_has_header,
    )

    print("已生成标准预测输入文件：")
    for item in input_results:
        print(f"  - {item['output_path']}")
        print(f"    检查报告：{item['report_path']}")

    workspace_root = temp_dir / "_pipeline_work"
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    selected_properties = TASK_MAP[task]
    all_prediction_jobs: List[Dict[str, object]] = []
    all_ensemble_jobs: List[Dict[str, object]] = []

    print("========== Step 2: 按属性和输入文件调用三种模型 ==========")

    try:
        for prop in selected_properties:
            config = PROPERTY_CONFIG[prop]
            property_label = config["zh_label"]

            # 这里只保存 ensemble_result 的最终结果，不保存单模型结果文件夹。
            property_ensemble_dir = user_save_dir / property_label
            property_ensemble_dir.mkdir(parents=True, exist_ok=True)

            model_pairs = discover_model_pairs(config["sfs_dir"], config["rank_dir"])

            print(f"\n================ 当前属性：{property_label} / {prop} ================")
            print(f"最终集成结果目录：{property_ensemble_dir}")
            print(f"单模型结果根目录：{software_output_dir}")
            print(f"发现 {len(model_pairs)} 组模型。")

            for input_item in input_results:
                input_file = Path(input_item["output_path"])
                input_stem = sanitize_name(input_file.stem, fallback=f"input_{input_item.get('input_index', '')}")

                # 命名规则：A -> 活性_seq1；H -> 热稳定性_seq1。
                prediction_folder_name = f"{property_label}_{input_stem}"

                # 单模型预测结果保存在 software_output_dir，而不是用户最终保存目录。
                single_prediction_dir = software_output_dir / prediction_folder_name
                single_prediction_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n---------- 输入文件：{input_file.name} ----------")
                print(f"单模型结果文件夹：{single_prediction_dir}")

                model_result_paths: List[Path] = []
                model_names: List[str] = []

                for model_key, sfs_path, rank_path in model_pairs:
                    model_result_stem = model_result_name_from_key(model_key)

                    print("当前模型：")
                    print(f"  model_key: {model_key}")
                    print(f"  result:    {model_result_stem}.csv")
                    print(f"  SFS:       {sfs_path}")
                    print(f"  rank:      {rank_path}")

                    final_result_path = run_one_model_for_one_input(
                        input_file=input_file,
                        final_output_dir=single_prediction_dir,
                        model_result_stem=model_result_stem,
                        sfs_path=sfs_path,
                        rank_path=rank_path,
                        data_json=data_json,
                        workspace_root=workspace_root,
                        overwrite_existing=overwrite_existing,
                    )

                    print(f"  已生成单模型结果：{final_result_path}")

                    model_result_paths.append(final_result_path)
                    model_names.append(model_result_stem)

                    all_prediction_jobs.append({
                        "input_file": str(input_file),
                        "property": prop,
                        "property_label": property_label,
                        "prediction_folder": str(single_prediction_dir),
                        "model_key": model_key,
                        "model_result_name": model_result_stem,
                        "sfs_path": str(sfs_path),
                        "rank_path": str(rank_path),
                        "result_path": str(final_result_path),
                    })

                print("开始生成集成结果：")
                ensemble_df = run_ensemble(
                    input_dir=str(single_prediction_dir),
                    output_dir=str(property_ensemble_dir),
                    output_prefix=prediction_folder_name,
                )

                ensemble_result_path = property_ensemble_dir / f"{prediction_folder_name}_Ensemble_Result.csv"
                print(f"  已生成集成结果：{ensemble_result_path}")

                all_ensemble_jobs.append({
                    "input_file": str(input_file),
                    "property": prop,
                    "property_label": property_label,
                    "prediction_folder": str(single_prediction_dir),
                    "ensemble_output_dir": str(property_ensemble_dir),
                    "ensemble_result_path": str(ensemble_result_path),
                    "n_rows": int(len(ensemble_df)),
                })

    finally:
        if cleanup_workspace and workspace_root.exists():
            shutil.rmtree(workspace_root)
            print(f"临时工作目录已删除：{workspace_root}")

    if cleanup_process_artifacts:
        removed_artifacts = cleanup_runtime_artifacts(process_artifact_root)
        removed_items = removed_artifacts["directories"] + removed_artifacts["files"]
        if removed_items:
            print("以下过程文件/文件夹已删除：")
            for item in removed_items:
                print(f"  - {item}")
        else:
            print("未发现需要删除的过程文件/文件夹。")

    cleaned_temp_output_items: Dict[str, List[str]] = {"temp_dir": [], "software_output_dir": []}
    if cleanup_temp_and_output:
        cleaned_temp_output_items["temp_dir"] = clear_directory_contents(temp_dir)
        cleaned_temp_output_items["software_output_dir"] = clear_directory_contents(software_output_dir)
        print("以下目录中的过程内容已清空，目录本身保留：")
        print(f"  - {temp_dir}")
        print(f"  - {software_output_dir}")

    print("\n========== 全部预测与集成完成 ==========")
    print(f"标准输入目录：{temp_dir}")
    print(f"单模型结果目录：{software_output_dir}")
    print(f"最终集成结果目录：{user_save_dir}")

    return {
        "input_results": input_results,
        "prediction_jobs": all_prediction_jobs,
        "ensemble_jobs": all_ensemble_jobs,
        "temp_dir": str(temp_dir),
        "software_output_dir": str(software_output_dir),
        "user_save_dir": str(user_save_dir),
        "cleanup_process_artifacts": cleanup_process_artifacts,
        "cleanup_temp_and_output": cleanup_temp_and_output,
        "cleaned_temp_output_items": cleaned_temp_output_items,
        "process_artifact_root": str(process_artifact_root),
    }


if __name__ == "__main__":
    # 示例野生型序列。正式使用时请替换为你的 WT_SEQUENCE。
    WT_SEQUENCE = "GVFGENQKRNDQPFAWQVASLSERYQEQFDIGAAVEPYQLEGRQAQILKHHYNSLVAENAMKPVSLQPREGEWNWEGADKIVEFARKHNMELRFHTLVWHSQVPEWFFIDENGNRMVDETDPEKRKANKQLLLERMENHIKTVVERYKDDVTSWDVVNEVIDDGGGLRESEWYQITGTDYIKVAFETARKYGGEEAKLYINDYNTEVPSKRDDLYNLVKDLLEQGVPIDGVGHQSHIQIGWPSIEDTRASFEKFTSLGLDNQVTELDMSLYGWPPTGAYTSYDDIPEELFQAQADRYDQLFELYEELSATISSVTFWGIADNHTWLDDRAREYNNGVGVDAPFVFDHNYRVKPAYWRIID"

    run_prediction_pipeline(
        inputs=[
            #"Test_Sample/TXT_titled.txt",
            #"Test_Sample/rcsb_pdb_2UWF.fasta",
            "Test_Sample/S7_mut.csv",
        ],
        wt_seq=WT_SEQUENCE,
        task=1,
        user_save_dir="software_result",
        temp_dir="software_temp",
        software_output_dir="software_output",
        clean_temp=True,
        clean_save_dir=False,
        clean_software_output=False, 
        cleanup_workspace=True,
        cleanup_process_artifacts=True,
        cleanup_temp_and_output=True, #清空software_temp和software_output两个文件夹里的所有文件
        process_artifact_root=".",
        overwrite_existing=True,
    )
