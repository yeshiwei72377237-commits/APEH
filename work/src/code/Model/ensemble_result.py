import os
import re
import argparse
import numpy as np
import pandas as pd



def read_single_result(file_path, model_name):
    """
    读取单个 pred 脚本输出的 CSV。

    预期列：
    Mutant, Seq, Fitness_pred
    """
    df = pd.read_csv(file_path)

    required_cols = {"Mutant", "Seq", "Fitness_pred"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{file_path} 缺少必要列：{missing}")

    df = df[["Mutant", "Seq", "Fitness_pred"]].copy()
    df = df.rename(columns={"Fitness_pred": model_name})

    return df


def infer_model_name_from_file(file_path):
    """
    根据模型结果文件名推断模型名称。

    示例：
    software_output/CSV_titled/baggingregressor_MANP780101.csv
    -> baggingregressor_MANP780101
    """
    stem = os.path.splitext(os.path.basename(str(file_path)))[0]
    return stem.strip()


def is_model_result_file(file_path):
    """
    判断一个 CSV 是否是单模型预测结果文件。

    合法单模型结果文件必须包含：
    Mutant, Seq, Fitness_pred

    自动排除已经生成的 Ensemble_Result / Top10 文件。
    """
    if not str(file_path).lower().endswith(".csv"):
        return False

    file_name = os.path.basename(str(file_path))
    lower_name = file_name.lower()

    # 排除集成结果和 Top 结果，避免重复作为输入
    excluded_keywords = [
        "ensemble_result",
        "top10_by_mean",
        "top10_by_ucb",
        "top_by_mean",
        "top_by_ucb",
    ]
    if any(k in lower_name for k in excluded_keywords):
        return False

    try:
        df_head = pd.read_csv(file_path, nrows=1)
    except Exception:
        return False

    required_cols = {"Mutant", "Seq", "Fitness_pred"}
    return required_cols.issubset(set(df_head.columns))


def collect_model_result_files(input_dir):
    """
    从一个输入文件夹中自动收集单模型预测结果文件。

    示例输入：
    software_output/CSV_titled

    该目录下若包含：
    - baggingregressor_MANP780101.csv
    - gradientboostingregressor_ROSG850101.csv
    - xgb_JANJ780102.csv

    则自动返回这 3 个文件路径和对应 model_names。
    """
    input_dir = os.path.abspath(str(input_dir))

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"输入目录不存在或不是文件夹：{input_dir}")

    file_paths = []
    for file_name in sorted(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, file_name)
        if os.path.isfile(file_path) and is_model_result_file(file_path):
            file_paths.append(file_path)

    if not file_paths:
        raise ValueError(
            f"在目录中没有找到合法的单模型预测结果 CSV：{input_dir}\n"
            "合法文件需要包含列：Mutant, Seq, Fitness_pred。"
        )

    model_names = [infer_model_name_from_file(p) for p in file_paths]

    return file_paths, model_names


def merge_model_results(file_paths, model_names=None):
    """
    合并多个模型预测结果。

    参数：
    file_paths: 多个单模型预测结果 CSV 路径
    model_names: 每个模型的名称，例如 ["GBR", "Bagging", "XGB"]

    返回：
    merged_df: 合并后的 DataFrame
    pred_cols: 各模型预测值列名
    """
    if model_names is None:
        model_names = [infer_model_name_from_file(p) for p in file_paths]

    if len(file_paths) != len(model_names):
        raise ValueError("file_paths 和 model_names 数量不一致。")

    dfs = []
    for file_path, model_name in zip(file_paths, model_names):
        dfs.append(read_single_result(file_path, model_name))

    merged_df = dfs[0]

    for df in dfs[1:]:
        merged_df = pd.merge(
            merged_df,
            df,
            on=["Mutant", "Seq"],
            how="inner"
        )

    pred_cols = model_names

    if merged_df.empty:
        raise ValueError("多个模型结果合并后为空，请检查 Mutant 和 Seq 是否一致。")

    return merged_df, pred_cols


def bootstrap_mean_ci(values, n_boot=5000, ci=95, random_state=42):
    """
    基于多个模型预测值进行 bootstrap 均值置信区间估计。

    注意：
    这里是“模型间预测值 bootstrap”，不是严格意义上的训练集 bootstrap CI。
    如果只有 3 个模型，该区间只能作为模型分歧参考。
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(random_state)

    boot_means = []
    n = len(values)

    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = (100 - ci) / 2
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)

    return lower, upper


def classify_consensus(std_value, mean_value):
    """
    根据模型间标准差判断模型一致性。
    """
    if mean_value == 0:
        rel_std = np.inf
    else:
        rel_std = std_value / abs(mean_value)

    if rel_std <= 0.05:
        return "High"
    elif rel_std <= 0.15:
        return "Medium"
    else:
        return "Low"


def classify_prediction_level(mean_pred):
    """
    根据预测均值给出预测等级。
    默认预测值含义是相对于 WT 的倍数。
    """
    if mean_pred >= 2.0:
        return "High"
    elif mean_pred >= 1.2:
        return "Medium"
    elif mean_pred >= 0.8:
        return "Neutral"
    else:
        return "Low"


def make_recommendation(mean_pred, std_pred, consensus_level):
    """
    生成推荐结论。
    """
    level = classify_prediction_level(mean_pred)

    if level == "High" and consensus_level == "High":
        return "优先实验验证：预测值高，且多个模型判断较一致"
    elif level == "High" and consensus_level in ["Medium", "Low"]:
        return "高潜力候选：预测值高，但模型间存在一定分歧"
    elif level == "Medium" and consensus_level == "High":
        return "可考虑验证：预测值中等，模型判断较一致"
    elif level == "Medium":
        return "备选候选：预测值中等，建议结合结构信息进一步筛选"
    elif level == "Neutral":
        return "优先级较低：预测值接近 WT"
    else:
        return "不推荐优先验证：预测值较低"


def add_ensemble_metrics(
    df,
    pred_cols,
    k_ucb=0.7,
    n_boot=5000,
    ci=95,
    random_state=42
):
    """
    添加多模型集成指标。
    """
    result_df = df.copy()

    pred_matrix = result_df[pred_cols].astype(float)

    result_df["Mean_pred"] = pred_matrix.mean(axis=1)
    result_df["Std_pred"] = pred_matrix.std(axis=1, ddof=1)
    result_df["Min_pred"] = pred_matrix.min(axis=1)
    result_df["Max_pred"] = pred_matrix.max(axis=1)
    result_df["Range_pred"] = result_df["Max_pred"] - result_df["Min_pred"]

    result_df["UCB_score"] = result_df["Mean_pred"] + k_ucb * result_df["Std_pred"]
    result_df["LCB_score"] = result_df["Mean_pred"] - k_ucb * result_df["Std_pred"]

    ci_lowers = []
    ci_uppers = []

    for _, row in result_df.iterrows():
        lower, upper = bootstrap_mean_ci(
            row[pred_cols].values,
            n_boot=n_boot,
            ci=ci,
            random_state=random_state
        )
        ci_lowers.append(lower)
        ci_uppers.append(upper)

    result_df[f"Bootstrap_CI{ci}_lower"] = ci_lowers
    result_df[f"Bootstrap_CI{ci}_upper"] = ci_uppers

    result_df["Prediction_level"] = result_df["Mean_pred"].apply(classify_prediction_level)

    result_df["Consensus_level"] = result_df.apply(
        lambda row: classify_consensus(row["Std_pred"], row["Mean_pred"]),
        axis=1
    )

    result_df["Recommendation"] = result_df.apply(
        lambda row: make_recommendation(
            row["Mean_pred"],
            row["Std_pred"],
            row["Consensus_level"]
        ),
        axis=1
    )

    result_df["Rank_by_mean"] = (
        result_df["Mean_pred"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    result_df["Rank_by_ucb"] = (
        result_df["UCB_score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    result_df["Tie_group"] = result_df["Mean_pred"].round(6)
    tie_counts = result_df.groupby("Tie_group")["Tie_group"].transform("count")
    result_df["Tie_count"] = tie_counts

    result_df = result_df.sort_values(
        by=["Mean_pred", "Std_pred"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return result_df


def get_output_prefix_from_dir(output_dir):
    """
    根据输出目录名生成结果文件名前缀。

    示例：
    output_dir = software_output/CSV_titled
    prefix = CSV_titled
    """
    prefix = os.path.basename(os.path.abspath(str(output_dir)))
    prefix = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", prefix).strip("_")
    return prefix or "Ensemble"


def save_outputs(result_df, output_dir, output_prefix=None):
    """
    保存完整集成结果。

    - output_dir 控制最终结果保存到哪个文件夹；
    - output_prefix 控制结果文件名前缀。

    典型用法：
    input_dir = software_output/CSV_titled
    output_dir = ensemble_output
    output_prefix = CSV_titled

    输出：
    ensemble_output/CSV_titled_Ensemble_Result.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    if output_prefix is None:
        prefix = get_output_prefix_from_dir(output_dir)
    else:
        prefix = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", str(output_prefix)).strip("_")
        if not prefix:
            prefix = "Ensemble"

    full_path = os.path.join(output_dir, f"{prefix}_Ensemble_Result.csv")
    result_df.to_csv(full_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    print(f"完整集成结果已保存：{full_path}")
    return full_path


def run_ensemble(
    file_paths=None,
    output_dir=None,
    model_names=None,
    input_dir=None,
    output_prefix=None,
    k_ucb=0.7,
    n_boot=5000,
    ci=95,
    random_state=42
):
    """
    主函数：合并多个 pred 输出结果，并计算集成指标。

    两种调用方式：

    方式 1：显式传入 file_paths
    run_ensemble(
        file_paths=["model1.csv", "model2.csv", "model3.csv"],
        model_names=["model1", "model2", "model3"],
        output_dir="software_output/CSV_titled"
    )

    方式 2：只传入 input_dir / output_dir，脚本自动扫描该文件夹中的模型结果
    run_ensemble(
        input_dir="software_output/CSV_titled",
        output_dir="software_output/CSV_titled"
    )

    如果 output_dir 为空，则默认使用 input_dir。
    如果 output_dir 手动指定到其他目录，默认仍使用 input_dir 的文件夹名作为输出文件名前缀。
    例如：
    run_ensemble(
        input_dir="software_output/CSV_titled",
        output_dir="ensemble_output"
    )
    会输出：ensemble_output/CSV_titled_Ensemble_Result.csv
    """
    if input_dir is not None:
        if file_paths is not None:
            raise ValueError("input_dir 和 file_paths 不要同时传入。")
        file_paths, model_names = collect_model_result_files(input_dir)
        if output_prefix is None:
            output_prefix = get_output_prefix_from_dir(input_dir)
        if output_dir is None:
            output_dir = input_dir

    if file_paths is None:
        raise ValueError("必须提供 input_dir 或 file_paths。")

    if output_dir is None:
        raise ValueError("必须提供 output_dir；或者提供 input_dir 并让 output_dir 默认等于 input_dir。")

    merged_df, pred_cols = merge_model_results(file_paths, model_names)

    result_df = add_ensemble_metrics(
        merged_df,
        pred_cols=pred_cols,
        k_ucb=k_ucb,
        n_boot=n_boot,
        ci=ci,
        random_state=random_state
    )

    save_outputs(result_df, output_dir, output_prefix=output_prefix)

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="整合同一预测目标下多个模型的 pred 输出结果，并计算 Mean / Std / UCB / Bootstrap CI。"
    )

    parser.add_argument(
        "--input_dir",
        default=None,
        help="单个输入文件对应的模型结果文件夹。例如 software_output/CSV_titled。提供该参数时会自动扫描该目录下的模型结果 CSV。"
    )

    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="多个 pred 输出的 CSV 文件路径。若已提供 --input_dir，则不需要该参数。"
    )

    parser.add_argument(
        "--model_names",
        nargs="+",
        default=None,
        help="模型名称，数量需与 --files 一致。使用 --input_dir 时可省略，会从文件名自动推断。"
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="集成结果输出文件夹。若省略且提供了 --input_dir，则默认输出到 input_dir。"
    )

    parser.add_argument(
        "--output_prefix",
        default=None,
        help="结果文件名前缀。若省略且提供了 --input_dir，则默认使用 input_dir 的文件夹名。"
    )

    parser.add_argument(
        "--k_ucb",
        type=float,
        default=0.7,
        help="UCB 系数，默认 0.7。UCB = Mean + k * Std"
    )

    parser.add_argument(
        "--n_boot",
        type=int,
        default=5000,
        help="Bootstrap 次数，默认 5000。"
    )

    parser.add_argument(
        "--ci",
        type=float,
        default=95,
        help="Bootstrap CI 百分比，默认 95。"
    )

    args = parser.parse_args()

    run_ensemble(
        file_paths=args.files,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        model_names=args.model_names,
        k_ucb=args.k_ucb,
        n_boot=args.n_boot,
        ci=args.ci
    )
