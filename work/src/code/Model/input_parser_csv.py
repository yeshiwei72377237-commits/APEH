"""
input_parser.py

用于将用户输入的不同格式的木聚糖酶突变体序列统一转换为 PySAR/pred_v2 可读取的预测集格式。

支持输入类型：
1. 手动输入的单条序列
2. 手动输入的 FASTA 文本
3. txt 文件（支持纯序列、FASTA文本、逗号/制表符分隔表格）
4. fasta/fa/faa 文件
5. csv 文件（支持有表头/无表头自动识别）
6. xls/xlsx 文件（支持有表头/无表头自动识别）

输出：
1. 标准预测文件：Name, Mut_seq, fitness
2. 输入检查报告：Name, Raw_length, Clean_length, Valid, Mutations, Errors, Warnings

说明：
- 默认只支持标准 20 种氨基酸：ACDEFGHIKLMNPQRSTVWY
- 默认不支持插入/缺失突变，即输入序列长度必须与野生型一致
- fitness 列为占位值，默认填 0，用于兼容现有 pred_v2.py / PySAR 读取逻辑
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import pandas as pd


# 标准 20 种氨基酸
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# 常见非标准或未知氨基酸字符
# B: D/N, Z: E/Q, J: I/L, X: unknown, U: selenocysteine, O: pyrrolysine
UNKNOWN_AA = set("BJOUXZ")


def read_text_file(file_path: Union[str, Path]) -> str:
    """
    读取 txt/fasta 等文本文件。
    优先使用 utf-8，失败后尝试 gbk。
    """
    file_path = Path(file_path)

    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="gbk")



def read_excel_file(file_path: Union[str, Path], header=0) -> pd.DataFrame:
    """
    读取 Excel 文件。

    说明：
    pandas 读取 .xlsx 文件通常需要 openpyxl。
    如果当前 Python 环境没有安装 openpyxl，会给出更清晰的报错提示。
    """
    try:
        return pd.read_excel(file_path, header=header)
    except ImportError as e:
        if "openpyxl" in str(e).lower():
            raise ImportError(
                "读取 .xlsx 文件需要安装 openpyxl。\n"
                "请在当前 Python/conda 环境中安装：\n"
                "  conda install openpyxl\n"
                "或：\n"
                "  pip install openpyxl\n"
                "如果你是在 VS Code / PyCharm 中运行，请确认安装 openpyxl 的环境与当前运行脚本的解释器是同一个。"
            ) from e
        raise


def clean_sequence(seq: str, remove_numbers: bool = True) -> str:
    """
    清洗氨基酸序列。

    功能：
    1. 转为大写
    2. 去除空格、换行、制表符等空白字符
    3. 可选：去除数字编号
    4. 去除常见 FASTA 中可能残留的非序列分隔符

    注意：
    - 只对序列本身使用，不要对 Name/ID 使用。
    """
    if seq is None:
        return ""

    seq = str(seq).strip().upper()

    # 去除所有空白字符：空格、换行、制表符等
    seq = re.sub(r"\s+", "", seq)

    # 去除序列编号，例如：
    # 1 MKTLLI 10
    # 11 LAVAAA 20
    if remove_numbers:
        seq = re.sub(r"\d+", "", seq)

    # 去除一些常见分隔符，避免用户复制序列时带入
    seq = seq.replace("-", "")
    seq = seq.replace("_", "")

    return seq


def parse_fasta_text(text: str) -> pd.DataFrame:
    """
    解析 FASTA 文本，返回包含 Name 和 Mut_seq 的 DataFrame。
    """
    records: List[Dict[str, str]] = []
    current_name: Optional[str] = None
    current_seq_lines: List[str] = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):
            if current_name is not None:
                records.append({
                    "Name": current_name,
                    "Mut_seq": clean_sequence("".join(current_seq_lines))
                })

            current_name = line[1:].strip()
            if not current_name:
                current_name = f"Seq_{len(records) + 1}"

            current_seq_lines = []
        else:
            current_seq_lines.append(line)

    if current_name is not None:
        records.append({
            "Name": current_name,
            "Mut_seq": clean_sequence("".join(current_seq_lines))
        })

    return pd.DataFrame(records)


def parse_plain_text_as_sequences(text: str) -> pd.DataFrame:
    """
    解析非 FASTA 的纯文本序列。

    支持两种情况：
    1. 单条序列：
       MKTLLILAVAAAG...

    2. 多条序列，每条序列单独一行：
       MKTLLILAVAAAG...
       MKTLLVLAVAAAG...

    如果文本中包含 FASTA 标记 ">"，请使用 parse_fasta_text。
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return pd.DataFrame(columns=["Name", "Mut_seq"])

    # 如果只有一行，按单条序列处理
    if len(lines) == 1:
        return pd.DataFrame([{
            "Name": "Manual_001",
            "Mut_seq": clean_sequence(lines[0])
        }])

    # 多行时：
    # 如果每行看起来都是独立序列，则按多条序列处理
    # 如果是复制的分段序列，例如每行 60 aa，这里会合并为一条序列
    # 判断策略：如果每行清洗后长度相近且没有分隔符，默认合并为一条；
    # 如果用户需要多条，建议使用 FASTA 或 CSV。
    joined_seq = clean_sequence("".join(lines))

    return pd.DataFrame([{
        "Name": "Manual_001",
        "Mut_seq": joined_seq
    }])


def looks_like_sequence(value: object, min_length: int = 10) -> bool:
    """
    判断一个单元格内容是否像氨基酸序列。

    用途：
    - 自动判断 CSV/Excel 是否无表头
    - 在无表头多列表格中自动寻找序列列
    """
    if value is None:
        return False

    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return False

    seq = clean_sequence(raw)
    if len(seq) < min_length:
        return False

    allowed_chars = VALID_AA | UNKNOWN_AA
    return set(seq) <= allowed_chars


def detect_table_header(file_path: Union[str, Path], ext: str) -> bool:
    """
    自动判断 CSV/Excel 是否包含表头。

    返回：
    - True：按有表头读取
    - False：按无表头读取

    判断规则：
    1. 第一行包含 Name / Sequence / Mut_seq 等典型列名 -> 有表头
    2. 第一行本身像氨基酸序列 -> 无表头
    3. 第一行不像序列，但第二行像序列 -> 有表头
    4. 其他情况默认无表头，避免把第一条真实序列误读成表头
    """
    file_path = Path(file_path)

    known_header_keywords = {
        "name", "id", "mutant", "sample", "sample_id",
        "mut_seq", "sequence", "seq", "protein_sequence", "aa_sequence",
        "fitness", "activity"
    }

    if ext == ".csv":
        preview = pd.read_csv(file_path, header=None, nrows=3)
    else:
        preview = read_excel_file(file_path, header=None).head(3)

    if preview.empty:
        return True

    first_row_values = [
        str(x).strip().lower()
        for x in preview.iloc[0].tolist()
        if str(x).strip() and str(x).strip().lower() != "nan"
    ]

    # 第一行出现典型表头关键词
    if any(x in known_header_keywords for x in first_row_values):
        return True

    # 第一行已经像序列，通常说明没有表头
    first_has_sequence = any(looks_like_sequence(x) for x in preview.iloc[0].tolist())
    if first_has_sequence:
        return False

    # 第一行不像序列，第二行像序列，通常说明第一行是表头
    if len(preview) >= 2:
        second_has_sequence = any(looks_like_sequence(x) for x in preview.iloc[1].tolist())
        if second_has_sequence:
            return True

    # 兜底：按无表头处理，避免丢失第一条序列
    return False


def normalize_headerless_table_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理无表头 CSV/Excel。

    支持：
    1. 单列：该列为序列，自动生成 Name
    2. 两列：自动判断哪一列更像序列，另一列作为 Name
    3. 多列：自动寻找最像序列的列；第一列若不是序列列，则作为 Name
    """
    if df.empty:
        return pd.DataFrame(columns=["Name", "Mut_seq"])

    df = df.copy()
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    if df.empty:
        return pd.DataFrame(columns=["Name", "Mut_seq"])

    n_cols = df.shape[1]

    if n_cols == 1:
        return pd.DataFrame({
            "Name": [f"Seq_{i + 1}" for i in range(len(df))],
            "Mut_seq": df.iloc[:, 0].apply(clean_sequence)
        })

    # 给每一列打分：越多行像氨基酸序列，分数越高
    sequence_scores = []
    for col_idx in range(n_cols):
        values = df.iloc[:, col_idx].dropna().astype(str).tolist()
        score = sum(looks_like_sequence(v, min_length=10) for v in values)
        sequence_scores.append(score)

    seq_col_idx = int(max(range(n_cols), key=lambda i: sequence_scores[i]))

    # 如果完全判断不出来，默认第二列为序列
    if sequence_scores[seq_col_idx] == 0:
        seq_col_idx = 1 if n_cols >= 2 else 0

    # Name 列：如果第一列不是序列列，用第一列作为 Name；否则自动生成 Name
    if seq_col_idx != 0:
        raw_names = df.iloc[:, 0].fillna("").astype(str).tolist()
        names = [
            x.strip() if x.strip() and x.strip().lower() != "nan" else f"Seq_{i + 1}"
            for i, x in enumerate(raw_names)
        ]
    else:
        names = [f"Seq_{i + 1}" for i in range(len(df))]

    return pd.DataFrame({
        "Name": names,
        "Mut_seq": df.iloc[:, seq_col_idx].apply(clean_sequence)
    })


def normalize_table_input(df: pd.DataFrame, has_header: bool = True) -> pd.DataFrame:
    """
    将 CSV/Excel 表格输入统一转换为 Name 和 Mut_seq 两列。

    has_header=True：按表头列名识别 Name 和序列列。
    has_header=False：按无表头表格处理。
    """
    if not has_header:
        return normalize_headerless_table_input(df)

    if df.empty:
        return pd.DataFrame(columns=["Name", "Mut_seq"])

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    columns = list(df.columns)

    name_col_candidates = [
        "Name", "name",
        "ID", "Id", "id",
        "Mutant", "mutant",
        "Sample", "sample",
        "Sample_ID", "sample_id"
    ]

    seq_col_candidates = [
        "Mut_seq", "mut_seq",
        "Sequence", "sequence",
        "Seq", "seq",
        "Protein_sequence", "protein_sequence",
        "AA_sequence", "aa_sequence"
    ]

    name_col = None
    seq_col = None

    for col in name_col_candidates:
        if col in columns:
            name_col = col
            break

    for col in seq_col_candidates:
        if col in columns:
            seq_col = col
            break

    # 如果按有表头读取后找不到序列列，很可能是无表头文件被 pandas 把第一条序列当成表头。
    # 这里把“列名”恢复成第一行数据，再按无表头处理。
    if seq_col is None:
        recovered_first_row = pd.DataFrame([list(df.columns)], columns=df.columns)
        recovered_df = pd.concat([recovered_first_row, df], ignore_index=True)
        recovered_df.columns = range(recovered_df.shape[1])
        return normalize_headerless_table_input(recovered_df)

    if name_col is None:
        names = [f"Seq_{i + 1}" for i in range(len(df))]
    else:
        names = df[name_col].fillna("").astype(str).tolist()
        names = [
            name if name.strip() else f"Seq_{i + 1}"
            for i, name in enumerate(names)
        ]

    return pd.DataFrame({
        "Name": names,
        "Mut_seq": df[seq_col].apply(clean_sequence)
    })



def strip_wrapping_quotes(line: str) -> str:
    """
    去掉整行首尾包裹的引号。

    例如：
    "Mut_001,ACDE..."
    会变成：
    Mut_001,ACDE...

    只去掉首尾成对的单引号或双引号，不影响序列内部字符。
    """
    line = str(line).strip()
    if len(line) >= 2:
        if (line[0] == line[-1]) and line[0] in ['"', "'"]:
            return line[1:-1].strip()
    return line


def text_looks_like_table(text: str) -> bool:
    """
    判断 txt 内容是否更像表格，而不是纯序列或 FASTA。

    支持这类格式：
    "Name,Mut_seq"
    "Mut_001,AAAA..."
    "Mut_002,BBBB..."

    或：
    Mut_001,AAAA...
    Mut_002,BBBB...
    """
    lines = [strip_wrapping_quotes(x) for x in text.splitlines() if x.strip()]

    if not lines:
        return False

    if lines[0].startswith(">"):
        return False

    # 只要前几行中有逗号、制表符，基本就可以按表格文本处理
    preview = lines[:5]
    if any("," in line or "\t" in line for line in preview):
        return True

    return False


def parse_table_text(text: str, table_has_header: Optional[bool] = None) -> pd.DataFrame:
    """
    解析表格型 txt。

    支持：
    1. 有表头：
       "Name,Mut_seq"
       "Mut_001,AAAA..."

    2. 无表头：
       "Mut_001,AAAA..."
       "Mut_002,BBBB..."

    3. 单列无表头：
       "AAAA..."
       "BBBB..."

    注意：
    有些 txt 每一整行都被双引号包起来，例如：
       "Mut_001,AAAA..."
    这会导致 pandas 认为整行是一个字段，所以这里先去掉整行包裹引号。
    """
    from io import StringIO

    raw_lines = [line for line in text.splitlines() if line.strip()]
    cleaned_lines = [strip_wrapping_quotes(line) for line in raw_lines]
    cleaned_text = "\n".join(cleaned_lines)

    # 根据内容选择分隔符
    if any("\t" in line for line in cleaned_lines[:5]):
        sep = "\t"
    else:
        sep = ","

    # 自动判断是否有表头
    if table_has_header is None:
        first = cleaned_lines[0].strip().lower()
        first_cells = [x.strip().lower() for x in first.split(sep)]

        known_header_keywords = {
            "name", "id", "mutant", "sample", "sample_id",
            "mut_seq", "sequence", "seq", "protein_sequence", "aa_sequence",
            "fitness", "activity"
        }

        if any(cell in known_header_keywords for cell in first_cells):
            has_header = True
        else:
            has_header = False
    else:
        has_header = table_has_header

    df = pd.read_csv(
        StringIO(cleaned_text),
        sep=sep,
        header=0 if has_header else None
    )

    return normalize_table_input(df, has_header=has_header)


def read_user_input(
    file_path: Optional[Union[str, Path]] = None,
    manual_text: Optional[str] = None,
    table_has_header: Optional[bool] = None
) -> pd.DataFrame:
    """
    读取用户输入，并统一转换为包含 Name 和 Mut_seq 的 DataFrame。

    参数：
    - file_path: 用户上传的文件路径，支持 txt/fasta/fa/faa/csv/xls/xlsx
    - manual_text: 用户手动输入的文本，可以是单条序列或 FASTA 文本
    - table_has_header: 仅用于 CSV/Excel
        True：强制按有表头读取
        False：强制按无表头读取
        None：自动判断，推荐

    返回：
    - DataFrame: Name, Mut_seq
    """
    if manual_text is not None and str(manual_text).strip():
        text = str(manual_text).strip()

        if text.startswith(">") or "\n>" in text:
            return parse_fasta_text(text)
        else:
            return parse_plain_text_as_sequences(text)

    if file_path is None:
        raise ValueError("file_path 和 manual_text 至少需要提供一个。")

    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext in [".fasta", ".fa", ".faa"]:
        text = read_text_file(file_path)
        return parse_fasta_text(text)

    if ext == ".txt":
        text = read_text_file(file_path)

        if text.strip().startswith(">") or "\n>" in text:
            return parse_fasta_text(text)

        # 支持表格型 txt，例如：
        # "Name,Mut_seq"
        # "Mut_001,ACDE..."
        # 或：
        # "Mut_001,ACDE..."
        if text_looks_like_table(text):
            return parse_table_text(text, table_has_header=table_has_header)

        return parse_plain_text_as_sequences(text)

    if ext == ".csv":
        has_header = detect_table_header(file_path, ext) if table_has_header is None else table_has_header
        df = pd.read_csv(file_path, header=0 if has_header else None)
        return normalize_table_input(df, has_header=has_header)

    if ext in [".xls", ".xlsx"]:
        has_header = detect_table_header(file_path, ext) if table_has_header is None else table_has_header
        df = read_excel_file(file_path, header=0 if has_header else None)
        return normalize_table_input(df, has_header=has_header)

    raise ValueError(
        f"暂不支持该文件格式：{ext}。"
        "请上传 txt、fasta、fa、faa、csv、xls 或 xlsx 文件。"
    )


def identify_mutations(seq: str, wt_seq: str) -> List[str]:
    """
    在序列长度一致的情况下，与野生型序列逐位比较，识别突变位点。

    返回示例：
    ["Q28Y", "R69F", "E137M"]
    """
    mutations = []

    for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, seq), start=1):
        if wt_aa != mut_aa:
            mutations.append(f"{wt_aa}{i}{mut_aa}")

    return mutations


def validate_sequence(
    seq: str,
    wt_seq: Optional[str] = None,
    allow_unknown: bool = False,
    allow_indel: bool = False
) -> Dict[str, object]:
    """
    检查单条序列是否合法。

    检查内容：
    1. 是否为空
    2. 是否包含非法氨基酸字符
    3. 是否包含未知或非标准氨基酸字符
    4. 是否与野生型长度一致
    5. 是否可能存在插入/缺失
    6. 若长度一致，识别突变位点

    参数：
    - seq: 清洗后的氨基酸序列
    - wt_seq: 野生型序列。如果提供，则会检查长度并识别突变位点
    - allow_unknown: 是否允许 X/B/Z/J/U/O 等未知或非标准氨基酸
    - allow_indel: 是否允许插入/缺失突变

    返回：
    - dict
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "length": len(seq),
        "mutations": [],
        "mutation_count": 0,
        "has_unknown": False,
        "has_indel": False,
    }

    if not seq:
        result["valid"] = False
        result["errors"].append("序列为空。")
        return result

    seq_chars = set(seq)

    unknown_chars = seq_chars & UNKNOWN_AA
    invalid_chars = seq_chars - VALID_AA

    if unknown_chars:
        result["has_unknown"] = True

    if invalid_chars:
        if allow_unknown and invalid_chars <= UNKNOWN_AA:
            result["warnings"].append(
                "序列中包含未知或非标准氨基酸字符："
                + ",".join(sorted(invalid_chars))
            )
        else:
            result["valid"] = False
            result["errors"].append(
                "序列中存在非法或未知氨基酸字符："
                + ",".join(sorted(invalid_chars))
                + "。当前模型默认仅支持标准 20 种氨基酸。"
            )

    if wt_seq is not None:
        wt_seq = clean_sequence(wt_seq)
        wt_len = len(wt_seq)

        if len(seq) != wt_len:
            result["has_indel"] = True
            diff = len(seq) - wt_len

            if diff > 0:
                msg = (
                    f"输入序列长度为 {len(seq)} aa，野生型长度为 {wt_len} aa，"
                    f"多出 {diff} aa，可能存在插入突变。"
                )
            else:
                msg = (
                    f"输入序列长度为 {len(seq)} aa，野生型长度为 {wt_len} aa，"
                    f"缺少 {abs(diff)} aa，可能存在缺失突变。"
                )

            if allow_indel:
                result["warnings"].append(msg)
            else:
                result["valid"] = False
                result["errors"].append(
                    msg + "当前模型暂不支持插入/缺失类型突变。"
                )
        else:
            mutations = identify_mutations(seq, wt_seq)
            result["mutations"] = mutations
            result["mutation_count"] = len(mutations)

    return result


def validate_sequences_dataframe(
    input_df: pd.DataFrame,
    wt_seq: Optional[str] = None,
    allow_unknown: bool = False,
    allow_indel: bool = False
) -> pd.DataFrame:
    """
    批量检查序列，生成检查报告 DataFrame。

    输入 DataFrame 需要至少包含：
    - Name
    - Mut_seq

    输出报告列：
    - Name
    - Raw_length
    - Clean_length
    - Valid
    - Mutation_count
    - Mutations
    - Has_unknown
    - Has_indel
    - Errors
    - Warnings
    """
    report_rows = []

    for _, row in input_df.iterrows():
        name = str(row.get("Name", "")).strip()
        raw_seq = str(row.get("Mut_seq", ""))
        clean_seq = clean_sequence(raw_seq)

        check = validate_sequence(
            clean_seq,
            wt_seq=wt_seq,
            allow_unknown=allow_unknown,
            allow_indel=allow_indel
        )

        report_rows.append({
            "Name": name,
            "Raw_length": len(str(raw_seq)),
            "Clean_length": len(clean_seq),
            "Valid": check["valid"],
            "Mutation_count": check["mutation_count"],
            "Mutations": ";".join(check["mutations"]),
            "Has_unknown": check["has_unknown"],
            "Has_indel": check["has_indel"],
            "Errors": ";".join(check["errors"]),
            "Warnings": ";".join(check["warnings"]),
        })

    return pd.DataFrame(report_rows)


def prepare_prediction_file(
    input_df: pd.DataFrame,
    wt_seq: Optional[str],
    output_path: Union[str, Path],
    report_path: Optional[Union[str, Path]] = None,
    allow_unknown: bool = False,
    allow_indel: bool = False,
    fitness_placeholder: float = 0.0
) -> Tuple[Path, pd.DataFrame]:
    """
    将输入序列 DataFrame 转换成标准预测文件。

    标准预测文件格式：
    Name, Mut_seq, fitness

    只有通过检查的序列会写入标准预测文件。
    所有序列都会写入检查报告。

    参数：
    - input_df: read_user_input() 生成的 DataFrame
    - wt_seq: 野生型序列
    - output_path: 标准预测文件保存路径
    - report_path: 检查报告保存路径，可选
    - allow_unknown: 是否允许未知氨基酸
    - allow_indel: 是否允许插入/缺失
    - fitness_placeholder: fitness 占位值，默认 0.0

    返回：
    - output_path
    - report_df
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

    if input_df.empty:
        raise ValueError("未读取到任何序列。")

    if "Name" not in input_df.columns or "Mut_seq" not in input_df.columns:
        raise ValueError("input_df 必须包含 Name 和 Mut_seq 两列。")

    # 清洗序列
    work_df = input_df.copy()
    work_df["Name"] = work_df["Name"].astype(str)
    work_df["Mut_seq"] = work_df["Mut_seq"].apply(clean_sequence)

    # 生成检查报告
    report_df = validate_sequences_dataframe(
        work_df,
        wt_seq=wt_seq,
        allow_unknown=allow_unknown,
        allow_indel=allow_indel
    )

    # 只保留合法序列用于预测
    valid_names = set(report_df.loc[report_df["Valid"] == True, "Name"])
    pred_df = work_df[work_df["Name"].isin(valid_names)].copy()

    if pred_df.empty:
        # 仍然保存报告，方便用户查看原因
        if report_path is not None:
            report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

        error_summary = report_df[["Name", "Errors"]].to_string(index=False)
        raise ValueError(
            "没有可用于预测的合法序列，原因如下：\n"
            + error_summary
        )

    # 增加 fitness 占位列，用于兼容 PySAR / pred_v2.py
    pred_df["fitness"] = fitness_placeholder

    # 保证列顺序
    pred_df = pred_df[["Name", "Mut_seq", "fitness"]]

    pred_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    if report_path is not None:
        report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    return output_path, report_df


def convert_user_input_to_prediction_file(
    output_path: Union[str, Path],
    wt_seq: Optional[str],
    file_path: Optional[Union[str, Path]] = None,
    manual_text: Optional[str] = None,
    report_path: Optional[Union[str, Path]] = None,
    table_has_header: Optional[bool] = None,
    allow_unknown: bool = False,
    allow_indel: bool = False,
    fitness_placeholder: float = 0.0
) -> Tuple[Path, pd.DataFrame]:
    """
    一站式函数：
    用户输入/文件 -> 统一读取 -> 清洗检查 -> 标准预测文件 + 检查报告

    推荐在软件界面中直接调用该函数。
    """
    input_df = read_user_input(
        file_path=file_path,
        manual_text=manual_text,
        table_has_header=table_has_header
    )

    return prepare_prediction_file(
        input_df=input_df,
        wt_seq=wt_seq,
        output_path=output_path,
        report_path=report_path,
        allow_unknown=allow_unknown,
        allow_indel=allow_indel,
        fitness_placeholder=fitness_placeholder
    )


if __name__ == "__main__":
    """
    命令行简单测试示例。

    运行方式示例：
    python input_parser.py
    """

    # 示例野生型序列。正式使用时，请替换为真实木聚糖酶野生型序列。
    WT_SEQUENCE = 'GVFGENQKRNDQPFAWQVASLSERYQEQFDIGAAVEPYQLEGRQAQILKHHYNSLVAENAMKPVSLQPREGEWNWEGADKIVEFARKHNMELRFHTLVWHSQVPEWFFIDENGNRMVDETDPEKRKANKQLLLERMENHIKTVVERYKDDVTSWDVVNEVIDDGGGLRESEWYQITGTDYIKVAFETARKYGGEEAKLYINDYNTEVPSKRDDLYNLVKDLLEQGVPIDGVGHQSHIQIGWPSIEDTRASFEKFTSLGLDNQVTELDMSLYGWPPTGAYTSYDDIPEELFQAQADRYDQLFELYEELSATISSVTFWGIADNHTWLDDRAREYNNGVGVDAPFVFDHNYRVKPAYWRIID'

    # 示例：手动输入一条序列
    manual_seq = 'GVFGENQKRNDQPFAWQVASLSERYQEFDIGAAVEPYQLEGRYQAQILKHHYNSLVAENAMKPVSLQPREGEWNWEGADKIVEFARKHNMELRFHTLVWHSQVPEWFFIDENGNRMVDETDPEKRKANKQLLLERMENHIKTVVERYKDDVTSWDVVNEVIDDGGGLRESEWYQITGTDYIKVAFETARKYGGEEAKLYINDYNTEVPSKRDDLYNLVKDLLEQGVPIDGVGHQSHIQIGWPSIEDTRASFEKFTSLGLDNQVTELDMSLYGWPPTGAYTSYDDIPEELFQAQADRYDQLFELYEELSATISSVTFWGIADNHTWLDDRAREYNNGVGVDAPFVFDHNYRVKPAYWRIID'

    out_path, report = convert_user_input_to_prediction_file(
        #manual_text=manual_seq,
        file_path="Test_Sample/TXT_titled_column.txt",
        wt_seq=WT_SEQUENCE,
        output_path="software_temp/input_standard.csv",
        report_path="software_temp/input_check_report.csv",
    )

    print(f"标准预测文件已生成：{out_path}")
    print(report)
