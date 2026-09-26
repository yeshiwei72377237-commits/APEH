"""
input_parser_txt.py

脚本功能概述
------------
本脚本用于将用户上传或手动输入的木聚糖酶突变体氨基酸序列，
统一转换为后续预测程序 pred_software.py / pred_v2.py / PySAR 可读取的标准输入格式。

它不是模型预测脚本，不直接输出活性或热稳定性预测结果。
它的作用是完成“输入解析、序列清洗、合法性检查、突变位点识别、标准预测文件生成”。

主要功能
--------
1. 支持多种输入来源
   - 单条手动输入的氨基酸序列
   - 手动输入的 FASTA 文本
   - txt 文件
   - fasta / fa / faa 文件
   - csv 文件
   - xls / xlsx 文件
   - 多个文件路径组成的列表
   - 多条手动序列组成的列表

2. 自动识别输入类型
   - 如果输入是存在的文件路径，则按文件读取
   - 如果输入不是文件路径，则按手动输入的序列或文本解析
   - 如果字符串看起来像文件路径但文件不存在，会报错，避免误当成氨基酸序列

3. 支持不同格式的序列文件
   - 纯序列文本
   - FASTA 格式
   - 带表头的表格，例如 Name, Mut_seq
   - 不带表头的表格
   - 逗号分隔 csv
   - 制表符分隔 txt
   - Excel 表格

4. 自动清洗序列
   - 转换为大写
   - 去除空格、换行、制表符
   - 去除数字编号
   - 去除常见分隔符，例如 "-" 和 "_"

5. 检查序列是否合法
   - 检查序列是否为空
   - 检查是否只包含标准 20 种氨基酸字符
   - 检查是否包含 B / J / O / U / X / Z 等未知或非标准氨基酸
   - 检查序列长度是否与野生型一致
   - 默认不允许插入或缺失突变
   - 默认不允许未知氨基酸字符

6. 识别突变位点
   - 在输入序列和野生型序列长度一致的情况下，逐位比较
   - 自动生成类似 Q28Y、R69F、E137M 这样的突变位点名称
   - 多突变会合并为 Q28Y;R69F;E137M

7. 生成标准预测输入文件
   标准输出文件包含三列：
   - Name
   - Mut_seq
   - fitness

   其中：
   - Name 默认使用识别出的突变位点作为名称，例如 "('Q28Y',)" 或 "('Q28Y', 'R69F')"
   - Mut_seq 为清洗后的突变体氨基酸序列
   - fitness 为占位值，默认 1.0，仅用于兼容后续预测程序读取，不代表真实实验结果

8. 生成输入检查报告
   检查报告包含：
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

   该报告用于追踪每条输入序列是否合法，以及不合法的具体原因。

主要输出
--------
本脚本最终会输出两类文件：

1. 标准预测输入文件
   默认后缀为 .txt
   示例：
   - software_temp/input_standard.txt
   - software_temp/S7_mut_1.txt
   - software_temp/manual_seq_1.txt

2. 输入检查报告
   默认后缀为 .csv
   示例：
   - software_temp/input_check_report.csv
   - software_temp/S7_mut_1_input_check_report.csv
   - software_temp/manual_seq_1_input_check_report.csv

重要说明
--------
1. 直接运行本脚本时，如果 __main__ 部分的示例代码仍然被注释，则不会生成输出文件。

2. 若要单独测试本脚本，可以取消 __main__ 部分示例 1 或示例 2 的注释。

3. 示例 1 用于单个文件输入测试：
   convert_user_input_to_prediction_file(...)

4. 示例 2 用于多个文件或多个输入同时处理：
   convert_user_inputs_to_prediction_files(...)

5. 如果读取 .xlsx 文件，需要当前 Python 环境安装 openpyxl：
   conda install openpyxl
   或：
   pip install openpyxl

6. 本脚本默认只支持标准 20 种氨基酸：
   A C D E F G H I K L M N P Q R S T V W Y

7. 本脚本默认不支持插入或缺失突变。
   输入序列长度必须与野生型序列长度一致。

8. 本脚本只负责生成预测程序的输入文件，不负责调用模型，也不负责输出最终预测结果。
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Sequence

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


SUPPORTED_INPUT_EXTENSIONS = {".txt", ".fasta", ".fa", ".faa", ".csv", ".xls", ".xlsx"}


def looks_like_possible_file_path(value: object) -> bool:
    """
    粗略判断一个字符串是否“有必要”按文件路径继续检查。

    这个函数的目的不是判断文件是否存在，而是避免把很长的氨基酸序列
    直接传给 Path(...).is_file()，否则在 macOS / Linux 下可能触发：
    OSError: [Errno 63] File name too long

    判断为可能文件路径的情况：
    1. 是 Path 对象；
    2. 字符串带有受支持的文件扩展名，例如 .txt / .csv / .xlsx / .fasta；
    3. 字符串长度没有明显超过常见文件名长度；
    4. 字符串中没有换行符。
    """
    if isinstance(value, Path):
        return True

    if value is None:
        return False

    raw = str(value).strip()
    if not raw:
        return False

    # 手动输入的 FASTA / 多行序列一般会包含换行，不应当作路径检查
    if "\n" in raw or "\r" in raw:
        return False

    # 很长的纯序列不要进入 Path(...).is_file()
    # 这里用 240 做保守阈值，避免触发系统文件名长度限制
    if len(raw) > 240:
        return False

    # 没有受支持扩展名，就没有必要当作文件路径处理
    suffix = Path(raw).suffix.lower()
    if suffix not in SUPPORTED_INPUT_EXTENSIONS:
        return False

    return True


def is_existing_supported_file(value: Union[str, Path]) -> bool:
    """
    判断输入是否为已经存在且格式受支持的文件路径。

    关键修正：
    - 先用 looks_like_possible_file_path() 过滤；
    - 对 Path(...).is_file() 增加 OSError 保护；
    - 长氨基酸序列不会再被当成文件名检查。
    """
    if not looks_like_possible_file_path(value):
        return False

    try:
        path = Path(value)
        return path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
    except (TypeError, OSError, ValueError):
        return False


def string_has_supported_file_suffix(value: object) -> bool:
    """
    判断字符串是否看起来像一个受支持格式的文件名。
    例如：sample.csv、/path/to/sample.fasta。

    注意：
    - 这里同样先过滤明显不是路径的长序列，避免误判。
    """
    if not looks_like_possible_file_path(value):
        return False

    raw = str(value).strip()
    return Path(raw).suffix.lower() in SUPPORTED_INPUT_EXTENSIONS


def classify_user_input_item(item: Union[str, Path]) -> Tuple[str, Union[str, Path]]:
    """
    自动判断一个输入项是“文件路径”还是“手动输入文本/序列”。

    返回：
    - ("file", Path(...))：真实存在的受支持文件
    - ("manual", str(...))：手动输入的序列、FASTA 文本或表格型文本

    判断原则：
    1. 如果路径真实存在且扩展名受支持 -> 文件
    2. 如果看起来像文件名，但文件不存在 -> 报错，不把它误判为序列
    3. 其他字符串 -> 按手动输入文本处理，后续再进行序列合法性检查
    """
    if is_existing_supported_file(item):
        return "file", Path(item)

    raw = str(item).strip()
    if not raw:
        raise ValueError("输入项为空。")

    if string_has_supported_file_suffix(raw):
        raise FileNotFoundError(
            f"输入看起来像文件路径，但文件不存在或格式不受支持：{raw}"
        )

    return "manual", raw


def safe_stem_from_input(item: Union[str, Path], index: int) -> str:
    """
    根据输入项生成安全的文件名主体。

    文件输入：优先使用原文件名 stem。
    手动输入：使用 manual_001、manual_002 ...
    """
    if is_existing_supported_file(item):
        stem = Path(item).stem
    else:
        stem = f"manual_{index:03d}"

    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(stem)).strip("_")
    return stem or f"input_{index:03d}"




def make_unique_path(path: Union[str, Path]) -> Path:
    """
    如果目标文件已存在，自动追加 _1、_2，避免覆盖。

    示例：
    - TXT_titled.txt 已存在 -> TXT_titled_1.txt
    - TXT_titled_1.txt 已存在 -> TXT_titled_2.txt
    """
    path = Path(path)
    if not path.exists():
        return path

    parent = path.parent
    stem = path.stem
    suffix = path.suffix

    # 如果传入的 stem 已经以 _数字 结尾，则先还原成基础文件名，
    # 避免出现 TXT_titled_1_1.txt 这样的名字。
    match = re.match(r"^(.*)_(\d+)$", stem)
    if match:
        base_stem = match.group(1)
    else:
        base_stem = stem

    counter = 1
    while True:
        candidate = parent / f"{base_stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def split_manual_inputs(inputs: Union[str, Path, Sequence[Union[str, Path]]]) -> List[Union[str, Path]]:
    """
    将手动输入的多个序列拆成多个输入项。

    设计目标：
    - 文件路径列表保持不变；
    - 单条手动序列保持为 1 个输入项；
    - FASTA 文本中有多条记录时，拆成多条 FASTA 文本，每条单独输出；
    - 非 FASTA 多行文本中，如果每一行都像一条完整氨基酸序列，则拆成多条序列；
    - 如果是复制粘贴的分段序列，则仍保持为一个输入项，交给 parse_plain_text_as_sequences 合并。
    """
    input_items = normalize_user_inputs(inputs)
    expanded_items: List[Union[str, Path]] = []

    for item in input_items:
        input_type, value = classify_user_input_item(item)

        # 文件输入不拆分，保留原文件名。
        if input_type == "file":
            expanded_items.append(value)
            continue

        text = str(value).strip()
        if not text:
            continue

        # FASTA 手动输入：如果包含多条记录，则拆成多条单独 FASTA。
        if text.startswith(">") or "\n>" in text:
            fasta_df = parse_fasta_text(text)
            if fasta_df.empty:
                expanded_items.append(text)
            else:
                for i, row in fasta_df.iterrows():
                    name = str(row.get("Name", f"manual_seq_{i + 1}")).strip() or f"manual_seq_{i + 1}"
                    seq = str(row.get("Mut_seq", "")).strip()
                    expanded_items.append(f">{name}\n{seq}")
            continue

        # 表格型手动文本仍作为一个输入项处理，避免误拆 CSV/TSV。
        if text_looks_like_table(text):
            expanded_items.append(text)
            continue

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # 多行且每行都像完整序列时，拆成多条手动输入。
        # 对复制粘贴的 60 aa 分段序列，通常每行明显短于 WT，后续会作为一个输入项合并。
        if len(lines) > 1 and all(looks_like_sequence(line, min_length=30) for line in lines):
            expanded_items.extend(lines)
        else:
            expanded_items.append(text)

    if not expanded_items:
        raise ValueError("inputs 不能为空。")

    return expanded_items


def make_output_paths_for_input(
    item: Union[str, Path],
    index: int,
    output_dir: Union[str, Path],
    input_type: str,
    output_prefix: str = "input_standard",
    report_prefix: str = "input_check_report",
    output_suffix: str = ".txt",
    report_suffix: str = ".csv",
    preserve_file_stem: bool = True,
    avoid_overwrite: bool = False
) -> Tuple[Path, Path]:
    """
    根据输入项生成标准预测文件和检查报告路径。

    preserve_file_stem=True 时：
    - 文件输入保留原文件名主体。
      例如：TXT_titled.txt -> software_temp/TXT_titled.txt
      例如：CSV_titled.csv -> software_temp/CSV_titled.txt
    - 手动输入没有原始文件名时，由 convert_user_inputs_to_prediction_files() 命名为 manual_seq_1.txt、manual_seq_2.txt ...。

    检查报告默认命名为：原文件名主体_input_check_report.csv
    避免与标准预测文件混淆。
    """
    output_dir = Path(output_dir)

    if preserve_file_stem and input_type == "file":
        stem = safe_stem_from_input(item, index)
        output_path = output_dir / f"{stem}{output_suffix}"
        report_path = output_dir / f"{stem}_{report_prefix}{report_suffix}"
    else:
        stem = safe_stem_from_input(item, index)
        output_path = output_dir / f"{output_prefix}_{idx:03d}_{stem}{output_suffix}" if False else output_dir / f"{output_prefix}_{index:03d}_{stem}{output_suffix}"
        report_path = output_dir / f"{report_prefix}_{index:03d}_{stem}{report_suffix}"

    if avoid_overwrite:
        output_path = make_unique_path(output_path)
        report_path = make_unique_path(report_path)

    return output_path, report_path

def read_user_input_auto(
    user_input: Union[str, Path],
    table_has_header: Optional[bool] = None
) -> pd.DataFrame:
    """
    自动读取单个输入项。

    user_input 可以是：
    1. 已存在的文件路径：txt / fasta / fa / faa / csv / xls / xlsx
    2. 单条氨基酸序列字符串
    3. FASTA 文本字符串
    4. 表格型文本字符串

    该函数用于兼容前端只传一个字符串的情况：
    - 如果字符串是存在的文件路径，则当文件读入；
    - 否则当作用户手动输入的序列/文本。
    """
    input_type, value = classify_user_input_item(user_input)

    if input_type == "file":
        return read_user_input(file_path=value, table_has_header=table_has_header)

    text = str(value).strip()

    if text.startswith(">") or "\n>" in text:
        return parse_fasta_text(text)

    if text_looks_like_table(text):
        return parse_table_text(text, table_has_header=table_has_header)

    return parse_plain_text_as_sequences(text)


def normalize_user_inputs(
    inputs: Union[str, Path, Sequence[Union[str, Path]]]
) -> List[Union[str, Path]]:
    """
    将前端传入的输入统一整理成列表。

    推荐前端：
    - 多文件：传 List[str]，每个元素是一个文件路径；
    - 单文件：可以传 str，也可以传 [str]；
    - 多条手动序列：推荐传 FASTA 文本字符串；也可以传 List[str]，每个元素一条序列。
    """
    if isinstance(inputs, (str, Path)):
        return [inputs]

    if isinstance(inputs, Sequence):
        normalized = list(inputs)
        if not normalized:
            raise ValueError("inputs 不能为空。")
        return normalized

    raise TypeError("inputs 必须是字符串、Path，或由字符串/Path 组成的列表。")


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

def format_mutation_name(mutations: str, fallback_name: str) -> str:
    """
    将突变位点字符串转换为 S7_mut 文件中的 Name 格式。

    输入：
    mutations = "Q28Y" 或 "Q28Y;R69F"

    输出：
    "('Q28Y',)" 或 "('Q28Y', 'R69F')"
    """
    if not mutations:
        return fallback_name

    mut_list = [m for m in str(mutations).split(";") if m]

    if not mut_list:
        return fallback_name

    return str(tuple(mut_list))

def prepare_prediction_file(
    input_df: pd.DataFrame,
    wt_seq: Optional[str],
    output_path: Union[str, Path],
    report_path: Optional[Union[str, Path]] = None,
    allow_unknown: bool = False,
    allow_indel: bool = False,
    fitness_placeholder: float = 1.0,
    use_mutation_name_as_name: bool = True
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
    valid_mask = report_df["Valid"] == True
    pred_df = work_df.loc[valid_mask].copy()
    valid_report_df = report_df.loc[valid_mask].copy()

    if pred_df.empty:
        # 仍然保存报告，方便用户查看原因
        if report_path is not None:
            report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

        error_summary = report_df[["Name", "Errors"]].to_string(index=False)
        raise ValueError(
            "没有可用于预测的合法序列，原因如下：\n"
            + error_summary
            )

    #如果希望输出格式与 S7_mut_1.txt 一致，则用突变位点作为 Name
    if use_mutation_name_as_name:
        pred_df["Name"] = [
            format_mutation_name(mutations, fallback_name)
            for mutations, fallback_name in zip(
                valid_report_df["Mutations"],
                pred_df["Name"]
            )
        ]

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
    fitness_placeholder: float = 1.0,
    use_mutation_name_as_name: bool = True
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
        fitness_placeholder=fitness_placeholder,
        use_mutation_name_as_name=use_mutation_name_as_name
    )


def convert_user_inputs_to_prediction_files(
    inputs: Union[str, Path, Sequence[Union[str, Path]]],
    wt_seq: Optional[str],
    output_dir: Union[str, Path] = "software_temp",
    output_prefix: str = "input_standard",
    report_prefix: str = "input_check_report",
    table_has_header: Optional[bool] = None,
    allow_unknown: bool = False,
    allow_indel: bool = False,
    fitness_placeholder: float = 1.0,
    use_mutation_name_as_name: bool = True,
    output_suffix: str = ".txt",
    report_suffix: str = ".csv",
    preserve_file_stem: bool = True,
    avoid_overwrite: bool = False
) -> List[Dict[str, object]]:
    """
    批量转换用户输入，适合软件前端一次选择一个或多个文件。

    参数：
    - inputs:
        1. 单个文件路径字符串 / Path
        2. 单条序列字符串
        3. FASTA 文本字符串
        4. List[str | Path]：多个文件路径，或多条手动序列
    - output_dir:
        输出目录，默认 software_temp
    - output_prefix:
        标准预测文件名前缀
    - report_prefix:
        检查报告文件名前缀

    输出文件命名：
    - 默认 preserve_file_stem=True：文件输入保留原文件名主体。
      例如 TXT_titled.txt -> software_temp/TXT_titled.txt
      例如 CSV_titled.csv -> software_temp/CSV_titled.txt
    - 手动输入没有原始文件名，输出为 manual_seq_1.txt、manual_seq_2.txt ...。
    - 检查报告命名为 原文件名主体_input_check_report.csv。

    返回：
    - List[dict]，每个输入项对应一个结果，包含：
      input_index, input_type, source, output_path, report_path, report_df

    说明：
    - “对应数量的文件”指每个输入项生成 1 个标准预测文件；
    - 如果某个输入文件内部有多条序列，会写入同一个标准预测文件的多行；
    - 如果想让多条手动序列生成一个文件，建议前端传入一个 FASTA 文本字符串；
    - 如果前端传入 List[str]，则每个字符串会各自生成一个标准预测文件。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_items = split_manual_inputs(inputs)
    results: List[Dict[str, object]] = []
    manual_seq_counter = 0

    for idx, item in enumerate(input_items, start=1):
        input_type, value = classify_user_input_item(item)

        # 手动输入的序列按 manual_seq_1、manual_seq_2 ... 命名。
        # 这些文件名不再表达突变位点；突变位点请查看对应 input_check 文件。
        if input_type == "manual":
            manual_seq_counter += 1
            output_path = output_dir / f"manual_seq_{manual_seq_counter}{output_suffix}"
            report_path = output_dir / f"manual_seq_{manual_seq_counter}_{report_prefix}{report_suffix}"

            if avoid_overwrite:
                output_path = make_unique_path(output_path)
                report_path = make_unique_path(report_path)
        else:
            output_path, report_path = make_output_paths_for_input(
                item=value,
                index=idx,
                output_dir=output_dir,
                input_type=input_type,
                output_prefix=output_prefix,
                report_prefix=report_prefix,
                output_suffix=output_suffix,
                report_suffix=report_suffix,
                preserve_file_stem=preserve_file_stem,
                avoid_overwrite=avoid_overwrite
            )

        input_df = read_user_input_auto(
            user_input=value,
            table_has_header=table_has_header
        )

        final_output_path, report_df = prepare_prediction_file(
            input_df=input_df,
            wt_seq=wt_seq,
            output_path=output_path,
            report_path=report_path,
            allow_unknown=allow_unknown,
            allow_indel=allow_indel,
            fitness_placeholder=fitness_placeholder,
            use_mutation_name_as_name=use_mutation_name_as_name
        )

        results.append({
            "input_index": idx,
            "input_type": input_type,
            "source": str(value),
            "output_path": final_output_path,
            "report_path": report_path,
            "report_df": report_df,
        })

    return results


if __name__ == "__main__":
    """
    本部分为脚本单独运行时的测试示例。

    注意：
    1. 默认情况下，下面的示例代码是注释状态，直接运行脚本不会生成任何文件。
    2. 如果需要测试脚本功能，可以取消“示例 1”或“示例 2”的注释。
    3. 测试前需要确认示例中的输入文件路径真实存在。
    4. 正式软件流程中，通常由前端或主程序调用下面两个核心函数：

       convert_user_input_to_prediction_file()
       - 用于处理单个输入文件或单条手动序列

       convert_user_inputs_to_prediction_files()
       - 用于批量处理多个输入文件或多条输入序列

    5. 本脚本生成的是模型预测前的标准输入文件和检查报告，
       不会直接输出活性或热稳定性预测结果。
    """

    # 示例野生型序列。正式使用时，请替换为真实木聚糖酶野生型序列。
    WT_SEQUENCE = 'GVFGENQKRNDQPFAWQVASLSERYQEQFDIGAAVEPYQLEGRQAQILKHHYNSLVAENAMKPVSLQPREGEWNWEGADKIVEFARKHNMELRFHTLVWHSQVPEWFFIDENGNRMVDETDPEKRKANKQLLLERMENHIKTVVERYKDDVTSWDVVNEVIDDGGGLRESEWYQITGTDYIKVAFETARKYGGEEAKLYINDYNTEVPSKRDDLYNLVKDLLEQGVPIDGVGHQSHIQIGWPSIEDTRASFEKFTSLGLDNQVTELDMSLYGWPPTGAYTSYDDIPEELFQAQADRYDQLFELYEELSATISSVTFWGIADNHTWLDDRAREYNNGVGVDAPFVFDHNYRVKPAYWRIID'

    # 示例：手动输入一条序列
    manual_seq = 'GVFGENQKRNDQPFAWQVASLSERYQEFDIGAAVEPYQLEGRYQAQILKHHYNSLVAENAMKPVSLQPREGEWNWEGADKIVEFARKHNMELRFHTLVWHSQVPEWFFIDENGNRMVDETDPEKRKANKQLLLERMENHIKTVVERYKDDVTSWDVVNEVIDDGGGLRESEWYQITGTDYIKVAFETARKYGGEEAKLYINDYNTEVPSKRDDLYNLVKDLLEQGVPIDGVGHQSHIQIGWPSIEDTRASFEKFTSLGLDNQVTELDMSLYGWPPTGAYTSYDDIPEELFQAQADRYDQLFELYEELSATISSVTFWGIADNHTWLDDRAREYNNGVGVDAPFVFDHNYRVKPAYWRIID'

    # 示例 1：单个输入，兼容原来的写法
    # out_path, report = convert_user_input_to_prediction_file(
    #     file_path="Test_Sample/S7_mut_1.txt",
    #     wt_seq=WT_SEQUENCE,
    #     output_path="software_temp/input_standard.txt",
    #     report_path="software_temp/input_check_report.csv",
    #     fitness_placeholder=1.0,
    #     use_mutation_name_as_name=True
    # )
    # print(f"标准预测文件已生成：{out_path}")
    # print(report)

    # 示例 2：自动判断字符串是文件路径还是序列本身，并支持多个输入
    # 前端多文件选择时，建议直接把路径列表传给 inputs。
    # batch_results = convert_user_inputs_to_prediction_files(
    #     inputs=[
    #         "Test_Sample/S7_mut_1.txt",
    #         manual_seq,
    #         "Test_Sample/XLSX_untitled.xlsx",
    #         ],
    #         wt_seq=WT_SEQUENCE,
    #         output_dir="software_temp",
    #         fitness_placeholder=1.0,
    #         use_mutation_name_as_name=True
    #     )

    # for item in batch_results:
    #     print(f"第 {item['input_index']} 个输入已生成：{item['output_path']}")
    #     print(f"检查报告已生成：{item['report_path']}")
    #     print(item["report_df"])
