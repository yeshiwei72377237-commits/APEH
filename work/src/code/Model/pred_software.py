from pySAR.pySAR import *
from pySAR.encoding import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from scipy.stats import spearmanr, kendalltau
import re
import pandas as pd
import numpy as np
import os, json
import shutil
from tqdm import tqdm
pys = PySAR('data.json')
    
# 获取路径下文件内包含的氨基酸索引和特征描述符
DESCRIPTOR_NAMES_FOR_MODEL = [
    "amino_acid_composition",
    "dipeptide_composition",
    "conjoint_triad",
]


def _parse_aai_desc_from_rank_columns(sorted_path):
    """
    优先从 rank 文件列名中解析 AAI 和描述符组合。

    这样比从文件名解析更稳健，因为 descriptor 本身含有下划线，
    例如 amino_acid_composition / dipeptide_composition，
    直接 split('_') 很容易把描述符拆碎，导致误判。
    """
    df_head = pd.read_csv(sorted_path, nrows=0)
    columns = list(df_head.columns)

    index_cols = [col for col in columns if str(col).startswith("Index_")]
    if not index_cols:
        return None, None

    # 正常情况下只有一个 Index_ 列；如果有多个，优先选含有已知描述符最多的列。
    def score_col(col):
        col = str(col)
        return sum(desc in col for desc in DESCRIPTOR_NAMES_FOR_MODEL)

    index_col = max(index_cols, key=score_col)
    feature_key = str(index_col)[len("Index_"):]

    descriptor_positions = []
    for desc in DESCRIPTOR_NAMES_FOR_MODEL:
        pos = feature_key.find(desc)
        if pos != -1:
            descriptor_positions.append((pos, desc))

    descriptor_positions = sorted(descriptor_positions, key=lambda x: x[0])

    if len(descriptor_positions) < 2:
        return None, None

    first_desc_pos = descriptor_positions[0][0]
    aa_index = feature_key[:first_desc_pos]
    desc1 = descriptor_positions[0][1]
    desc2 = descriptor_positions[1][1]

    if not aa_index:
        return None, None

    sorted_dic = {
        0: aa_index,
        1: desc1,
        2: desc2,
    }
    return sorted_dic, index_col


def _parse_aai_desc_from_filename_fallback(sorted_path):
    """
    兜底：从文件名解析模型信息。

    仅在 rank 文件列名无法解析时使用。相比旧版本，不再简单 split('_')，
    而是直接查找完整描述符名称在文件名中的位置。
    """
    stem = os.path.splitext(os.path.basename(str(sorted_path)))[0]
    if stem.startswith("rank_"):
        stem = stem[len("rank_"):]
    if stem.endswith("_sfs"):
        stem = stem[:-len("_sfs")]

    if "_" not in stem:
        raise ValueError(f"无法从模型文件名解析算法和 AAI：{stem}")

    algorithm, rest = stem.split("_", 1)

    descriptor_positions = []
    for desc in DESCRIPTOR_NAMES_FOR_MODEL:
        pos = rest.find(desc)
        if pos != -1:
            descriptor_positions.append((pos, desc))

    descriptor_positions = sorted(descriptor_positions, key=lambda x: x[0])
    if len(descriptor_positions) < 2:
        raise ValueError(
            f"无法从文件名解析两个描述符：{stem}\n"
            f"请检查文件名是否包含 amino_acid_composition / dipeptide_composition / conjoint_triad。"
        )

    aa_index = rest[:descriptor_positions[0][0]]
    if not aa_index:
        raise ValueError(f"无法从文件名解析 AAI index：{stem}")

    return {
        0: aa_index,
        1: descriptor_positions[0][1],
        2: descriptor_positions[1][1],
    }


def find_aai_desc(sorted_path, count=False):
    """
    获取模型对应的 AAI index 和两个特征描述符。

    修正版逻辑：
    1. 优先从 rank 文件的 Index_ 列名解析，例如：
       Index_TANS770101amino_acid_compositionconjoint_triad
       -> TANS770101 + amino_acid_composition + conjoint_triad
    2. 如果列名无法解析，再从文件名兜底解析。

    这样可以避免旧版把 amino_acid_composition 误解析成 dipeptide_composition 的问题。
    """
    sorted_dic, index_col = _parse_aai_desc_from_rank_columns(sorted_path)

    if sorted_dic is None:
        sorted_dic = _parse_aai_desc_from_filename_fallback(sorted_path)
        df_head = pd.read_csv(sorted_path, nrows=0)
        expected_col = 'Index_' + sorted_dic[0] + sorted_dic[1] + sorted_dic[2]
        index_col = expected_col if expected_col in df_head.columns else df_head.columns[0]

    if count:
        df = pd.read_csv(sorted_path)
        if index_col not in df.columns:
            raise ValueError(
                f"在 rank 文件中未找到解析出的 Index 列：{index_col}\n"
                f"当前 rank 文件实际列名为：{list(df.columns)}"
            )

        count_dic = {}
        aa_index = sorted_dic[0]
        count_dic[aa_index] = df[index_col].astype(str).str.contains(aa_index, regex=False).sum()
        for k in [1, 2]:
            key = sorted_dic[k]
            count_dic[key] = df[index_col].astype(str).str.contains(key, regex=False).sum()
        return sorted_dic, count_dic

    return sorted_dic

# 获取Shap排序前的位置
def get_original_index(sorted_path, idx_sfs):
    """
    根据 SHAP 排序文件 sorted_path 和 SFS 选择的 idx_sfs，
    返回 PySAR encode_aai_descriptor 所需要的原始特征列索引 index_ori。

    兼容两类 rank 文件编号规则：

    1. 全局编号规则：
       例如：
       aai_1
       dipeptide_composition_433
       conj_triad_777

       这种情况下，编号已经对应完整特征矩阵中的列编号，
       因此直接使用 feature_no - 1。

    2. 局部编号规则：
       例如：
       RACS820103_1
       amino_acid_composition_5
       dipeptide_composition_123
       conjoint_triad_20

       这种情况下，编号是各描述符内部编号，
       需要根据 AAI / AAC / DPC / CTD 的排列顺序加偏移量。
    """

    sorted_dic, count_dic = find_aai_desc(sorted_path, count=True)

    df_rank = pd.read_csv(sorted_path)

    col_name = 'Index_' + sorted_dic[0] + sorted_dic[1] + sorted_dic[2]

    if col_name not in df_rank.columns:
        raise ValueError(
            f"在 rank 文件中未找到列：{col_name}\n"
            f"当前 rank 文件实际列名为：{list(df_rank.columns)}"
        )

    all_feature_names = df_rank.loc[:, col_name].tolist()

    selected_feature_names = []
    for idx in idx_sfs:
        if idx < 0 or idx >= len(all_feature_names):
            raise IndexError(
                f"idx_sfs 中的索引 {idx} 超出 rank 文件范围。"
                f"rank 文件共有 {len(all_feature_names)} 行。"
            )
        selected_feature_names.append(all_feature_names[idx])

    def extract_last_number(feature_name):
        """
        从特征名末尾提取数字。

        支持：
        RACS820103_1
        RACS8201031
        amino_acid_composition_5
        dipeptide_composition_433
        conjoint_triad_20
        conj_triad_777
        aai_1
        """
        feature_name = str(feature_name).strip()
        match = re.search(r'(\d+)$', feature_name)

        if match is None:
            raise ValueError(
                f"无法从特征名中提取编号：{feature_name}。"
                "请检查 rank 文件中特征名是否以数字结尾。"
            )

        return int(match.group(1))

    def is_global_index_rule(feature_names, sorted_path):
        """
        判断当前 rank 文件是否更像“全局编号规则”。

        Bagging 模型的 rank 文件中常见：
        aai_1, conj_triad_423, conj_triad_777 等。

        这类编号已经对应完整特征矩阵的列编号，
        不能再额外加 360、400 等偏移。
        """
        file_name = os.path.basename(sorted_path).lower()

        # 你当前的 bagging 模型属于全局编号格式
        if "baggingregressor" in file_name:
            return True

        for name in feature_names[:50]:
            name = str(name).strip().lower()
            feature_no = extract_last_number(name)

            # PySAR 生成后的全局 AAI 特征常见为 aai_1, aai_2 ...
            if name.startswith("aai_"):
                return True

            # conj_triad_423 / conj_triad_777 这种编号超过 343，
            # 说明它不是 conjoint triad 内部编号，而是全局编号
            if name.startswith("conj_triad") and feature_no > 343:
                return True

            # dipeptide_composition_433 这种编号超过 400，
            # 说明它不是 dipeptide 内部编号，而是全局编号
            if name.startswith("dipeptide_composition") and feature_no > 400:
                return True

            # amino_acid_composition_25 超过 20，也说明可能是全局编号
            if name.startswith("amino_acid_composition") and feature_no > 20:
                return True

        return False

    def descriptor_dim(desc_name):
        """
        返回描述符维度。
        """
        if desc_name == "amino_acid_composition":
            return 20
        elif desc_name == "dipeptide_composition":
            return 400
        elif desc_name == "conjoint_triad":
            return 343
        else:
            raise ValueError(f"未知描述符类型：{desc_name}")

    def is_conjoint_triad_name(feature_name):
        """
        兼容 conjoint_triad 和 conj_triad 两种写法。
        """
        feature_name = str(feature_name).strip().lower()
        return feature_name.startswith("conjoint_triad") or feature_name.startswith("conj_triad")

    index_ori = []

    use_global_rule = is_global_index_rule(selected_feature_names, sorted_path)

    print("sorted_dic =", sorted_dic)
    print("selected feature examples =", selected_feature_names[:10])
    print("use_global_index_rule =", use_global_rule)

    if use_global_rule:
        # Bagging 模型：rank 文件中的编号已经是完整特征矩阵中的全局编号
        for feature_name in selected_feature_names:
            feature_no = extract_last_number(feature_name)
            index_ori.append(feature_no - 1)

    else:
        # GradientBoosting 等旧模型：rank 文件中的编号是局部编号，需要加偏移
        aai_dim = 360
        desc1 = sorted_dic[1]
        desc2 = sorted_dic[2]

        desc1_dim = descriptor_dim(desc1)

        for feature_name in selected_feature_names:
            feature_name = str(feature_name).strip()
            feature_no = extract_last_number(feature_name)

            # AAI 特征，例如 RACS820103_1
            if feature_name.startswith(sorted_dic[0]):
                index_ori.append(feature_no - 1)

            # 第一个描述符
            elif feature_name.startswith(desc1):
                index_ori.append(aai_dim + feature_no - 1)

            # 第二个描述符
            elif feature_name.startswith(desc2) or is_conjoint_triad_name(feature_name):
                index_ori.append(aai_dim + desc1_dim + feature_no - 1)

            else:
                raise ValueError(
                    f"无法识别特征类型：{feature_name}。\n"
                    f"当前识别到的 sorted_dic 为：{sorted_dic}\n"
                    f"请检查 rank 文件中特征名前缀是否与描述符名称一致。"
                )

    print("index_ori min:", min(index_ori))
    print("index_ori max:", max(index_ori))
    print("index_ori length:", len(index_ori))

    return index_ori

# 建立预测模型
def pred(sfs_path, sorted_path, pys):
    # 获取SFS结果
    df = pd.read_csv(sfs_path)
    first_row_values = str(df.iloc[0, 0])  # 第一行第一列
    idx_sfs = [int(x) for x in re.findall(r'\d+', first_row_values)]

    # 获取Shap排序前的位置和算法
    sorted_dic = find_aai_desc(sorted_path, count = False)
    index_ori = get_original_index(sorted_path, idx_sfs)
    alg = sorted_path.split('/')[-1].split('_')[1]

    # 使用特征选择后的特征，重新训练模型
    pys.algorithm = alg
    pys.encode_aai_descriptor(indices = sorted_dic[0], 
                              descriptors = [sorted_dic[1], sorted_dic[2]],
                              show_plot = False, 
                              print_results = True,
                              idx_flag = True,
                              idx = index_ori
                              )

# 使用S7_mut序列预测结果并生成文件
def make_output_csv_path(input_file_path, output_path):
    """
    根据输入文件名生成输出 CSV 路径。

    示例：
    - 输入：TXT_titled.txt
    - 输出：software_output/TXT_titled.csv

    如果输出目录中已经存在同名文件，则自动追加 _001、_002，避免覆盖。
    """
    os.makedirs(output_path, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(str(input_file_path)))[0]
    base_name = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff]+', '_', base_name).strip('_')
    if not base_name:
        base_name = 'Result'

    csv_path = os.path.join(output_path, base_name + '.csv')

    if not os.path.exists(csv_path):
        return csv_path

    counter = 1
    while True:
        candidate = os.path.join(output_path, f"{base_name}_{counter:03d}.csv")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


# 使用S7_mut序列预测结果并生成文件
def sum_code(pys2, new_path, output_path, sorted_dic, index_ori, model):
    # 读取desc2中的desc
    pys2.encode_aai_descriptorforp(indices = sorted_dic[0], 
                                   descriptors = [sorted_dic[1], sorted_dic[2]],
                                   show_plot = False, 
                                   print_results = False,
                                   idx_flag = True, 
                                   idx = index_ori
                                   )
    # 预测
    pys2.dataset = new_path
    pys2.data = pys2.read_data()
    pys2.sequences = pys2.data[pys2.sequence_col]
    pys2.activity = pys2.data[pys2.activity_col]
    
    # 存储：输出文件名与输入文件名保持一致，只把后缀改为 .csv
    csv_path = make_output_csv_path(new_path, output_path)
    fitness_result = model.predict_new(pys2.model.X)

    if len(fitness_result) != len(pys2.data):
        raise ValueError(
            f"预测结果数量与输入序列数量不一致。\n"
            f"当前输入文件：{new_path}\n"
            f"输入序列数量：{len(pys2.data)}\n"
            f"预测结果数量：{len(fitness_result)}\n"
            f"这通常说明 desc2 描述符缓存没有清空，或 pys2.model.X 没有根据当前文件重新生成。"
        )

    df1 = pd.DataFrame({'Mutant': pys2.data['Name'].to_numpy(),
                        'Seq': pys2.data['Mut_seq'].to_numpy(),
                       #'Fitness': pys2.data['fitness'],
                        'Fitness_pred': fitness_result
                        })
    df1.to_csv(csv_path, index=False, float_format='%.6f') # 生成csv文件，结果保留6位小数
    print(f"预测结果已生成：{csv_path}")

# 强制清空 desc2/
def clear_desc2_cache(desc_dir="desc2"):
    if os.path.exists(desc_dir):
        for f in os.listdir(desc_dir):
            file_path = os.path.join(desc_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(desc_dir)

# 保存计算出的描述符，重复使用；并调用以上函数进行预测以及输出结果文件
def generate_desc_pred(folder_path, output_path, sfs_path, sorted_path, pys):
    # 获取SFS结果
    df = pd.read_csv(sfs_path)
    first_row_values = str(df.iloc[0, 0])  # 第一行第一列
    idx_sfs = [int(x) for x in re.findall(r'\d+', first_row_values)]
    index_ori = get_original_index(sorted_path, idx_sfs)

    # 获取描述符
    sorted_dic = find_aai_desc(sorted_path, count = False)
    desc_list = [sorted_dic[1], sorted_dic[2]]

    # 建立预测模型
    pred(sfs_path, sorted_path, pys)

    # 查看保留的特征数量
    print('被保留的特征维度数量：' + str(len(index_ori)))
    
    # 获取 S7_mut 文件夹中的所有文件
    files = os.listdir(folder_path)
    files = [f for f in files if f.endswith('.txt')]  # 只取 .txt 文件

    # 创建保存文件夹（如果没有）
    if not os.path.exists('desc2'):
        os.makedirs('desc2')
    if not os.path.exists('desc2_save'):
        os.makedirs('desc2_save') 

    # 修改 JSON 中的 dataset 路径
    for file in files:
        # 构建新的路径
        new_path = os.path.join(folder_path, file)
        print('当前处理文件：' + new_path)
        
        # 在每处理一个文件前，强制清空 decs2/
        clear_desc2_cache("desc2")
        # 尝试读取mut序列文件，若文件为空则跳过
        try:
            df_check = pd.read_csv(new_path, sep=None, engine="python", on_bad_lines="skip")
        except Exception as e:
            print(f"[跳过本文件] {file} 读取失败: {e}")
            continue
        # 读取 JSON 文件
        with open("data_Pred.json", 'r') as f:
            data = json.load(f)
        # 替换路径
        data["dataset"]["dataset"] = new_path
        # 保存修改后的 JSON 文件
        with open("data_Pred.json", 'w') as f:
            json.dump(data, f, indent=4)

        pys2 = PySAR(config_file = 'data_Pred.json')
        desc = Descriptors("data_Pred.json")
        count = new_path.split('_')[-1][:-4] # 获取突变位点数量，用作之后文件保存的编号

        for index in tqdm(desc_list[:int(len(desc_list))], unit = " indices", position = 0, 
                          desc = "AAI Indices", mininterval = 30):
            file_path = 'desc2/{}.csv'.format(index)
            df = desc.get_descriptor_encoding_pred(index, iftest = True)
            df.to_csv(file_path, index = False)
        
        sum_code(pys2, new_path, output_path, sorted_dic, index_ori, pys.model)
        
        # # 保存文件到 "desc2_save" 并删除原文件
        # for index in desc_list[:int(len(desc_list))]:
        #     old_file_path = 'desc2/{}.csv'.format(index)
        #     new_file_path = 'desc2_save/{}_{}.csv'.format(index, count)
        #     if os.path.exists(old_file_path):
        #         shutil.move(old_file_path, new_file_path)
  
folder_path = 'software_temp' 
output_path = 'software_output'
#sfs_path = 'Trained/H_shap_sfs/xgb_JANJ780102dipeptide_compositionconjoint_triad_sfs.csv'
#sorted_path = 'Trained/H_shapvalues_rank/rank_xgb_JANJ780102dipeptide_compositionconjoint_triad.csv'
sfs_path = 'Trained//A_shap_sfs/xgb_TANS770101amino_acid_compositionconjoint_triad_sfs.csv'
sorted_path = 'Trained/A_shapvalues_rank/rank_xgb_TANS770101amino_acid_compositionconjoint_triad.csv'
#generate_desc_pred(folder_path, output_path, sfs_path, sorted_path, pys)