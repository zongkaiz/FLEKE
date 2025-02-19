import copy
import os
import sys

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 设置工作目录为项目的根目录
sys.path.append(os.path.join(current_dir, '..'))  # 假设 baselines 目录在上一级目录
###########################上面这几行是调试时用的，不然会显示没法import别的文件################################
import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# from baselines.ft import FTHyperParams, apply_ft_to_model
# from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from util.eval_utils.eval_utils_counterfact import compute_rewrite_quality_counterfact
from util.eval_utils.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams, apply_memit_to_model
from pmet import PMETHyperParams, apply_pmet_to_model
from util import nethook
from util.globals import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # 新增
from pathlib import Path# 新增
ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "PMET": (PMETHyperParams, apply_pmet_to_model),
    # "ROME": (ROMEHyperParams, apply_rome_to_model),
    # "FT": (FTHyperParams, apply_ft_to_model),
    # "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}

#定义客户端数据的路径和客户端名称列表。
#CLIENTS_DIR = Path("20b_MEMIT_mcf_client")###
#NUM_CLIENTS = 8###
#CLIENT_NAMES = [f"client{i}" for i in range(1, NUM_CLIENTS + 1)]###
def load_client_vectors(client_path: Path, t: int, T: int) -> dict:
    """
    加载特定客户端和时刻 t 的 'v_star' 向量。将客户端的所有 npz 文件平均划分为 T 份，
    返回第 t 份对应的向量。

    Args:
        client_path (Path): 客户端目录路径。
        t (int): 时刻，范围从 0 到 T-1。
        T (int): 总时刻数。

    Returns:
        dict: 从 case_id 到 'v_star' 向量的映射。
    """
    vectors = {}
    
    # 获取所有 .npz 文件并排序以确保分配的一致性
    npz_files = sorted(client_path.glob("*.npz"))
    total_files = len(npz_files)
    
    if total_files == 0:
        print(f"警告: 在 {client_path} 中未找到任何 .npz 文件。")
        return vectors
    
    files_per_t = total_files // T
    remainder = total_files % T
    
    # 计算当前时刻 t 的文件索引范围
    start_idx = t * files_per_t + min(t, remainder)
    end_idx = start_idx + files_per_t + (1 if t < remainder else 0)
    
    selected_files = npz_files[start_idx:end_idx]
    
    print(f"加载 {client_path} 中时刻 {t} 的文件: {start_idx} 到 {end_idx} 共 {len(selected_files)} 个文件。")
    
    for npz_file in selected_files:
        try:
            case_id = int(npz_file.stem.split("_")[-1])
            data = np.load(npz_file)
            if 'v_star' in data:
                vectors[case_id] = data['v_star']
            else:
                print(f"警告: 文件 {npz_file} 中缺少 'v_star' 键。")
        except Exception as e:
            print(f"错误: 无法加载文件 {npz_file}。原因: {e}")
    
    return vectors
def index_ds_records(ds):
    """
    将数据集 ds 中的记录按照 case_id 进行索引。

    Args:
        ds: 数据集实例，支持迭代并包含 'case_id' 以及评估所需的其他字段。

    Returns:
        dict: 从 case_id 到完整记录的映射。
    """
    ds_index = {}
    for record in ds:
        case_id = record["case_id"]
        ds_index[case_id] = record  # 存储整个记录，包含所有字段
    return ds_index
def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    model_path: str = None,
    T: int = 4,  # 新增时刻参数
    similarity_threshold: float = 0.4,  # 余弦相似度阈值
    num_clients: int = 8,  # 客户端数量
    clients_dir: str = "20b_MEMIT_mcf_client"  # 客户端数据所在的目录
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        if model_path:
            print(f"Instantiating model: {model_name} from {model_path}")
            if "neox" in model_name:
                model = AutoModelForCausalLM.from_pretrained(model_path + model_name).half().cuda()
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path + model_name).cuda()
            tok = AutoTokenizer.from_pretrained(model_path + model_name)
        else:
            print(f"Instantiating model: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
            tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
    
    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None
    gen_test_vars = [snips, vec]
    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    # 索引 ds 数据集
    print("索引数据集 ds")
    ds_index = index_ds_records(ds)
    print(f"共索引了 {len(ds_index)} 条记录。")

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")
    print(f"kvs cache template: {cache_template}")
    # 预加载所有客户端的向量
    CLIENT_NAMES = [f"client{i}" for i in range(1, num_clients + 1)]###
    client_vectors = {}
    for client in CLIENT_NAMES:
        client_path = Path(clients_dir) / client
        client_vectors[client] = {}
        for t in range(T):
            client_vectors[client][t] = load_client_vectors(client_path, t, T)
    print("已加载所有客户端的向量。")
    args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
    etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["MEMIT", "PMET"]) else dict()
    # 遍历每个客户端
    for client in CLIENT_NAMES:
        print(f"正在处理 {client}")

        # 创建客户端的子文件夹
        client_dir = run_dir / client
        client_dir.mkdir(parents=True, exist_ok=True)

        # 存储所有编辑记录以供评估
        all_records = []
        
        for t in range(T):
            print(f"  时刻 {t}")

            # 提取当前客户端在时刻t的数据
            current_client_data = client_vectors[client][t]
            # 对 current_client_data 按键排序
            current_client_data = dict(sorted(current_client_data.items()))
            if not current_client_data:
                print(f"    {client} 在时刻 {t} 没有找到数据")
                continue

            # 准备当前客户端的数据记录，结合 ds 数据集的 requested_rewrite
            records = []
            for case_id, v_star in current_client_data.items():
                if case_id not in ds_index:
                    print(f"    警告: ds 数据集中不存在 case_id {case_id} 的记录。跳过。")
                    continue

                record = ds_index[case_id].copy()
                record["case_id"] = case_id  # 确保 'case_id' 存在
                records.append(record)

            if not records:
                print(f"    {client} 在时刻 {t} 没有有效的记录进行编辑。")
                continue

            # 应用编辑算法到当前客户端的数据
            print(f"    应用编辑算法到 {len(records)} 条记录。")
            start = time()
            edited_model, weights_copy = apply_algo(
                model if t==0 else edited_model,
                tok,
                [
                    {"case_id": record["case_id"], **record["requested_rewrite"]}
                    for record in records
                ],
                hparams,
                copy=False,
                return_orig_weights=True,
                **args_conserve_memory,
            **etc_args,
            )
            if t==0:
                weights_copy_new=copy.deepcopy(weights_copy)
            exec_time = time() - start
            print(f"    执行耗时 {exec_time:.2f} 秒。")

            # 将记录添加到所有编辑记录中以供评估
            all_records.extend(records)

            # 从其他客户端中找到相似的数据
            similar_records = []
            for other_client in CLIENT_NAMES:
                if other_client == client:
                    continue  # 跳过同一个客户端
                other_client_data = client_vectors[other_client][t]
                for other_case_id, other_v_star in other_client_data.items():
                    # 计算余弦相似度
                    # 获取当前客户端所有 v_star 向量的矩阵
                    current_vectors = np.array([v for v in current_client_data.values()])
                    other_vector = np.array([other_v_star])

                    # 计算所有相似度
                    similarities = cosine_similarity(current_vectors, other_vector).flatten()

                    # 检查是否有一半以上的相似度超过阈值
                    if np.sum(similarities > similarity_threshold) >= (len(similarities) / 2):
                        if other_case_id in ds_index:
                            # 复制整个记录，确保包含所有必要字段
                            similar_record = ds_index[other_case_id].copy()
                            similar_record["case_id"] = other_case_id  # 确保 'case_id' 存在
                            similar_records.append(similar_record)
                        else:
                            print(f"    警告: ds 数据集中不存在 case_id {other_case_id} 的记录。跳过。")

            print(f"    找到 {len(similar_records)} 条来自其他客户端的相似记录。")

            if similar_records:
                # 应用编辑算法到相似记录，保持结构不变
                print(f"    应用编辑算法到相似记录。")
                start = time()
                edited_model, weights_copy = apply_algo(
                    edited_model,
                    tok,
                    [
                        {"case_id": record["case_id"], **record["requested_rewrite"]}
                        for record in similar_records
                    ],
                    hparams,
                    copy=False,
                    return_orig_weights=True,
                    **args_conserve_memory,
            **etc_args,
                )
                exec_time = time() - start
                print(f"    相似记录的执行耗时 {exec_time:.2f} 秒。")

                # 将相似记录添加到所有编辑记录中以供评估
                all_records.extend(similar_records)
            else:
                print(f"    没有找到符合相似度要求的记录。")
        # 在处理完该客户端的所有时刻t后进行评估和恢复权重
        print(f"完成处理 {client} 的所有时刻。现在评估编辑后的模型。")
        start = time()
        # 评估整个客户端的编辑效果
        # 遍历所有编辑记录进行评估
        for record in all_records:
            out_file = client_dir / f"{num_edits}_edits-case_{record['case_id']}.json"
            if out_file.exists():
                print(f"      跳过 {out_file}; 已存在")
                continue
            
            metrics = {
                "case_id": record["case_id"],
                "grouped_case_ids": [record["case_id"] for record in all_records],  
                "num_edits": num_edits,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),  # 仅在满足间隔时进行生成测试
                ),
            }
            # 将指标保存为 .json 文件
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)
        print(f"评估耗时 {time() - start:.2f} 秒。")
        # 恢复原始权重
        print(f"恢复 {client} 的原始权重。")
        with torch.no_grad():
            for k, v in weights_copy_new.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")
        print(f"已恢复 {client} 的原始权重。")
        print(f"已完成处理 {client}。")

    print("所有客户端均已处理完毕。")
    


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "PMET"],
        default="MEMIT",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=False,
    )
    parser.add_argument(
        "--model_path",
        default="/media/h3c/users/zongkai/PMET1/"
    )
    parser.add_argument(
        "--model_name",
        choices=["EleutherAI/gpt-neox-20b", "EleutherAI/gpt-j-6B"],
        default="EleutherAI/gpt-neox-20b",
        help="Model to edit.",
        required=False,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="EleutherAI_gpt-neox-20b.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=False,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=100,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        default=True,
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=10,
        help="要处理的时刻数量。",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.4,
        help="用于识别相似数据的余弦相似度阈值。",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=8,
        help="客户端数量。",
    )
    parser.add_argument(
        "--clients_dir",
        type=str,
        default="20b_MEMIT_mcf_client",
        help="指定客户端数据所在的目录",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        model_path = args.model_path,
        T=args.T,
        similarity_threshold=args.similarity_threshold,
        num_clients=args.num_clients,
        clients_dir = args.clients_dir
    )