import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .memit_hparams import MEMITHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

#execute_memit 负责计算更新矩阵，但不会对模型做持久性更改。apply_memit_to_model 负责将这些更新应用到模型中，并进行持久性的权重更新。这样做可以保持计算和应用更新的逻辑分离，便于调试和管理模型更新。
def apply_memit_to_model(#应用 MEMIT 算法进行模型的更新
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating(重新分配) the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}#保存模型中将被修改的权重，以便之后可以还原。
    if copy:
        model = deepcopy(model)

    deltas = execute_memit(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():#这部分才是真正的插入部分
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_memit(#MEMIT 算法的核心部分
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function（这个注释表示函数会对模型做某些暂时的计算操作，但在结束时，会恢复模型的初始状态，不会对模型产生永久性的更改。
    """
    # Retrieve weights that user desires to change
    weights = {#提取出来修要修改的模型权重，就是345678层的fc_out_weight，例如：'transformer.h.3.mlp.fc_out.weight' 、weights['transformer.h.4.mlp.fc_out.weight']
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Divide requests into num_clients parts
    num_clients = 8

    # Create a dictionary to store weights for each client
    #weights_clients = {
    #    f"client_{i+1}": {k: v.detach().clone() for k, v in weights.items()}
    #    for i in range(num_clients)
    #}

    split_requests = [requests[i::num_clients] for i in range(num_clients)]
    # Initialize deltas accumulators
    accumulated_deltas = {}
    all_deltas = []
    client=0
    zs_clients = []  # 用于存储每个客户端的 zs
    for requests in split_requests:
        print(f"第{client}个客户端正在计算zs")
        client+=1
        deltas = {}  # Initialize for each subset
        # Update target and print info
        requests = deepcopy(requests)
        for i, request in enumerate(requests):
            if request["target_new"]["str"][0] != " ":
                # Space required for correct tokenization这一步是在遍历 requests 时，确保每个 target_new["str"] 字符串的开头有一个空格，以确保在后续的分词步骤中，这个字符串能被模型正确地分词和处理。这对于确保插入的文本不会与前面的内容混淆是非常重要的。
                requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
        for request in requests[:10]:
            print(
                f"MEMIT request sample: "
                f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
            )
        # Compute z for final layer
        context_templates = get_context_templates(model, tok)
        z_layer = hparams.layers[-1]
        z_list = []

        for request in requests:
            # Retrieve k/v pair if already stored in cache
            cache_fname = (#往那个一开始代码产生的之前下载好的kv对儿文件名字中括号里填参数名字
                Path(
                    str(cache_template).format(
                        z_layer, hparams.clamp_norm_factor, request["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            data_loaded = False
            if (
                cache_fname is not None  # Require cache template
                and cache_fname.exists()  # Cache file must exist
            ):
                try:
                    data = np.load(cache_fname)
                    z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))#z的值代表模型在某一层（L 层）对给定输入的隐藏表示，即最后一层的向量h_i^L。维数通常是与 transformer 模型的隐藏层维度（hidden dimension）一致的。比如，如果你使用的是 GPT-J 模型，那么它的隐藏层维度是 4096，这意味着每个 z 向量的长度是 4096。
                    data_loaded = True
                except Exception as e:
                    print(f"Error reading cache file due to {e}. Recomputing...")

            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                cur_z = compute_z(
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer,
                    context_templates,
                )

                z_list.append(cur_z)

                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{
                            "v_star": cur_z.detach().cpu().numpy(),
                        },
                    )
                    print(f"Cached k/v pair at {cache_fname}")
        zs = torch.stack(z_list, dim=1)#zs 代表的是我们希望模型学会的新知识或目标表示，它是我们要插入或修改的目标
        # 在每个客户端循环的结尾处添加：
        zs_clients.append(zs) # 存储每个客户端的 zs
    # Insert
    for requests, zs in zip(split_requests, zs_clients):
        for i, layer in enumerate(hparams.layers):#从第三层开始插入
            print(f"\n\nLAYER {layer}\n")

            # Get current model activations
            layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
            print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

            # Compute residual error 论文中公式R
            cur_zs = get_module_input_output_at_words(#cur_zs 代表的是模型在没有应用插入或修改的情况下，给定输入和上下文时的原始表示。
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            targets = zs - cur_zs#论文中公式（14）中的R（Residual Error）
            print("z error", torch.linalg.norm(targets, dim=0).mean())

            repeat_factor = (layer_ks.size(1) // targets.size(1))#在维度 1 上的比例因子。这一比例因子表示的是目标表示 targets 的列数是否少于模型当前激活值 layer_ks 的列数。
            targets = targets.repeat_interleave(repeat_factor, dim=1)#这一行的作用是将 targets 在列方向（即 dim=1）重复 repeat_factor 次，使得 targets 的形状与 layer_ks 匹配。

            # Load covariance matrix   应该就是加载的data/stats下面的那几个文件
            force_recompute = False#如果 force_recompute 为 False，则可以避免重复计算，节省计算资源，直接加载先前已计算并存储的协方差矩阵。
            # force_recompute = layer != hparams.layers[0]
            cov = get_cov(#对应（15）式下面第一行
                model,
                tok,
                hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,#mom2 通常表示 二阶矩（second moment）。在统计学和机器学习中，二阶矩通常用来描述数据的分布特性，特别是数据的协方差结构
                hparams.mom2_n_samples
                if not force_recompute
                else hparams.mom2_n_samples // 10,
                hparams.mom2_dtype,
                force_recompute=force_recompute,
            )

            # Compute update in double precision
            layer_ks, targets = (#将变量 layer_ks 和 targets 转换为双精度浮点格式（torch.float64），以提高后续线性代数计算的数值精度。
                layer_ks.double(),
                targets.double(),
            )
            #通过求解线性方程组来计算矩阵 adj_k：（对应论文14式去掉R）
            adj_k = torch.linalg.solve(#λ 对应 hparams.mom2_update_weight。C_0是二者之积：hparams.mom2_update_weight * cov.double()。K_1对应 layer_ks
                hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,
                layer_ks,
            )#下面这行与论文算法描述不一致，按照论文算法应该是：resid = targets / (hparams.layers[-1] - hparams.layers[i] + 1)
            resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers算法1中eqn.20的r_i^l组合成一起的R^l
            upd_matrix = resid @ adj_k.T#式14，乘上R^lupd_matrix 计算了要应用于层权重的更新 Δ


            # Adjust update matrix shape
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)#将 upd_matrix（ Δ） 调整到与 weights[weight_name] 相同的形状。

            print("orig norm", torch.linalg.norm(weights[weight_name]))#计算更新矩阵 upd_matrix （ Δ）的 L2 范数（即更新的幅度）。打印这两个范数值可以帮助开发者了解更新的大小相对于原始权重的比例，从而检查更新是否在合理范围内。
            print("upd norm", torch.linalg.norm(upd_matrix))

            # Update model weights and record desired changes in `delta` variable
            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
                deltas[weight_name] = (
                    adj_k.detach().cpu(),
                    resid.detach().cpu(),
                )
            # Clear GPU memory
            cov.cpu()
            for x in [layer_ks, cur_zs, targets]:
                x.cpu()
                del x
            torch.cuda.empty_cache()
        # # After computing `deltas` for this subset, accumulate them
        #for k, v in deltas.items():
        #    if k in accumulated_deltas:
        #        accumulated_deltas[k] = tuple((accumulated_deltas[k][i] + v[i]) / 2 for i in range(2))
        #    else:
        #        accumulated_deltas[k] = v

        all_deltas.append(deltas)
        # Restore state of original model
        with torch.no_grad():
            for k, v in weights.items():
                v[...] = weights_copy[k]
        print(f"Deltas successfully computed for {list(weights.keys())}")

    # Compute the average of all clients' deltas
    for k in all_deltas[0]:  # Iterate through keys
        accumulated_deltas[k] = tuple(
            sum(delta[k][i] for delta in all_deltas) / len(all_deltas) for i in range(2)
        )
    return accumulated_deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics（协方差统计量；对应（15）式下面第一行）, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):#生成一些用于模型生成的上下文模板（context templates），并将其缓存到全局变量 CONTEXT_TEMPLATES_CACHE 中。上下文模板是用于填充模型生成文本时的一些模板化句子（例如：“The {text}”）。缓存是为了避免每次调用时都重复生成这些模板。
    global CONTEXT_TEMPLATES_CACHE#声明全局变量 CONTEXT_TEMPLATES_CACHE。这个缓存会用于存储生成的模板，避免多次生成相同的模板。

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"#这是对生成的结果进行处理，将模板中的 {} 替换为空格，并且在末尾追加 ". {}"。这会创建一个类似于 "The [生成的文本]. {}" 的结构，方便后续使用。
                for f in generate_fast(#这里这个generate_fast函数，是对下面第300行五个单词为起点，生成长度为max_out_len的模版
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,#n_gen 是总共要生成的模板数。n_gen_per_prompt 是对每个给定的提示词（prompt），生成的句子模板的数量
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
