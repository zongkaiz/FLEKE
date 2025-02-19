from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compute_z import get_module_input_output_at_words
from .memit_hparams import MEMITHyperParams


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
):
    layer_ks = get_module_input_output_at_words(#layer_ks就是context_templates把每一个文本换成tensor的表示
        model,
        tok,
        layer,
        context_templates=[
            context.format(request["prompt"])
            for request in requests
            for context_type in context_templates
            for context in context_type
        ],
        words=[
            request["subject"]
            for request in requests
            for context_type in context_templates
            for _ in context_type
        ],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )[0]

    context_type_lens = [0] + [len(context_type) for context_type in context_templates]#计算每个 context_type 的长度，并在前面加上 0
    context_len = sum(context_type_lens)
    context_type_csum = np.cumsum(context_type_lens).tolist()#计算 context_type_lens 的累积和，使用 np.cumsum 函数。累积和用于确定每个上下文的起始和结束位置，以用于确定每种上下文模板在 layer_ks 中的起始和结束位置。比如context_type_lens=【0,1,5】，那这里就为【0，1,6】

    ans = []
    for i in range(0, layer_ks.size(0), context_len):
        tmp = []
        for j in range(len(context_type_csum) - 1):
            start, end = context_type_csum[j], context_type_csum[j + 1]
            tmp.append(layer_ks[i + start : i + end].mean(0))
        ans.append(torch.stack(tmp, 0).mean(0))
    return torch.stack(ans, dim=0)
