"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets


def compute_rewrite_quality_zsre(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.#拿case_id=1举例
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )#'Ramalinaceae'  {'str': 'Lecanorales'}   {'str': '<|endoftext|>'}
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]#['Which family does Ramalinaceae belong to?']
    paraphrase_prompts = record["paraphrase_prompts"]#['What family are Ramalinaceae?']
    neighborhood_prompts = record["neighborhood_prompts"]#[{'prompt': 'nq question: types of skiing in the winter olympics 2018?', 'target': ' Down'}, {'prompt': 'nq question: types of skiing in the winter olympics 2018? Down', 'target': 'hill'}]

    # Form a list of lists of prefixes to test.
    prob_prompts = [               #[['Which family does Ramalinaceae belong to?'], ['What family are Ramalinaceae?']]
        rewrite_prompts,
        paraphrase_prompts,
    ]
    # Flatten all the evaluated prefixes into one list.
    target_tok = tok(" " + target_new["str"])["input_ids"]#将目标字符串（如 " Lecanorales"）转为分词后的 input_ids
    inp_prompts_og = list(chain(*prob_prompts))#['Which family does Ramalinaceae belong to?', 'What family are Ramalinaceae?'] 相当于把prob_prompts拆了一层列表
    inp_prompts = [#构造输入前缀：['Which family does Ramalinaceae belong to?', 'Which family does Ramalinaceae belong to? L', 'Which family does Ramalinaceae belong to? Lec', 'Which family does Ramalinaceae belong to? Lecan', 'Which family does Ramalinaceae belong to? Lecanor', 'What family are Ramalinaceae?', 'What family are Ramalinaceae? L', 'What family are Ramalinaceae? Lec', 'What family are Ramalinaceae? Lecan', 'What family are Ramalinaceae? Lecanor']
        el + tok.decode(target_tok[:i])
        for el in inp_prompts_og
        for i in range(len(target_tok))
    ]
    inp_targets = [#对应于每个前缀，构造目标字符串的后缀：[' L', 'ec', 'an', 'or', 'ales', ' L', 'ec', 'an', 'or', 'ales']在自然语言生成任务（如语言模型重写任务）中，模型的输出通常是逐步生成的，而不是一次性生成完整的目标字符串。通过构造目标字符串的后缀，可以更细粒度地评估模型是否能够在每一步生成正确的目标内容。
        tok.decode(target_tok[i])
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]

    stuff_probs = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets)#测试模型的回答准确率

    # Predict for neighborhood prompts (dictionary format).
    neighborhood_correct = test_batch_prediction_acc(
        model,
        tok,
        [
            el["prompt"].format(record["requested_rewrite"])
            for el in neighborhood_prompts
        ],
        [el["target"] for el in neighborhood_prompts],
    )

    probs = stuff_probs + neighborhood_correct#合并了 rewrite_prompts 和 paraphrase_prompts 的准确率，以及邻域问题的准确率。

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(#计算不同问题类型在 probs 中的切分索引，用于将结果分组。
        [l * len(target_tok) for l in map(len, prob_prompts)]
    ).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]#将 probs 切分为与问题类型一致的子列表。
    # Structure the restuls as a dictionary.
    ret = {#返回结果字典
        f"{key}_correct": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }
    ret["neighborhood_prompts_correct"] = neighborhood_correct

    return ret


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target):
    prompt_tok = tok(#对输入提示进行分词
        prompts,
        padding=True,# 对输入进行补全，使得所有输入序列长度相同
        return_tensors="pt",# 返回 PyTorch 张量
    ).to("cuda")#  将张量移动到 GPU 上

    with torch.no_grad():#关闭梯度计算，因为这是推理阶段，不需要反向传播。
        logits = model(**prompt_tok).logits#model(**prompt_tok)：分词后的输入张量传入模型。返回的结果包括每个 token 的 logits（即未归一化的概率分布）。   logits：张量形状为 (batch_size, sequence_length, vocab_size)。表示每个输入序列的每个 token 在词汇表（vocabulary）上所有可能输出的分数。
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1#获取每个输入序列的最后一个非填充位置
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)#提取最后一个 token 的 logits
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)#获取模型预测的 token

        correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")[#对目标字符串进行分词
            "input_ids"
        ]
        # Temporary hack to deal with foreign characters.
        correct_id = correct_id[:, 0].squeeze()

        return (ans == correct_id).detach().cpu().numpy().tolist()#比较预测和目标
