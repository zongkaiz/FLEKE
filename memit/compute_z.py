from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook

from .memit_hparams import MEMITHyperParams


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,#获取 lm_head 模块（这是 GPT 模型等语言模型中的输出层，它将隐藏层的表示映射到词汇表的分布。）的权重矩阵，并对其进行了转置
        nethook.get_module(model, hparams.ln_f_module),#hparams.ln_f_module='transformer.ln_f' 'transformer.ln_f' 是指 Transformer 模块的最终归一化层。在 Transformer 的结构中，这个归一化层位于网络末尾，紧接着输出层 lm_head。ln_f 层的作用是对输出的表示进行标准化，以保证值范围均衡，稳定模型的输出。
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]#要修改的新知识的id

    # Compile list of rewriting and KL x/y pairs  kl_prompts=[‘{} is a’]对应ROME论文3.1 Step3 上面那段里描述
    rewriting_prompts, kl_prompts = [#根据提供的 context_templates 创建两组不同的上下文：rewriting_prompts 和 kl_prompts。rewriting_prompts用于生成要插入的目标，而 kl_prompts 用于后续 KL 散度的计算，帮助保持模型整体分布的一致性。
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],#对每个 prompt 进行 format 替换，将 request["subject"] 填入到每个 prompt 的 {} 占位符位置。这样就可以根据 request 的具体内容生成一个包含上下文的 prompt 列表。
        return_tensors="pt",#将分词后的结果返回为 PyTorch 张量格式，便于后续直接输入到模型。
        padding=True,#自动为输入序列进行 padding，使得它们的长度相同，以便可以一起组成批量输入。
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(#初始化 rewriting_targets 张量：生成一个值为 -100 的张量，并将其放在 GPU 上
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]#再将张量扩展成与 input_tok["input_ids"] 形状一致的张量。具体来说，它会在第一个维度（行）上重复 len(rewriting_prompts) 次，这样每一个 prompt 都有一个对应的行。最终的 rewriting_targets 形状为 (len(rewriting_prompts), sequence_length)，其中 sequence_length 对应于input_tok["input_ids"] 的长度。
    )
    for i in range(len(rewriting_prompts)):#循环遍历 rewriting_prompts：遍历每个 prompt，为每个 prompt 设置它的目标序列。
        ex_len = input_tok["attention_mask"][i].sum()#计算第 i 个 prompt 的有效 token 长度（非填充部分）。attention_mask 是一个 0-1 序列，填充部分为 0，有效 token 为 1，因此求和即可得出有效 token 的长度。
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids#最终结构是一个目标张量，告诉模型每个 prompt 下应该生成什么 token。在优化过程中，rewriting_targets 会与模型输出进行比较，生成一个损失值。这一损失用于引导模型的梯度更新，使得模型在输入特定 prompt 时能够生成 target_ids 中指定的目标 token。

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [#每个模板中，主语最后一个token的id列表 比如Lookup index found: 7 | Sentence: The mother tongue of Danielle Darrieux is | Token: ux
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)#verbose=(i == 0)是减少不必要的冗长输出，同时提升代码的可读性和执行效率。
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)#这里用到了超参数里的27层（hparams.v_loss_layer=27） 第二个参数layer是rewrite layer
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.#这段注释描述的是一个通过优化潜在向量在 rewrite layer 产生特定输出的过程，这样在经过模型的后续层时，可以让模型在最终预测中生成目标 token。这种设置通过局部层的插入影响模型最终的输出，是 MEMIT 算法实现知识插入的核心机制。
    delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")#GPT-6J的model.config.n_embd是4096
    #delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device="cuda")#GPTNeoX的
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                #print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                #print(cur_out[0][i, idx, :].device)##
                cur_out[0][i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)#设置优化器：Adam 优化器 hparams.v_lr 是学习率超参数，控制每次更新 delta 的步长。
    nethook.set_requires_grad(False, model)#将模型的所有参数的 requires_grad 属性设置为 False，即冻结模型参数，不参与梯度更新。这样可以确保在优化过程中，只有 delta 被更新，而模型本身的权重不会改变。这种设计保持了模型的原始结构，同时通过调整 delta 来实现目标知识的插入。

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):#hparams.v_num_grad_steps=25
        opt.zero_grad()#为什么每次循环一次都要先清零梯度？1.避免梯度累积；2.保证每次更新的独立性；3.控制优化步幅

        # Forward propagation
        with nethook.TraceDict(#TraceDict 用于监控模型中指定层的输出，捕捉 loss_layer 和 layer 的输出。edit_output=edit_output_fn 表示在特定层的输出上应用 delta
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(#从 logits 中提取与 KL 散度相关的特定位置的值（通过kl_prompts），以计算模型在指定上下文（由 kl_prompts 提供）下的输出分布
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:#这两行代码确保了 kl_distr_init 只在第一次迭代时被初始化，保存模型在 kl_prompts 上的初始输出分布。续的 KL 散度计算会使用这个初始分布作为对比，以衡量插入知识前后的分布差异，从而在插入新知识的同时保持模型生成行为的稳定性。detach() 会断开 kl_log_probs 与计算图的连接，防止梯度回传影响 kl_distr_init。
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][#TraceDict 会记录特定层的输出，这里提取的是在 loss_layer 记录的输出
            : len(rewriting_prompts)#由于 tr 包含了 rewriting_prompts 和 kl_prompts 的输出，这里只提取与 rewriting_prompts 相关的部分。
        ]
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)#ln_f(full_repr)：对 full_repr 应用层归一化函数 ln_f，对表示进行标准化。@ lm_w + lm_b：lm_w 和 lm_b 分别是语言模型头部的权重和偏置，用于将 full_repr 从隐藏层维度映射到词汇表维度。@ 是矩阵乘法，将标准化的表示映射到词汇表上，得到未归一化的 logits。log_softmax 将 logits 转换为 log 概率分布，在词汇表维度上进行归一化（dim=2）。结果 log_probs 是 log 概率分布，表示模型生成每个 token 的对数概率。
        loss = torch.gather(#从 log_probs 中提取出目标 token 的 log 概率。
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),#将 rewriting_targets 中 -100 的位置替换为 0，确保我们只对非 -100 部分计算损失。
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()#掩码张量，它确保损失计算只在指定的目标 token 位置进行，不计算填充或无关位置。为什么上面已经把-100的位置换成0还要设置mask?这是因为log_probs值的0索引处的值会被取出来，而这些是不要的（因为我们只要主语最后一个token即非零的，但他上面gather其实是都取出来了），如果不用mask，log_probs值的0索引处的值会参与计算，这不对

        # Aggregate total losses         nll_loss_each对应ROME（4）式第一个括号 或 MEMIT（16）式后面那部分  nll_loss就是加上1/p求和
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)#计算每个句子的平均负对数似然损失 (NLL Loss) 形状为 (batch_size,)。#.sum(1)：在 sequence_length 维度上求和，得到每个句子的总负对数似然损失
        nll_loss = nll_loss_each.mean()#计算整个 batch 的平均 NLL 损失。
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(#kl_loss对应ROME（4）式第二个括号
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"#kl_log_probs 为目标分布的 log 概率。kl_distr_init 为初始分布的 log 概率。log_target=True 表示目标分布已经在 log 空间中。reduction="batchmean" 对 batch 中所有样本取平均。
        )#为什么顺序是 kl_distr_init 在前：在此情境中，kl_distr_init 被视为 目标分布（即我们希望模型维持的分布），而 kl_log_probs 是 近似分布（即插入知识后模型生成的分布）。通过计算 D_{KL}(\text{kl_distr_init} || \text{kl_log_probs})，我们试图衡量并最小化插入知识前后模型行为的变化，使得插入新知识不会显著改变模型原本的分布。
        weight_decay = hparams.v_weight_decay * (#计算权重衰减项 (Weight Decay) 权重衰减项在损失函数中加入了正则化，抑制 delta 过大，防止模型在插入知识时出现不稳定。这种约束确保模型对 delta 的更新保持适度，从而防止对生成行为的过度调整。
            torch.norm(delta) / torch.norm(target_init) ** 2#torch.norm(delta)：delta 的 L2 范数，衡量 delta 的大小。torch.norm(target_init) ** 2：target_init 的 L2 范数的平方，用于缩放 delta 的范数，使其在插入的表示与原始表示间保持合理的比例。
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay#比ROME （4）式多个正则项：weight_decay
        print(#输出样式：loss 3.682 = 3.682 + 0.0 + 0.0 avg prob of [ English] 0.026477385312318802（等式后面三部分对应上面一行代码等式后面的三部分；最后那个数字是模型生成目标 token（English） 的平均概率）
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )#np.round(,3)是保留三位小数；torch.exp(-nll_loss_each).mean().item()：将每个句子的负对数似然损失 nll_loss_each 转换成概率，并对 batch 取平均值，用于表示模型生成目标 token 的平均概率。
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball将 delta 向量的大小约束在一个 L2 范数球内，确保 delta 不会超过预设的最大范数，以防止模型在插入知识时发生过度调整，从而保持稳定性
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:#如果 delta 超出允许的范数
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()#将 delta 的大小缩小到 max_norm，而方向保持不变。这里通过将 delta 缩放到 max_norm 的比例来实现 L2 范数约束

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(#ret是主语最后的一个token在模版句子中的序号
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
