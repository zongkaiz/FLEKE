import json
from itertools import chain
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from dsets import AttributeSnippets
from util.globals import *

REMOTE_IDF_URL = f"{REMOTE_ROOT_URL}/data/dsets/idf.npy"
REMOTE_VOCAB_URL = f"{REMOTE_ROOT_URL}/data/dsets/tfidf_vocab.json"


def get_tfidf_vectorizer(data_dir: str):
    """
    Returns an sklearn TF-IDF vectorizer(将文本数据转换为数值形式.在TF-IDF向量化过程中，每个文档被转换成一个向量，向量的每个元素对应于词汇表中的一个词，并使用该词的TF-IDF值作为对应元素的值). See their website for docs.
    Loading hack inspired by some online blog post lol.
    """

    data_dir = Path(data_dir)

    idf_loc, vocab_loc = data_dir / "idf.npy", data_dir / "tfidf_vocab.json"#idf_loc指向存储IDF值的.npy文件，vocab_loc指向存储词汇表的.json文件。
    if not (idf_loc.exists() and vocab_loc.exists()):
        collect_stats(data_dir)

    idf = np.load(idf_loc)
    with open(vocab_loc, "r") as f:
        vocab = json.load(f)

    class MyVectorizer(TfidfVectorizer):#定义一个名为MyVectorizer的内部类，继承自TfidfVectorizer。该类将IDF值数组直接赋给TfidfVectorizer的idf_属性。
        TfidfVectorizer.idf_ = idf

    vec = MyVectorizer()
    vec.vocabulary_ = vocab#将加载的词汇表赋值给向量化器实例的vocabulary_属性。
    vec._tfidf._idf_diag = sp.spdiags(idf, diags=0, m=len(idf), n=len(idf))#使用scipy的spdiags函数创建一个以idf值为对角线元素的稀疏矩阵，然后将其赋值给向量化器的内部_idf_diag属性。这是必要的，因为TF-IDF转换需要用这个矩阵来计算文档的TF-IDF值。

    return vec


def collect_stats(data_dir: str):
    """
    Uses wikipedia snippets to collect statistics over a corpus of English text.#使用维基百科片段收集英文文本语料库的统计数据。
    Retrieved later when computing TF-IDF vectors.稍后在计算 TF-IDF 向量时检索。
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    idf_loc, vocab_loc = data_dir / "idf.npy", data_dir / "tfidf_vocab.json"

    try:
        print(f"Downloading IDF cache from {REMOTE_IDF_URL}")
        torch.hub.download_url_to_file(REMOTE_IDF_URL, idf_loc)
        print(f"Downloading TF-IDF vocab cache from {REMOTE_VOCAB_URL}")
        torch.hub.download_url_to_file(REMOTE_VOCAB_URL, vocab_loc)
        return
    except Exception as e:
        print(f"Error downloading file:", e)
        print("Recomputing TF-IDF stats...")

    snips_list = AttributeSnippets(data_dir).snippets_list
    documents = list(chain(*[[y["text"] for y in x["samples"]] for x in snips_list]))

    vec = TfidfVectorizer()
    vec.fit(documents)#创建TfidfVectorizer对象并使用文档列表训练它，从而计算IDF值和构建词汇表。
    #将计算得到的IDF数组保存为.npy文件，将词汇表保存为.json文件。:
    idfs = vec.idf_
    vocab = vec.vocabulary_

    np.save(data_dir / "idf.npy", idfs)
    with open(data_dir / "tfidf_vocab.json", "w") as f:
        json.dump(vocab, f, indent=1)
