# FLEKE
- Code for [``FLEKE: Federated Locate-then-Edit Knowledge Editing for Multi-Client Collaboration``](https://arxiv.org/abs/2502.15677)
## Table of Contents
- [FLEKE](#FLEKE)
- [Requirements](#Requirements)
- [Data](#Data)
- [Quick Start](#Quick-Start)
  - [An example for editing GPT-J (6B) on counterfact dataset using FedMEMIT](#An-example-for-editing-GPT-J-6B-on-counterfact-dataset-using-FedMEMIT)
- [Acknowledgment](#Acknowledgment)
## Requirements
- At least one A100/A800 80G GPU and another GPU with no less than 24G memory.
- Environment
    ``` bash
    conda create -n pmet python=3.10
    pip install -r requirements.txt
    ```
## Data

For all raw datasets, you can get them from [https://github.com/kmeng01/memit](https://github.com/kmeng01/memit).

### For zsRE dataset:
You can use our code in `FedEdit\preprocess_for_client\zsre_preprocess_for_client.py` for spectral clustering to classify the raw dataset according to the number of clients you need.

### For COUNTFACT dataset:
You can first use our code in `FedEdit\preprocess_for_client\mcf_preprocess_for_client\find_relation_id.py` to filter out relation_ids with more than 800 case_ids. Then, use `FLEKE_code\FedEdit\preprocess_for_client\mcf_preprocess_for_client\accrording_mcfJson800_set_client.py` to organize and classify the z vectors corresponding to the case_ids.
## Quick Start
### An example for editing GPT-J (6B) on counterfact dataset using FedMEMIT, where the number of cilents is 8, the number of time slots is 10 and the similarity threshold is 0.4.
#### 1. FedMEMIT
 
    python evaluate.py --model_path [your model path] --model_name EleutherAI/gpt-j-6B --alg_name PMET --hparams_fname EleutherAI_gpt-j-6B.json --ds_name mcf --T [10] --similarity_threshold [0.4] --num_clients [8] --clients_dir [your clients' data path]


 
#### 2. Summarize the results

    python summarize.py --dir_name=MEMIT --runs=run_<run1>/client1

## Acknowledgment
Our code is based on  [``MEMIT``](https://github.com/kmeng01/memit.git) and [``PMET``](https://github.com/xpq-tech/PMET).

## Citation

Zongkai Zhao, Guozeng Xu, Xiuhua Li, Kaiwen Wei, Jiang Zhong.
FLEKE: Federated Locate-then-Edit Knowledge Editing.
arXiv preprint arXiv:2502.15677 (2025).

```
@article{zhao2025fleke,
  title={FLEKE: Federated Locate-then-Edit Knowledge Editing},
  author={Zhao, Zongkai and Xu, Guozeng and Li, Xiuhua and Wei, Kaiwen and Zhong, Jiang},
  journal={arXiv preprint arXiv:2502.15677},
  year={2025}
}
```
