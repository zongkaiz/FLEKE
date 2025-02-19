# FedLEKE
- Code for [``FedLEKE: Federated Locate-then-Edit Knowledge Editing for Multi-Client Collaboration``](https://arxiv.org网址到时放到这里)

## Requirements
- At least one A100/A800 80G GPU and another GPU with no less than 24G memory.
- Environment
    ``` bash
    conda create -n pmet python=3.10
    pip install -r requirements.txt
    ```

## Quick Start
### An example for editing GPT-J (6B) on counterfact dataset using FedMEMIT, where the number of cilents is 8, the number of time slots is 10 and the similarity threshold is 0.4.
#### 1. FedMEMIT
 
    python evaluate.py --model_path [your model path] --model_name EleutherAI/gpt-j-6B --alg_name PMET --hparams_fname EleutherAI_gpt-j-6B.json --ds_name mcf --T [10] --similarity_threshold [0.4] --num_clients [8] --clients_dir [your clients' data path]


 
#### 2. Summarize the results

    python summarize.py --dir_name=MEMIT --runs=run_<run1>/client1

## Acknowledgment
Our code is based on  [``MEMIT``](https://github.com/kmeng01/memit.git) and [``PMET``](https://github.com/xpq-tech/PMET).