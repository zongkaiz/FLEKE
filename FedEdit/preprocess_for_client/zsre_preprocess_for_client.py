from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import shutil
# 设置路径
source_folder = 'rewriting-konwledge/kvs/EleutherAI_gpt-j-6B_MEMIT'
target_folder = 'FedEdit/client_z'

# 加载模型
model = SentenceTransformer('all-MiniLM-L6-v2/all-MiniLM-L6-v2')

# 加载数据
with open('data/zsre_mend_eval.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

src = [entry['src'] for entry in data[:19064]]

# 计算嵌入向量
embeddings = model.encode(src)
similarity_matrix = cosine_similarity(embeddings)

# 使用谱聚类
n_clusters = 8  # 聚类数目
spectral_cluster = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
cluster_labels = spectral_cluster.fit_predict(similarity_matrix)

# 根据聚类结果创建文件夹并复制文件
for i, label in enumerate(cluster_labels):
    client_folder = os.path.join(target_folder, f"client{label + 1}")
    
    # 如果文件夹不存在，则创建
    if not os.path.exists(client_folder):
        os.makedirs(client_folder)
    
    # 根据序号构造源文件路径
    source_file = os.path.join(source_folder, f"zsre_layer_8_clamp_0.75_case_{i}.npz")
    
    # 如果文件存在，进行拷贝操作
    if os.path.exists(source_file):
        target_file = os.path.join(client_folder, f"zsre_layer_8_clamp_0.75_case_{i}.npz")
        shutil.copy(source_file, target_file)
        print(f"已将 {source_file} 拷贝到 {target_file}")
    else:
        print(f"文件 {source_file} 不存在！")
