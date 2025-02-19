import os
import json
import shutil
import glob

# 文件路径
json_file = 'FedEdit/relation_case_id_above_800.json'
source_folder = 'rewriting-konwledge/kvs/EleutherAI_gpt-j-6B_MEMIT'
target_folder = 'FedEdit/mcf_client'

# 读取JSON文件
with open(json_file, 'r') as f:
    data = json.load(f)

# 遍历12个键值对
for idx, (key, values) in enumerate(data.items(), 1):
    # 创建client文件夹
    client_folder = os.path.join(target_folder, f'client{idx}')
    os.makedirs(client_folder, exist_ok=True)
    
    # 遍历对应的值
    for value in values:
        # 使用通配符匹配文件名
        npz_file_pattern = f'mcf_layer_*_case_{value}.npz'
        matched_files = glob.glob(os.path.join(source_folder, npz_file_pattern))
        
        # 如果文件存在，则复制到对应的client文件夹
        if matched_files:
            for file_path in matched_files:
                shutil.copy(file_path, client_folder)
                print(f'文件 {os.path.basename(file_path)} 复制到 {client_folder}')
        else:
            print(f'未找到匹配 case {value} 的文件，模式为 {npz_file_pattern}')

print("操作完成！")