#最开始的根据relation_id找对应的case_id,先运行这个得到relation_case_id_mapping.json
# import json
#
## 读取文件
#file_path = 'data/multi_counterfact.json'
#with open(file_path, 'r') as file:
#    data = json.load(file)
#
## 用来统计 relation_id 的字典
#relation_dict = {}
#
## 遍历数据并根据 relation_id 统计 case_id
#for entry in data:
#    relation_id = entry['requested_rewrite']['relation_id']
#    case_id = entry['case_id']
#    
#    if relation_id not in relation_dict:
#        relation_dict[relation_id] = []
#    
#    relation_dict[relation_id].append(case_id)
#
## 输出为 JSON 文件
#output_path = 'FedEdit/relation_case_id_mapping.json'
#with open(output_path, 'w') as output_file:
#    json.dump(relation_dict, output_file, indent=4)
#
#print(f"统计结果已保存到 {output_path}")

#筛选出800个case_id以上的：
import json

# 读取已生成的 JSON 文件
input_path = 'FedEdit/relation_case_id_mapping.json'
with open(input_path, 'r') as file:
    relation_dict = json.load(file)

# 提取出包含超过 800 个 case_id 的 relation_id
filtered_relation_dict = {relation_id: case_ids for relation_id, case_ids in relation_dict.items() if len(case_ids) > 800}

# 如果有满足条件的 relation_id，将它们保存为新的 JSON 文件
if filtered_relation_dict:
    output_path = '/relation_case_id_above_800.json'
    with open(output_path, 'w') as output_file:
        json.dump(filtered_relation_dict, output_file, indent=4)
    print(f"包含超过 800 个 case_id 的 relation_id 已保存到 {output_path}")
else:
    print("没有包含超过 800 个 case_id 的 relation_id。")