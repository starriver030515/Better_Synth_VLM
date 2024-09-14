
from PIL import Image
import os
import json


json_filename = "../../../input/pretrain_stage_1/mgm_pretrain_stage_1.jsonl"
target_path = "../mgm_pretrain_raw.jsonl"
json_data = []
with open(json_filename, 'r') as f:
    for line in f:
        json_data.append(json.loads(line.strip()))
    
new_data = []
start_index = 0
for index, item in enumerate(json_data):
    print(index)
    folder_index = start_index + (index // 10000)
    target_subfolder = f"{folder_index:05d}"
    target_image_name = f"{folder_index:05d}{index % 10000:04d}.jpg"
    item["images"] = ["images/" + os.path.join(target_subfolder, target_image_name)]
    new_data.append(item)

with open(target_path, 'w') as f:
    for entry in new_data:
            f.write(json.dumps(entry) + '\n')
    
print("所有图像已成功生成并保存。")