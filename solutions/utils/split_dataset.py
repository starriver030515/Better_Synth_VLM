import json
import os
import math

# 输入JSONL文件路径
anno_jsonl_path = "../mgm_pretrain_raw.jsonl"
# 统计总行数
total_annotations = 0
with open(anno_jsonl_path, 'r') as f:
    for _ in f:
        total_annotations += 1

num_parts = 8
annotations_per_part = math.ceil(total_annotations / num_parts)

anno_output_dir = "../annotations/"
if not os.path.exists(anno_output_dir):
    os.makedirs(anno_output_dir)

# 逐行读取和分割JSONL文件
with open(anno_jsonl_path, 'r') as f:
    for i in range(num_parts):
        part_anno_jsonl_path = os.path.join(anno_output_dir, f"annotations_part_{i + 1}.jsonl")
        with open(part_anno_jsonl_path, 'w') as part_file:
            for j in range(annotations_per_part):
                line = f.readline()
                if not line:
                    break
                part_file.write(line)
        print(f"Part {i + 1}: {j + 1} annotations")

print("标注已成功分成8份，并保存到文件夹中。")