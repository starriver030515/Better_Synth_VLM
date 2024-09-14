import os
import json
import shutil
import unittest
import argparse
from data_juicer import _cuda_device_count
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.caption_diffusion_mapper import CaptionDiffusionMapper


def args_parser():
    parser = argparse.ArgumentParser(description="Change format of a file.")
    parser.add_argument('--hf_diffusion', type=str, help='Path to the diffusion model')
    parser.add_argument('--caption_path', type=str, help='Path to the captions and images')
    return parser.parse_args()

def _run_mapper(dataset: Dataset, op, num_proc=1, total_num=1):
    dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True, batch_size = dataset.num_rows)

def get_data(caption_path: str):
    json_data = []
    with open(caption_path, 'r') as f:
        for line in f:
            json_data.append(json.loads(line))

    for index in range(len(json_data)):
        json_data[index]['text'] = json_data[index]['text'][14:-13]
        subdir = json_data[index]['images'][0].rsplit("/", 1)[0]
        if not os.path.exists(os.path.join(os.getcwd(), subdir)):
            os.mkdir(os.path.join(os.getcwd(), subdir))
        
    return json_data

def main():
    args = args_parser()
    ds_list = get_data(args.caption_path)
    dataset = Dataset.from_list(ds_list)
    op = CaptionDiffusionMapper(hf_diffusion=args.hf_diffusion, aug_num=1, keep_original_sample=False, caption_key='text')
    _run_mapper(dataset, op, total_num=len(ds_list))

if __name__ == '__main__':
    main()