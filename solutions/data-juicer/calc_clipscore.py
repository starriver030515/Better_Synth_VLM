# flake8: noqa: E501

import os
import unittest
import argparse
import json
from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.mapper.image_text_clipscore_mapper import \
    ImageTextClipscoreMapper
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, SKIPPED_TESTS

def args_parser():
    parser = argparse.ArgumentParser(description="Change format of a file.")
    parser.add_argument('--hf_clip', type=str, help='Path to the Clip model')
    parser.add_argument('--caption_path', type=str, help='Path to the captions and images')
    parser.add_argument('--top_n', type=int, help='top n clipscore caption-image pairs')
    parser.add_argument('--result_path', type=str, help='Result Path to the captions and images')
    parser.add_argument('--ssim_path', type=str,default=None, help='Path to the ssim values')

    return parser.parse_args()

def _run_mapper(dataset: Dataset, op, num_proc=1):
    if Fields.stats not in dataset.features:
        dataset = dataset.add_column(name=Fields.stats, column=[{}] * dataset.num_rows)
    dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True, batch_size = dataset.num_rows)
    res_list = dataset.to_list()
    return res_list

def get_data(caption_path: str):
    json_data = []
    with open(caption_path, 'r') as f:
        for line in f:
            json_data.append(json.loads(line.strip()))

    for index in range(len(json_data)):
        json_data[index]['text'] = json_data[index]['text'][14:-13]
        subdir = json_data[index]['images'][0].rsplit("/", 1)[0]
        if not os.path.exists(os.path.join(os.getcwd(), subdir)):
            os.mkdir(os.path.join(os.getcwd(), subdir))
        
    return json_data

def write_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def modify(res_list, args):
    data = []
    for item in res_list:
        data.append({"id": item['id'],
                    "text": item['text'],
                    "images": item['images'],
                    "clipscore": float(item['__dj__stats__']['image_text_matching_score'])})
        
    if args.ssim_path is not None:
        
        dic=dict()
        with open(args.ssim_path, 'r') as f:
            for line in f:
                item=json.loads(line.strip())
                dic[item['image_path']]=item['ssim']
        for item in data:
            item['ssim'] = dic[item['images'][0][-20:]]
        sorted_data = sorted(data, key=lambda x: x['clipscore']+(x['ssim'])*0.5, reverse=True)
    else:
        sorted_data=sorted(data, key=lambda x: x['clipscore'], reverse=True)
    return sorted_data[:args.top_n]

def main():
    args = args_parser()
    ds_list = get_data(args.caption_path)
    dataset = Dataset.from_list(ds_list)
    op = ImageTextClipscoreMapper(bf_clip=args.hf_clip, any_or_all='any')
    res_list = _run_mapper(dataset, op)
    res_list = modify(res_list, args)
    write_jsonl(res_list, args.result_path)

if __name__ == '__main__':
    main()
