

import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from multiprocessing import Pool, cpu_count
import json
import argparse

def args_parser():
    parser = argparse.ArgumentParser(description="Change format of a file.")
    parser.add_argument('--directory', type=str, help='Path to the directory containing images')
    parser.add_argument('--output_file', type=str, help='Path to the output file')
    return parser.parse_args()

def process_image(image_path):
    try:
        with Image.open(image_path) as img:
            # 将图像降采样为768x768
            img_resized_768 = img.resize((768, 768), Image.BILINEAR)
            
            # 将768x768图像降采样为336x336
            img_resized_336 = img_resized_768.resize((336, 336), Image.BILINEAR)
            
            # 将336x336图像升采样为768x768
            img_resized_768_again = img_resized_336.resize((768, 768), Image.BILINEAR)
            
            # 将图像转换为灰度图以计算SSIM
            img_gray_768 = img_resized_768.convert('L')
            img_gray_768_again = img_resized_768_again.convert('L')
            
            # 将图像转换为numpy数组
            img_array_768 = np.array(img_gray_768)
            img_array_768_again = np.array(img_gray_768_again)
            
            # 计算SSIM
            ssim_index = ssim(img_array_768, img_array_768_again)
            path_spilt = image_path.split("/")
            short_path=path_spilt[-3]="/"+path_spilt[-2]+"/"+path_spilt[-1]
        return {"image_path": short_path, "ssim": ssim_index}
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def main(directory, output_file):
    ssim_values = []
    
    # 获取所有图像文件的路径
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    
    # 使用多进程处理图像
    with Pool(cpu_count()) as pool:
        ssim_values = pool.map(process_image, image_paths)
    
    # 过滤掉处理失败的图像
    ssim_values = [value for value in ssim_values if value is not None]
    
    # 将结果写入JSONL文件
    with open(output_file, 'w') as f:
        for item in ssim_values:
            f.write(json.dumps(item) + '\n')
    
    # 计算SSIM均值和方差
    ssim_scores = [item["ssim"] for item in ssim_values]
    ssim_mean = np.mean(ssim_scores)
    ssim_std = np.std(ssim_scores)
    
    print(f"SSIM均值: {ssim_mean}")
    print(f"SSIM方差: {ssim_std}")

if __name__ == "__main__":
    args = args_parser()
    directory = args.directory
    output_file = args.output_file
    main(directory, output_file)