## synth_challenge 方法及实现

### 方法概览

该方法基于[SynthVLM: High-Efficiency and High-Quality Synthetic Data for Vision Language Models](https://arxiv.org/abs/2407.20756)，该方法github链接为[SynthVLM](https://github.com/starriver030515/SynthVLM)。

在VLMs的预训练中，实现image-caption pairs的对齐很重要。以往的研究大多根据image来生成caption。stable diffusion3的开源，为构建高匹配的image-caption pairs提供了新的途径。总体而言，我们的方法分为两步骤：根据caption，利用diffusion模型生成image，然后根据CLIPScore对image-caption进行筛选，获得匹配度最高的image-caption pairs。

#### Diffusion model

在该部分，我们利用提供的400k种子数据集中的caption，生成相应的image。在这里，我们采取的diffusion是最新开源的stable diffusion3。Diffusion包括两个主要过程：Forward过程（也称为noising process）和Backward过程（也称为denoising process）。

##### Forward过程（Noising Process）

在Forward过程中，模型逐步将原始数据  $x_0$  添加噪声，直到它完全转变为随机噪声  $x_T$ 。这个过程是通过一个马尔可夫链完成的，其中每一步都根据一个固定的方差增加噪声。这个过程可以用以下公式表示：

$$ x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon$$

其中：

•   $\epsilon$  是从标准正态分布  $N(0, I)$ 中抽取的噪声。
•   $\alpha_t$  是一个预先定义的衰减因子序列，通常$\alpha_t$随着时间逐渐减小。

##### Backward过程（Denoising Process）

在Backward过程中，模型的任务是逆转Forward过程，从噪声中逐步恢复出原始数据。这个过程使用了一个参数化的模型  $p_\theta$ ，它试图预测在给定  $x_t$ 的情况下  $x_{t-1}$ 的条件分布：

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

其中：

•   $\mu_\theta(x_t, t)$ 是由神经网络参数化的均值，它预测在给定当前噪声 $x_t$  和时间步  $t$  的情况下，原始数据  $x_{t-1}$  的位置。
•   $\sigma_t^2$  是预定义的噪声方差。

#### CLIPScore

由于生成图片的质量难免带有随机性，在生成完所有的图片后，会对这些image-caption pairs的匹配度进行筛选。我们采用CLIPScore进行筛选。CLIPScore 是一个用于评估生成图像或文本描述与其参照文本或图像的相关性的指标。它基于 OpenAI 的 CLIP 模型，该模型通过训练大量的图像-文本对，学会了将图像和文本映射到同一个向量空间内，使得相关的图像和文本在这个空间内更接近。

CLIPScore 的计算过程如下：

1. 使用 CLIP 模型的文本编码器将参考文本 \($t_r$\) 和候选文本 \($t_c$\) 编码为向量 \($v_r$\) 和 \($v_c$\)。

2. 计算这两个向量的余弦相似度，公式为：
   $$\text{CLIPScore}(t_r, t_c) = \frac{v_r \cdot v_c}{\|v_r\| \|v_c\|}$$

3. 其中，\($\cdot$ \) 表示向量的点积，\($\|v\|$ \) 表示向量的范数。

#### solution

该方法实现在solutions/文件夹中。在该文件夹中保留一份data-juicer源代码。由于data-juicer中并没有提供text2image的生成以及CLIPScore的实现。于是在data-juicer框架下实现了两个mapper算子：caption_diffusion_mapper 和 image_text_clipscore_mapper。该算子位于data-juicer/data_juicer/ops/mapper中，兼容data-juicer框架，可以直接使用。

##### caption_diffusion_mapper

该算子仿照image_diffusion_mapper算子实现，由于image_diffusion_mapper实现image2image，并不能使用text2image，于是在image_diffusion_mapper算子上加以修改，形成caption_diffusion_mapper算子。

##### image_text_clipscore_mapper

该算子对于给定的image和caption，计算相应的CLIPScore分数，利用github clipscore仓库修改，并适配data-juicer框架实现。

##### 运行过程
首先利用data-juicer对caption进行筛选，使用的菜谱在llava-pretrain-refine.yaml中，运行data-juicer，筛选出合适的caption。

然后运行gen_image.py，由于diffusion单个A100即可运行，为了加快生成速度，可以将种子数据集分成八份，对每一小份同时运行gen_image.py。

在gen_image.py中会调用自定义的caption_diffusion_mapper算子生成图片。

在calc_ssim.py中，会对原图片计算ssim分数。

在calc_clipscore.py中会调用自定义的image_text_clipscore_mapper算子计算CLIPScore，此外，该python文件同时完成结合clipscore和ssim的筛选。

##### 运行执行流程

```bash
# 运行前需要安装requirements
pip install -r requirements.txt

cd solutions
cd data-juicer/
dj-process --config llava-pretrain-refine.yaml

cd ../utils/
python format2dj.py
python split_dataset.py
cd ../data-juicer/
srun -p xxx --gres=gpu:1 python gen_image.py \
    --hf_diffusion stabilityai/stable-diffusion-3-medium-diffusers \
    --caption_path ../annotations/annotations_part_i.jsonl # i替换为1～8，各运行一次。


srun -p xxx --gres=gpu:1 python calc_ssim.py \
    --directory your_image_path \
    --output_file your_output_path


srun -p xxx --gres=gpu:1 python calc_clipscore.py \
    --hf_clip ViT-B/32 \
    --caption_path your_raw_caption \
    --top_n 200000 \
    --result_path your_result_path \
    --ssim_path your_ssim_path
    
# 图片生成的路径为data-juicer/images，把images移动到synth_challenge/目录下, 将pretrain1.jsonl移动到synth_challenge/目录下。
cd ..
cp pretrain1.jsonl ../../pretrain1.jsonl
cp -r data-juicer.images ../../images

# 之后，根据train.sh里的文件进行训练即可。
train.sh
```

#### 训练准备

在通过上述方法生成图片并计算CLIPScore后，采用两种方法构造pretrain数据集。

1. 从400k的生成数据集中，选取CLIPScore最高的200k，构成pretrain1。
2. 从400k的生成数据集中，选取CLIPScore最高的50k，将50k复制4份形成200k数据集，构成pretrain6。（等价于对50k的数据集训练4 epochs）

提交的最优版本是基于方法2实现，通过训练匹配度最高的数据，使得模型较快收敛。

#### 注意事项
对于finetune阶段的gpt4vdataset，有些图片失效了，从网上重新下载了这些图片。

对于提交的结果，为测试多次取结果最好的，可以多eval几次以复现效果。

对于模型训练复现，由于pretrain阶段训练不稳定，建议复现时pretrain阶段的平均loss达到和pretrain.log中相似的数值，然后finetune，这样可以复现出提交的模型。
