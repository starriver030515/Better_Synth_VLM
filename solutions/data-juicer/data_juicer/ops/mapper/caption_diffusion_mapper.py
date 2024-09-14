import copy
import os
from PIL import Image

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.mm_utils import (SpecialTokens, load_data_with_context,
                                        load_image, remove_special_tokens)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'caption_diffusion_mapper'

check_list = ['diffusers', 'torch', 'transformers', 'simhash-pybind']
with AvailabilityChecking(check_list, OP_NAME):
    import diffusers  # noqa: F401
    import simhash  # noqa: F401
    import torch
    import transformers  # noqa: F401

    # avoid hanging when calling stable diffusion in multiprocessing
    torch.set_num_threads(1)


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class CaptionDiffusionMapper(Mapper):
    """
        Generate image by diffusion model
    """

    _accelerator = 'cuda'
    _batched_op = True

    def __init__(self,
                 hf_diffusion: str = 'stabilityai/stable-diffusion-3-medium-diffusers',
                 trust_remote_code=False,
                 torch_dtype: str = 'fp16',
                 revision: str = 'main',
                 strength: float = 0.8,
                 guidance_scale: float = 10.0,
                 max_sequence_length: int = 512,
                 n_steps: int = 60,
                 height: int = 1024,
                 width: int = 1024,
                 aug_num: int = 1,
                 keep_original_sample: bool = False,
                 caption_key: str = None,
                 hf_img2seq='Salesforce/blip2-opt-2.7b',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_diffusion: diffusion model name on huggingface to generate
            the image.
        :param torch_dtype: the floating point type used to load the diffusion
            model. Can be one of ['fp32', 'fp16', 'bf16']
        :param revision: The specific model version to use. It can be a
            branch name, a tag name, a commit id, or any identifier allowed
            by Git.
        :param strength: Indicates extent to transform the reference image.
            Must be between 0 and 1. image is used as a starting point and
            more noise is added the higher the strength. The number of
            denoising steps depends on the amount of noise initially added.
            When strength is 1, added noise is maximum and the denoising
            process runs for the full number of iterations specified in
            num_inference_steps. A value of 1 essentially ignores image.
        :param guidance_scale: A higher guidance scale value encourages the
            model to generate images closely linked to the text prompt at the
            expense of lower image quality. Guidance scale is enabled when
            guidance_scale > 1.
        :param aug_num: The image number to be produced by stable-diffusion
            model.
        :param keep_candidate_mode: retain strategy for the generated
            $caption_num$ candidates.

            'random_any': Retain the random one from generated captions

            'similar_one_simhash': Retain the generated one that is most
                similar to the original caption

            'all': Retain all generated captions by concatenation

        Note:
            This is a batched_OP, whose input and output type are
            both list. Suppose there are $N$ list of input samples, whose batch
            size is $b$, and denote caption_num as $M$.
            The number of total samples after generation is $2Nb$ when
            keep_original_sample is True and $Nb$ when keep_original_sample is
            False. For 'random_any' and 'similar_one_simhash' mode,
            it's $(1+M)Nb$ for 'all' mode when keep_original_sample is True
            and $MNb$ when keep_original_sample is False.

        :param caption_key: the key name of fields in samples to store captions
            for each images. It can be a string if there is only one image in
            each sample. Otherwise, it should be a list. If it's none,
            ImageDiffusionMapper will produce captions for each images.
        :param hf_img2seq: model name on huggingface to generate caption if
            caption_key is None.
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.n_steps = n_steps
        self.height = height
        self.width = width
        self.max_sequence_length = max_sequence_length
        self.aug_num = aug_num
        self.keep_original_sample = keep_original_sample
        self.caption_key = caption_key
        self.prompt = 'A photo of a '
        self.hf_diffusion = hf_diffusion
        if not self.caption_key:
            from .image_captioning_mapper import ImageCaptioningMapper
            self.op_generate_caption = ImageCaptioningMapper(
                hf_img2seq=hf_img2seq,
                keep_original_sample=False,
                prompt=self.prompt)
        self.model_key = prepare_model(
            model_type='diffusion',
            pretrained_model_name_or_path=hf_diffusion,
            diffusion_type='text2image',
            torch_dtype=torch_dtype,
            revision=revision)

        self.diffusion_model = get_model(model_key=self.model_key, rank=None, use_cuda=True)

    def _real_guidance(self, caption: str):
        prompt = caption
        kwargs = dict(prompt=prompt,
                      guidance_scale=self.guidance_scale,
                      num_inference_steps=self.n_steps,
                      max_sequence_length = self.max_sequence_length)

        outputs = self.diffusion_model(**kwargs)
        canvas = outputs.images[0]
        return canvas

    def _process_single_sample(self, ori_sample, rank=None, context=False):
        """
        :param ori_sample: a single data sample before applying generation
        :return: batched results after generation
        """
        # there is no image in this sample
        if self.image_key not in ori_sample or \
                not ori_sample[self.image_key]:
            return []

        loaded_image_keys = ori_sample[self.image_key]
        captions = [ori_sample[self.caption_key]]

        # the generated results
        generated_samples = [
            copy.deepcopy(ori_sample) for _ in range(self.aug_num)
        ]
        for aug_id in range(self.aug_num):
            diffusion_image_keys = []
            for index, value in enumerate(loaded_image_keys):
                diffusion_image = self._real_guidance(captions[index])
                diffusion_image.save(value)
                if context:
                    generated_samples[aug_id][Fields.context][
                        diffusion_image_key] = diffusion_image
            generated_samples[aug_id][self.image_key] = diffusion_image_keys

        return generated_samples

    def process(self, samples, rank=None, context=False):
        """
            Note:
                This is a batched_OP, whose the input and output type are
                both list. Suppose there are $N$ input sample list with batch
                size as $b$, and denote aug_num as $M$.
                the number of total samples after generation is  $(1+M)Nb$.

            :param samples:
            :return:
        """
        # reconstruct samples from "dict of lists" to "list of dicts"
        reconstructed_samples = []
        for i in range(len(samples[self.text_key])):
            reconstructed_samples.append(
                {key: samples[key][i]
                 for key in samples})

        # do generation for each sample within the batch
        samples_after_generation = []
        for ori_sample in reconstructed_samples:
            if self.keep_original_sample:
                samples_after_generation.append(ori_sample)
            generated_samples = self._process_single_sample(ori_sample, rank=rank)
            if len(generated_samples) != 0:
                samples_after_generation.extend(generated_samples)

        # reconstruct samples from "list of dicts" to "dict of lists"
        keys = samples_after_generation[0].keys()
        res_samples = {}
        for key in keys:
            res_samples[key] = [s[key] for s in samples_after_generation]

        return res_samples
