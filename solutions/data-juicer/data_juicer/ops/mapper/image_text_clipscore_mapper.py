import numpy as np
import clip
import torch
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import tqdm
import sklearn.preprocessing
import collections
from packaging import version
import warnings
from jsonargparse.typing import ClosedUnitInterval
from PIL import ImageOps, Image

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import (SpecialTokens, load_data_with_context,
                                        load_image, remove_special_tokens)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'image_text_clipscore_mapper'

with AvailabilityChecking(['torch', 'transformers'], OP_NAME):
    import torch
    import transformers  # noqa: F401

    # avoid hanging when calling blip in multiprocessing
    # torch.set_num_threads(1)

class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix="A photo depicts"):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != " ":
            self.prefix += " "

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {"caption": c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose(
            [
                Resize(n_px, interpolation=Image.BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {"image": image}

    def __len__(self):
        return len(self.data)

@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageTextClipscoreMapper(Mapper):
    """Filter to keep samples those matching score between image and text
    within a specific range."""

    _accelerator = 'cuda'
    _batched_op = True

    def __init__(self,
                 bf_clip='ViT-B/32',
                 trust_remote_code=False,
                 any_or_all: str = 'any',
                 reduce_mode: str = 'avg',
                 top_n: int = 200000,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_clip: clip model name on huggingface to compute
            the matching score between image and text.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param reduce_mode: reduce mode when one text corresponds to
            multiple images in a chunk.
            'avg': Take the average of multiple values
            'max': Take the max of multiple values
            'min': Take the min of multiple values
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.bf_clip = bf_clip
        self.top_n = top_n
        self.any = (any_or_all == 'any')
        self.reduce_mode = reduce_mode

    def extract_all_captions(self, captions, model, device, batch_size=256, num_workers=8):
        data = torch.utils.data.DataLoader(
            CLIPCapDataset(captions),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        all_text_features = []
        with torch.no_grad():
            for b in tqdm.tqdm(data):
                b = b["caption"].to(device)
                all_text_features.append(model.encode_text(b).cpu().numpy())
        all_text_features = np.vstack(all_text_features)
        return all_text_features


    def extract_all_images(self, images, model, device, batch_size=64, num_workers=8):
        data = torch.utils.data.DataLoader(
            CLIPImageDataset(images),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        all_image_features = []
        with torch.no_grad():
            for b in tqdm.tqdm(data):
                b = b["image"].to(device)
                if device == "cuda":
                    b = b.to(torch.float16)
                all_image_features.append(model.encode_image(b).cpu().numpy())
        all_image_features = np.vstack(all_image_features)
        return all_image_features
    
    def get_clip_score(self, model, images, candidates, device, w=2.5):
        """
        get standard image-text clipscore.
        images can either be:
        - a list of strings specifying filepaths for images
        - a precomputed, ordered matrix of image features
        """
        if isinstance(images, list):
            # need to extract image features
            images = self.extract_all_images(images, model, device)

        candidates = self.extract_all_captions(candidates, model, device)

        # as of numpy 1.21, normalize doesn't work properly for float16
        if version.parse(np.__version__) < version.parse("1.21"):
            images = sklearn.preprocessing.normalize(images, axis=1)
            candidates = sklearn.preprocessing.normalize(candidates, axis=1)
        else:
            warnings.warn(
                "due to a numerical instability, new numpy normalization is slightly different than paper results. "
                "to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3."
            )
            images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
            candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

        per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
        return np.mean(per), per, candidates

    def compute_stats(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.image_text_matching_score in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][
                StatsKeys.image_text_matching_score] = np.array(
                    [], dtype=np.float64)
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        loaded_image_keys = [image[0] for image in loaded_image_keys]
        text = sample[self.text_key]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, transform = clip.load(self.bf_clip, device=device, jit=False)
        model.eval()
        
        _, per_instance_image_text, candidate_feats = self.get_clip_score(
            model, loaded_image_keys, text, device
        )
        for stat, score in zip(sample[Fields.stats], per_instance_image_text):
            stat[StatsKeys.image_text_matching_score] = score

        return sample

    def process(self, sample, rank=None):
        return self.compute_stats(sample)

        # itm_scores = sample[Fields.stats][StatsKeys.image_text_matching_score]
        # if len(itm_scores) <= 0:
        #     return True

        # indices_sorted = np.argsort(itm_scores)[::-1]
        # top_indices = indices_sorted[:self.top_n] 

        # keep_bools = np.zeros(len(itm_scores), dtype=bool)
        # keep_bools[top_indices] = True
        # return True
        # # different strategies
        # if self.any:
        #     return keep_bools.any()
        # else:
        #     return keep_bools.all()
