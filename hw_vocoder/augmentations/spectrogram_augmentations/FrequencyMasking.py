import torchaudio.transforms as T
from torch import Tensor
import random

from hw_asr.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = T.FrequencyMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        if random.random() < 0.1:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
        return data
