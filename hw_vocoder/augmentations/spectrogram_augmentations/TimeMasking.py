import torchaudio.transforms as T
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = T.TimeMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(2)
        return self._aug(x).squeeze(2)
