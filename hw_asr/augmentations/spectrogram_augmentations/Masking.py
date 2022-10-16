from torch import Tensor
from torchaudio.transforms import FrequencyMasking, TimeMasking

from hw_asr.augmentations.base import AugmentationBase


class FreqMask(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.aug = FrequencyMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self.aug(x).squeeze(1).float()


class TimeMask(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.aug = TimeMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self.aug(x).squeeze(1).float()
