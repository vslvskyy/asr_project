from torch import Tensor
from torchaudio.transforms import TimeStretch

from hw_asr.augmentations.base import AugmentationBase


class UpStretcher(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.stretcher = TimeStretch(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self.stretcher(x, 1.2).squeeze(1).float()


class DownStretcher(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.stretcher = TimeStretch(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self.stretcher(x, 0.9).squeeze(1).float()
