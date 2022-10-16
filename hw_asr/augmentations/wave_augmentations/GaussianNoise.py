from torch import Tensor
from torch import distributions

from hw_asr.augmentations.base import AugmentationBase


class GaussianNoise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.noiser = distributions.Normal(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        x = x + self.noiser.sample(data.size())
        return x.squeeze(1)

