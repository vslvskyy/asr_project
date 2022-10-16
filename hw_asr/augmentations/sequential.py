from typing import List, Callable

from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class SequentialAugmentation(AugmentationBase):
    def __init__(self, augmentation_list: List[Callable], probs_list: List[float]):
        assert len(augmentation_list) <= len(probs_list), "Not enough aug probabilities"
        self.augmentation_list = augmentation_list
        self.probs_list = probs_list

    def __call__(self, data: Tensor) -> Tensor:
        x = data
        for augmentation, p in zip(self.augmentation_list, self.probs_list):
            rand_apply = RandomApply(augmentation, p)
            x = rand_apply(x)
        return x
