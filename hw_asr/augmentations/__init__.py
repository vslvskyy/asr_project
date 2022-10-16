from collections import Callable
from typing import List

import hw_asr.augmentations.spectrogram_augmentations
import hw_asr.augmentations.wave_augmentations
from hw_asr.augmentations.sequential import SequentialAugmentation
from hw_asr.utils.parse_config import ConfigParser


def from_configs(configs: ConfigParser):
    wave_augs, wave_augs_probs = [], []
    if "augmentations" in configs.config and "wave" in configs.config["augmentations"]:
        for aug_dict in configs.config["augmentations"]["wave"]:
            wave_augs.append(
                configs.init_obj(aug_dict, hw_asr.augmentations.wave_augmentations)
            )
        wave_augs_probs = configs.config["augmentations"]["wave_probs"]

    spec_augs, spec_augs_probs = [], []
    if "augmentations" in configs.config and "spectrogram" in configs.config["augmentations"]:
        for aug_dict in configs.config["augmentations"]["spectrogram"]:
            spec_augs.append(
                configs.init_obj(aug_dict, hw_asr.augmentations.spectrogram_augmentations)
            )
    spec_augs_probs = configs.config["augmentations"]["spec_probs"]

    return _to_function(wave_augs, wave_augs_probs), _to_function(spec_augs, spec_augs_probs)


def _to_function(augs_list: List[Callable], probs_list: List[float]):
    if len(augs_list) == 0:
        return None
    return SequentialAugmentation(augs_list, probs_list)
