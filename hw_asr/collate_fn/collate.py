import logging
from typing import List, Any
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = defaultdict(list)

    for item in dataset_items:
        spectrogram = item["spectrogram"].squeeze().transpose(1,0)
        result_batch["spectrogram"].append(spectrogram)
        result_batch["spectrogram_length"].append(spectrogram.shape[0])

        encoded_text = item["text_encoded"].squeeze()
        result_batch["text_encoded"].append(encoded_text)
        result_batch["text_encoded_length"].append(len(encoded_text))

        result_batch["text"].append(item["text"])

        result_batch["audio_path"] = item["audio_path"]

    result_batch["spectrogram"] = pad_sequence(result_batch["spectrogram"], batch_first=True).transpose(2, 1)
    result_batch["spectrogram_length"] = torch.tensor(result_batch["spectrogram_length"])
    result_batch["text_encoded"] = pad_sequence(result_batch["text_encoded"], batch_first=True)
    result_batch["text_encoded_length"] = torch.tensor(result_batch["text_encoded_length"])

    return result_batch
