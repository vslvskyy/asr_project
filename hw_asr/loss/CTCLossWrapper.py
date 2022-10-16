import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def forward(self, log_probs, log_probs_length, text_encoded, text_encoded_length,
                **batch) -> Tensor:
        log_probs_t = torch.transpose(log_probs, 0, 1)

        # print(f"log_probs_length: {log_probs_length}")
        # print(f"text_encoded_length: {text_encoded_length}")

        # print(
        #     f'log_probs : {log_probs_t.size()}\n'
        #     f'targets : {text_encoded.size()}\n'
        #     f'input_lengths : {log_probs_length.size()}\n'
        #     f'target_lengths : {text_encoded_length.size()}\n'
        # )

        return super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )
