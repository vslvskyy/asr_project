from unicodedata import bidirectional
import torch
from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class DeepSpeechModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.conv_block = Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(41,11), stride=2, padding=(20, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(0, 20),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(21,11), stride=(2,1), padding=(10, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(0, 20)
        )

        n_out_feats = self.len_after_conv(torch.tensor([n_feats]), kernel_size=41, stride=2, pad=20)
        n_out_feats = self.len_after_conv(n_out_feats, kernel_size=21, stride=2, pad=10)

        self.rnn_block = Sequential(
            nn.LSTM(input_size=n_out_feats.item()*32, hidden_size=fc_hidden//2, num_layers=3, bidirectional=True, batch_first=True)
        )
        self.linear_block = Sequential(
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_class)
        )

    def forward(self, spectrogram, **batch):
        outputs = self.conv_block(spectrogram[:, None, :, :])

        lengths_after_conv = self.transform_input_lengths(batch["spectrogram_length"])

        batch_size, ch_n, ftrs_n, time = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = outputs.view(batch_size, time, ch_n * ftrs_n)

        outputs = nn.utils.rnn.pack_padded_sequence(outputs, lengths_after_conv, batch_first=True, enforce_sorted=False)
        outputs = self.rnn_block(outputs)[0]
        outputs = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)[0]

        outputs = self.linear_block(outputs)

        return {"logits": outputs}

    def len_after_conv(self, input_lengths, kernel_size, stride, pad=0, dil=1):
        numerator = input_lengths + 2 * pad - dil * (kernel_size - 1) - 1
        seq_lengths = numerator.float() / float(stride)
        seq_lengths = seq_lengths.int() + 1
        return seq_lengths

    def transform_input_lengths(self, input_lengths):
        res_len = self.len_after_conv(input_lengths=input_lengths, kernel_size=11, stride=2, pad=5)
        res_len = self.len_after_conv(input_lengths=res_len, kernel_size=11, stride=1, pad=5)
        return res_len
