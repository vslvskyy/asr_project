from pickle import EMPTY_DICT
from typing import List, NamedTuple
from collections import defaultdict
from pyctcdecode import build_ctcdecoder

import torch
import numpy as np

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, use_bs=True, use_bs_lm=True):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        bs_labels = [char.upper() for char in ([''] + vocab[1:])]
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.bs_decoder, self.bs_lm_decoder = None, None
        if use_bs:
            self.bs_decoder = build_ctcdecoder(labels=bs_labels)
        if use_bs_lm:
            self.bs_lm_decoder = build_ctcdecoder(
                        labels=bs_labels,
                        kenlm_model_path="data/lm_models/3-gram.arpa",
                        alpha=0.8,
                        beta=0
                    )



    def ctc_decode(self, inds: List[int]) -> str:
        res_lst: List[str] = []
        last_empty = False

        for ind in inds:
            el = self.ind2char[ind]

            if el == self.EMPTY_TOK:
                last_empty = True

            elif not res_lst or last_empty or res_lst[-1] != el:
                res_lst.append(el)
                last_empty = False

        return "".join(res_lst)


    def ctc_beam_search_lm(self, probs: np.array, beam_size: int = 100):
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        assert self.bs_decoder is not None, "You need to set use_bs=True"
        bs_res = self.bs_decoder.decode(probs, beam_width=beam_size)

        assert self.bs_lm_decoder is not None, "You need to set use_bs_lm=True"
        bs_lm_res = self.bs_lm_decoder.decode(probs, beam_width=beam_size)

        return  bs_res, bs_lm_res


    def extend_merge_and_cut(self, dp: dict, prob: torch.tensor, beam_size: int = 100) -> dict:
        new_dp = defaultdict(float)

        for (text, last_char), p in dp.items():
            for i, char in self.ind2char.items():

                if char != self.EMPTY_TOK and char != last_char:
                    new_key = (text + last_char, char)
                else:
                    new_key = (text, last_char)

                last_char = char
                new_dp[new_key] += p * prob[i]

        return dict(list(sorted(new_dp.items(), key=lambda x: -x[1]))[:beam_size])


    def my_ctc_beam_search(self, probs: torch.tensor, beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        dp = { ("", self.EMPTY_TOK): 1 }
        for i in range(char_length):
            dp = self.extend_merge_and_cut(dp, probs[i, :], beam_size)

        for (text, last_char), p in dp.items():
            hypos.append(Hypothesis((text+last_char).replace(self.EMPTY_TOK, ""), p))
        # return sorted(hypos, key=lambda x: x.prob, reverse=True)
        return hypos
