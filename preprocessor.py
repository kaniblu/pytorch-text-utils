import torch
import numpy as np


class Preprocessor(object):
    def __init__(self, vocab, omit_prob=0.0, swap_prob=0.0,
                 add_bos=True, add_eos=True):
        self.vocab = vocab
        self.unk_idx = vocab[vocab.unk]
        self.pad_idx = vocab[vocab.pad]
        self.bos_idx = vocab[vocab.bos]
        self.eos_idx = vocab[vocab.eos]
        self.omit_prob = omit_prob
        self.swap_prob = swap_prob
        self.add_bos = add_bos
        self.add_eos = add_eos

    def _random_idx(self, l, prob):
        n = round(l * prob)
        idx = np.random.permutation(l)[:n]

        return idx

    def add_noise(self, batch, lens):

        for sent, l in zip(batch, lens):
            # Randomly swap words

            if self.swap_prob > 0.0:
                swap_inds = self._random_idx(l - 3, self.swap_prob)

                for i in swap_inds:
                    sent[i + 1], sent[i + 2] = sent[i + 2], sent[i + 1]

            # Randomly omit words
            if self.omit_prob > 0.0:
                omit_inds = self._random_idx(l - 2, self.omit_prob)

                for i in omit_inds:
                    sent[i + 1] = self.unk_idx

    def __call__(self, batch):
        batch = [[self.vocab.f2i[w]
                                   if w in self.vocab.f2i else self.unk_idx
                  for w in words] for words in batch]

        if self.add_bos:
            batch = [[self.bos_idx] + words for words in batch]

        if self.add_eos:
            batch = [words + [self.eos_idx] for words in batch]

        lens = [len(s) for s in batch]
        max_len = max(lens)
        batch = [s + [self.pad_idx] * (max_len - len(s)) for s in batch]

        batch = torch.LongTensor(batch)
        lens = torch.LongTensor(lens)

        return batch, lens