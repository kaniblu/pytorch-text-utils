import torch
import numpy as np

from .ipo import ImmutablePropertiesObject


class Preprocessor(ImmutablePropertiesObject):
    def __init__(self, vocab, omit_prob=0, swap_prob=0):
        super(Preprocessor, self).__init__(
            vocab=vocab,
            unk_idx=vocab[vocab.unk],
            pad_idx=vocab[vocab.pad],
            bos_idx=vocab[vocab.bos],
            eos_idx=vocab[vocab.eos],
            omit_prob=omit_prob,
            swap_prob=swap_prob
        )

    def add_noise(self, batch, lens):

        for sent, l in zip(batch, lens):
            # Randomly swap words
            swaps = np.random.choice(2, l - 3,
                                     p=[self.swap_prob, 1 - self.swap_prob])
            swap_inds = np.where(swaps == 0)[0]

            for i in swap_inds:
                sent[i + 1], sent[i + 2] = sent[i + 2], sent[i + 1]

            # Randomly omit words
            omits = np.random.choice(2, l - 2,
                                     p=[self.omit_prob, 1 - self.omit_prob])
            omit_inds = np.where(omits == 0)[0]

            for i in omit_inds:
                sent[i + 1] = self.unk_idx

    def __call__(self, batch):
        batch = [[self.bos_idx] + [self.vocab[w] if w in self.vocab else self.unk_idx
                  for w in words] + [self.eos_idx] for words in batch]

        lens = [len(s) for s in batch]
        max_len = max(lens)
        batch = [s + [self.pad_idx] * (max_len - len(s)) for s in batch]

        batch = torch.LongTensor(batch)
        lens = torch.LongTensor(lens)

        return batch, lens