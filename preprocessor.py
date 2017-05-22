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

    def _add_noise(self, words):
        words = words.copy()

        # Randomly swap words
        num_swaps = round((len(words) - 1) * self.swap_prob)
        swap_inds = np.random.permutation(len(words) - 1)[:num_swaps]

        for i in swap_inds:
            words[i], words[i + 1] = words[i + 1], words[i]

        # Randomly omit words
        num_omits = round(len(words) * self.omit_prob)
        omit_inds = np.random.permutation(len(words))[:num_omits]

        for i in omit_inds:
            words[i] = self.unk_idx

        return words

    def __call__(self, batch, noise=False):
        batch = [[self.bos_idx] + [self.vocab[w] if w in self.vocab else self.unk_idx
                  for w in words] + [self.eos_idx] for words in batch]

        if noise:
            batch = [self._add_noise(words) for words in batch]

        max_len = max(len(s) for s in batch)
        batch = [s + [self.pad_idx] * (max_len - len(s)) for s in batch]

        return batch