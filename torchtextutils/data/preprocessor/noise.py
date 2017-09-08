import numpy as np


def random_idx(l, prob):
    n = round(l * prob)
    idx = np.random.permutation(l)[:n]

    return idx


class OmissionNoisifier(object):
    def __init__(self, omit_prob, unk_idx):
        self.omit_prob = omit_prob
        self.unk_idx = unk_idx

    def __call__(self, data):
        if self.omit_prob <= 0.0:
            return

        batch, lens = data

        for sent, l in zip(batch, lens):
            omit_inds = random_idx(l - 2, self.omit_prob)

            for i in omit_inds:
                sent[i + 1] = self.unk_idx


class SwapNoisifier(object):
    def __init__(self, swap_prob):
        self.swap_prob = swap_prob

    def __call__(self, data):
        if self.swap_prob <= 0:
            return

        batch, lens = data

        for sent, l in zip(batch, lens):
            swap_inds = random_idx(l - 3, self.swap_prob)

            for i in swap_inds:
                sent[i + 1], sent[i + 2] = sent[i + 2], sent[i + 1]